import atexit
from dataclasses import fields
from time import perf_counter
from tqdm.auto import tqdm
from transformers import AutoTokenizer
import torch.multiprocessing as mp

from ..config import Config
from .sequence import Sequence
from .model_runner import ModelRunner
from ..sampling_params import SamplingParams
from .scheduler import Scheduler
from ..utils import profiler as prof

class LLMEngine:

    def __init__(self, model, **kwargs):
        config_fields = {field.name for field in fields(Config)}
        config_kwargs = {k: v for k, v in kwargs.items() if k in config_fields}
        config = Config(model, **config_kwargs)
        Sequence.block_size = config.kvcache_block_size

        # 子进程和它们的 Event 统一放在这里，单卡时保持为空
        self.ps = []
        self.events = []

        # 按卡数选择初始化方式
        if config.tensor_parallel_size == 1:
            self._init_single_gpu(config)
        else:
            self._init_tensor_parallel(config)

        self.tokenizer = AutoTokenizer.from_pretrained(config.model, use_fast=True)
        config.eos = self.tokenizer.eos_token_id
        self.scheduler = Scheduler(config)
        atexit.register(self.exit)

    def _init_single_gpu(self, config):
        # 单卡初始化：不spawn子进程、不建进程组，
        # 唯一的 ModelRunner（rank 0）直接跑在引擎进程里
        self.model_runner = ModelRunner(config, 0, self.events)

    def _init_tensor_parallel(self, config):
        # TP初始化：引擎进程担任 rank 0，spawn出 rank 1..N-1 的 worker 进程
        ctx = mp.get_context("spawn") # 一个spawn启动的上下文
        # 创建TP进程组
        for i in range(1, config.tensor_parallel_size):
            event = ctx.Event()
            process = ctx.Process(target=ModelRunner, args=(config, i ,event))
            process.start()
            self.ps.append(process)
            self.events.append(event)

    def exit(self):
        self.model_runner.call("exit")
        del self.model_runner
        for p in self.ps:
            p.join()

    # prompt转化为seq，并且加入调度器
    def add_request(self, prompt: str | list[int], sampling_params: SamplingParams):
        with prof.phase("generate/add_request"):
            if isinstance(prompt, str):
                token_ids = self.tokenizer.encode(prompt)
            else:
                token_ids = prompt
            seq = Sequence(token_ids, sampling_params)
            self.scheduler.add(seq)

    # 核心方法，进行一轮完整调度：
    # 调度器拿出一组seq
    # model runner将该seqs给模型进行一次prefill/decode 采样后返回每个seq的新token_id
    # 调度器后处理：将每个seq的新tokenid拼接到末尾
    def step(self):
        with prof.phase("step/schedule"):
            seqs, is_prefill = self.scheduler.schedule()
            num_tokens = sum(seq.num_scheduled_tokens for seq in seqs) if is_prefill else -len(seqs)

        tag = "prefill" if is_prefill else "decode"
        with prof.phase(f"step/model_run:{tag}"):
            token_ids = self.model_runner.call("run", seqs, is_prefill)

        with prof.phase("step/postprocess"):
            self.scheduler.postprocess(seqs, token_ids, is_prefill)
            outputs = [(seq.seq_id, seq.completion_token_ids) for seq in seqs if seq.is_finished]
        return outputs, num_tokens


    def is_finished(self):
        return self.scheduler.is_finished()

    def generate(self,
                 prompts: list[str] | list[list[int]],
                 sampling_params: SamplingParams | list[SamplingParams],
                 use_tqdm: bool = True) -> list[str]:
        pbar = tqdm(total = len(prompts), desc="Generating", dynamic_ncols=True, disable=not use_tqdm)

        if not isinstance(sampling_params, list):
            sampling_params = [sampling_params] * len(prompts)

        for prompt, sp in zip(prompts, sampling_params):
            self.add_request(prompt, sp)

        outputs = {}
        prefill_throughout = decode_throughout = 0
        with prof.nvtx("generate/loop"):
            while not self.is_finished():
                t = perf_counter()
                output, num_tokens = self.step()

                if num_tokens > 0:
                    prefill_throughout = num_tokens / (perf_counter() - t)
                else:
                    decode_throughout = -num_tokens / (perf_counter() - t)
                pbar.set_postfix({
                    "Prefill": f"{int(prefill_throughout)}tok/s",
                    "Decode": f"{int(decode_throughout)}tok/s",
                })
                for seq_id, token_ids in output:
                    outputs[seq_id] = token_ids
                    pbar.update(1)

        pbar.close()
        outputs = [outputs[seq_id] for seq_id in sorted(outputs.keys())]
        with prof.phase("generate/decode_text"):
            outputs = [{"text": self.tokenizer.decode(token_ids), "token_ids": token_ids} for token_ids in outputs]
        return outputs
 