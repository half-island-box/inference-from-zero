import atexit
from dataclasses import dataclass, fields
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


@dataclass(slots=True)
class RequestRecord:
    """一个请求走到终态时的收尾记录（只在出队并释放的那一步生成一次）。"""

    seq_id: int
    num_prompt_tokens: int
    num_completion_tokens: int
    finish_reason: str | None


@dataclass(slots=True)
class StepRecord:
    """一步调度的记录；空步（只出队/释放）不记。"""

    index: int
    is_prefill: bool
    seq_ids: list[int]
    num_tokens: int
    finished: list[int]   # 本步 postprocess 判定成终态的 seq_id


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

        # 观测记录：step_records 每步攒一条（空步不记），request_records 每个请求一条。
        # 长跑会一直累积，bench / 服务侧需要时自己 clear()。
        self.step_index = 0
        self.step_records: list[StepRecord] = []
        self.request_records: list[RequestRecord] = []
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

    def abort_request(self, seq_ids: int | list[int]):
        """取消 waiting / running 里的请求。

        只把状态打成 FINISHED_ABORTED，不动队列、不释放 block；
        下一次 step 会把它从队列里出队、写完 RequestRecord、再统一归还 block。
        """
        with prof.phase("generate/abort_request"):
            self.scheduler.abort(seq_ids)

    # 核心方法，进行一轮完整调度：
    # 1) scheduler 判定哪些序列终态并出队（不释放 block）
    # 2) model runner 对活着的序列做 prefill/decode 并采样
    # 3) postprocess 只追加 token、只改状态
    # 4) 等 run 返回之后，才释放本步出队的终态序列的 block（ZOS 下 GPU 可能还在读）
    def step(self):
        with prof.phase("step/schedule"):
            seqs, is_prefill = self.scheduler.schedule()
            num_tokens = (sum(seq.num_scheduled_tokens for seq in seqs)
                          if is_prefill else -len(seqs))
            # 先取走，避免 run 期间（将来 ZOS 下）再有人往 finished 里加东西
            finished = self.scheduler.finished
            self.scheduler.finished = []

        if seqs:
            tag = "prefill" if is_prefill else "decode"
            with prof.phase(f"step/model_run:{tag}"):
                sampled_token_ids = self.model_runner.call("run", seqs, is_prefill)

            with prof.phase("step/postprocess"):
                self.scheduler.postprocess(seqs, sampled_token_ids, is_prefill)
                self.step_records.append(StepRecord(
                    index=self.step_index,
                    is_prefill=is_prefill,
                    seq_ids=[seq.seq_id for seq in seqs],
                    num_tokens=num_tokens,
                    finished=[seq.seq_id for seq in seqs if seq.is_finished],
                ))

        # 统一出口：所有终态都在这里归还 block、写 RequestRecord、交给调用方
        outputs = []
        for seq in finished:
            completion_token_ids = seq.completion_token_ids
            self.scheduler.block_manager.deallocate(seq)
            self.request_records.append(RequestRecord(
                seq_id=seq.seq_id,
                num_prompt_tokens=seq.num_prompt_tokens,
                num_completion_tokens=len(completion_token_ids),
                finish_reason=seq.finish_reason,
            ))
            outputs.append((seq.seq_id, completion_token_ids, seq.finish_reason))

        self.step_index += 1
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
                elif num_tokens < 0:
                    decode_throughout = -num_tokens / (perf_counter() - t)
                pbar.set_postfix({
                    "Prefill": f"{int(prefill_throughout)}tok/s",
                    "Decode": f"{int(decode_throughout)}tok/s",
                })
                for seq_id, token_ids, finish_reason in output:
                    outputs[seq_id] = (token_ids, finish_reason)
                    pbar.update(1)

        pbar.close()
        ordered = [outputs[seq_id] for seq_id in sorted(outputs.keys())]
        with prof.phase("generate/decode_text"):
            outputs = [
                {
                    "text": self.tokenizer.decode(token_ids),
                    "token_ids": token_ids,
                    "finish_reason": finish_reason,
                }
                for token_ids, finish_reason in ordered
            ]
        return outputs
 