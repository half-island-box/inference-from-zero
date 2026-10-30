import torch
from torch import nn

class Sampler(nn.Module):

    # 让 PyTorch 2.x 把它编译成融合 kernel。
    # softmax、exponential_、div_、argmax 这一串本来要多次读写显存，编译后能融合减少访存
    @torch.compile
    def forward(self, logits: torch.Tensor, temperatures: torch.Tensor):
        logits = logits.float().div_(temperatures.unsqueeze(dim=1))
        probs = torch.softmax(logits, dim=-1)
        sample_tokens = probs.div_(torch.empty_like(probs).exponential_(1).clamp_min(1e-10)).argmax(dim=-1)
        return sample_tokens