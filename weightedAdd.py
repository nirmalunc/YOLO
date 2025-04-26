import torch
import torch.nn as nn
import torch.nn.functional as F

class WeightedAdd(nn.Module):
    def __init__(self, n_inputs: int, eps: float = 1e-4):
        super().__init__()
        self.eps = eps
        self.w   = nn.Parameter(torch.ones(n_inputs, dtype=torch.float32))

    def forward(self, *xs):
        # If Ultralytics bundled the tensors in a single list, unwrap it
        if len(xs) == 1 and isinstance(xs[0], (list, tuple)):
            xs = xs[0]

        assert len(xs) == self.w.numel(), "WeightedAdd: input count mismatch"

        w = F.relu(self.w)
        return sum(w[i] * x for i, x in enumerate(xs)) / (w.sum() + self.eps)