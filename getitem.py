
import torch
import torch.nn as nn

class GetItem(nn.Module):
    """Return x[index]. Helps us split multiple BiFPN outputs into separate lines."""
    def __init__(self, index=0):
        super().__init__()
        self.index = index

    def forward(self, x):
        return x[self.index]
