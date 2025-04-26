import torch.nn as nn

class DWConv(nn.Module):
    """
    Depth-wise separable 3×3 convolution:
        DW → BN → SiLU → PW(1×1) → BN → SiLU
    Keeps channel count unchanged.
    """
    def __init__(self, c1, c2, k=3, s=1):      # c1 = input, c2 = output
        super().__init__()
        assert c1 == c2, "BiFPN DWConv expects same in/out channels"
        p = k // 2
        self.dw = nn.Conv2d(c1, c1, k, s, p, groups=c1, bias=False)
        self.bn1 = nn.BatchNorm2d(c1, eps=1e-3, momentum=0.03)
        self.pw  = nn.Conv2d(c1, c2, 1, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(c2, eps=1e-3, momentum=0.03)
        self.act = nn.SiLU(inplace=True)
    def forward(self, x):
        x = self.act(self.bn1(self.dw(x)))
        return self.act(self.bn2(self.pw(x)))
