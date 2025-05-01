import torch
import torch.nn as nn


class SimAM(torch.nn.Module):
    def __init__(self, channels = None, e_lambda = 1e-4):
        super(SimAM, self).__init__()

        self.activaton = nn.Sigmoid()
        self.e_lambda = e_lambda

    def __repr__(self):
        s = self.__class__.__name__ + '('
        s += ('lambda=%f)' % self.e_lambda)
        return s

    @staticmethod
    def get_module_name():
        return "SimAM"

    def forward(self, x):
        # x: (B, C, H, W)
        B, C, H, W = x.shape

        # Mean over spatial dimensions (H, W)
        mean = x.mean(dim=[2, 3], keepdim=True)  # (B, C, 1, 1)

        # Variance over spatial dims
        var = ((x - mean) ** 2).mean(dim=[2, 3], keepdim=True)  # (B, C, 1, 1)

        # Attention map (no loops, all broadcasted)
        attention = (x - mean) ** 2 / (4 * (var + self.e_lambda)) + 0.5
        return x * torch.sigmoid(attention)