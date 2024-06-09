import torch.nn.functional as F
import math
import torch
import time

from NeuFlow import utils


class Matching:

    def init_grid(self, batch_size, height, width):
        self.grid = utils.coords_grid(batch_size, height, width)  # [B, 2, H, W]
        self.flatten_grid = self.grid.view(batch_size, 2, -1).permute(0, 2, 1)  # [B, H*W, 2]

    def global_correlation_softmax(self, feature0, feature1):

        b, c, h, w = feature0.shape

        feature0 = feature0.flatten(-2).permute(0, 2, 1)
        feature1 = feature1.flatten(-2).permute(0, 2, 1)

        # correspondence = F.scaled_dot_product_attention(feature0, feature1, self.flatten_grid)

        # correspondence = correspondence.view(b, h, w, 2).permute(0, 3, 1, 2)  # [B, 2, H, W]

        # flow = correspondence - self.grid

        corr = feature0 @ feature1.transpose(-2, -1) / math.sqrt(c)
        corr_conf = F.softmax(corr, dim=-1, dtype=torch.half) * F.softmax(corr, dim=-2, dtype=torch.half)

        corr_conf, correspondence = corr_conf.max(dim=-1)

        correspondence = correspondence.view(b, h, w)

        flow = torch.stack((correspondence % w, correspondence // w), dim=1) - self.grid

        return flow, corr_conf.view(b, 1, h, w)
