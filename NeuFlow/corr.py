import torch
import torch.nn.functional as F
import math

from NeuFlow import utils


def bilinear_sample(img, coords):
    """ Wrapper for grid_sample, uses pixel coordinates """
    H, W = img.shape[-2:]
    xgrid, ygrid = coords.split([1,1], dim=-1)
    xgrid = 2*xgrid/(W-1) - 1
    ygrid = 2*ygrid/(H-1) - 1

    grid = torch.cat([xgrid, ygrid], dim=-1)

    with torch.backends.cudnn.flags(enabled=False):
        img = F.grid_sample(img, grid, align_corners=True)

    return img


class CorrBlock:
    def __init__(self, radius, levels):

        self.radius = radius
        self.levels = levels

    def init_bhwd(self, batch_size, height, width, device, amp):

        xy_range = torch.linspace(-self.radius, self.radius, 2*self.radius+1, dtype=torch.half if amp else torch.float, device=device)

        delta = torch.stack(torch.meshgrid(xy_range, xy_range, indexing='ij'), axis=-1)
        delta = delta.view(1, 2*self.radius+1, 2*self.radius+1, 2)

        self.grid = utils.coords_grid(batch_size, height, width, device, amp)
        self.delta = delta.repeat(batch_size * height * width, 1, 1, 1)

    def __call__(self, corr_pyramid, flow):

        b, _, h, w = flow.shape

        coords = (self.grid + flow).permute(0, 2, 3, 1)
        coords = coords.reshape(b*h*w, 1, 1, 2)

        out_list = []

        for level, corr in enumerate(corr_pyramid):
            curr_coords = coords / 2**level + self.delta
            corr = bilinear_sample(corr, curr_coords)
            corr = corr.view(b, h, w, -1)
            out_list.append(corr)

        out = torch.cat(out_list, dim=-1)

        return out.permute(0, 3, 1, 2).contiguous()

    def init_corr_pyr(self, feature0, feature1):
        b, c, h, w = feature0.shape
        feature0 = feature0.view(b, c, h*w)
        feature1 = feature1.view(b, c, h*w)
        
        corr = torch.matmul(feature0.transpose(1,2), feature1)
        corr = corr.view(b*h*w, 1, h, w) / math.sqrt(c)

        corr_pyramid = [corr]
        for i in range(self.levels-1):
            corr = F.avg_pool2d(corr, kernel_size=2, stride=2)
            corr_pyramid.append(corr)

        return corr_pyramid


# class RAFTCorrBlock:
#     def __init__(self, fmap1, fmap2, num_levels=4, radius=4):
#         self.num_levels = num_levels
#         self.radius = radius
#         self.corr_pyramid = []

#         # all pairs correlation
#         corr = RAFTCorrBlock.corr(fmap1, fmap2)

#         batch, h1, w1, dim, h2, w2 = corr.shape
#         corr = corr.reshape(batch*h1*w1, dim, h2, w2)
        
#         self.corr_pyramid.append(corr)
#         for i in range(self.num_levels-1):
#             corr = F.avg_pool2d(corr, 2, stride=2)
#             self.corr_pyramid.append(corr)

#     def __call__(self, coords):
#         r = self.radius
#         coords = coords.permute(0, 2, 3, 1)
#         batch, h1, w1, _ = coords.shape

#         out_pyramid = []
#         for i in range(self.num_levels):
#             corr = self.corr_pyramid[i]
#             dx = torch.linspace(-r, r, 2*r+1, device=coords.device)
#             dy = torch.linspace(-r, r, 2*r+1, device=coords.device)
#             delta = torch.stack(torch.meshgrid(dy, dx), axis=-1)

#             centroid_lvl = coords.reshape(batch*h1*w1, 1, 1, 2) / 2**i
#             delta_lvl = delta.view(1, 2*r+1, 2*r+1, 2)
#             coords_lvl = centroid_lvl + delta_lvl

#             corr = bilinear_sample(corr, coords_lvl)
#             corr = corr.view(batch, h1, w1, -1)
#             out_pyramid.append(corr)

#         out = torch.cat(out_pyramid, dim=-1)
#         return out.permute(0, 3, 1, 2).contiguous().float()

#     @staticmethod
#     def corr(fmap1, fmap2):
#         batch, dim, ht, wd = fmap1.shape
#         fmap1 = fmap1.view(batch, dim, ht*wd)
#         fmap2 = fmap2.view(batch, dim, ht*wd) 
        
#         corr = torch.matmul(fmap1.transpose(1,2), fmap2)
#         corr = corr.view(batch, ht, wd, 1, ht, wd)
#         return corr  / torch.sqrt(torch.tensor(dim).float())
