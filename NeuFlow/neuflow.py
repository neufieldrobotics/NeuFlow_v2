import torch
import torch.nn.functional as F

from NeuFlow import backbone
from NeuFlow import transformer
from NeuFlow import matching
from NeuFlow import corr
from NeuFlow import refine
from NeuFlow import upsample
from NeuFlow import utils

from NeuFlow import config


class NeuFlow(torch.nn.Module):
    def __init__(self):
        super(NeuFlow, self).__init__()

        self.backbone = backbone.CNNEncoder(config.feature_dim_s16)
        self.cross_attn_s16 = transformer.FeatureAttention(config.feature_dim_s16+2, num_layers=2, ffn=True, ffn_dim_expansion=1, post_norm=True)
        
        self.matching_s16 = matching.Matching()

        self.flow_attn_s16 = transformer.FlowAttention(config.feature_dim_s16+2+1)

        self.merge_s8 = torch.nn.Sequential(torch.nn.Conv2d(config.feature_dim_s16+config.feature_dim_s8+4, config.feature_dim_s8 * 2, kernel_size=3, stride=1, padding=1, bias=False),
                                              torch.nn.GELU(),
                                              torch.nn.Conv2d(config.feature_dim_s8 * 2, config.feature_dim_s8, kernel_size=3, stride=1, padding=1, bias=False))

        self.corr_block = corr.CorrBlock(radius=4)
        self.refine_s8 = refine.Refine(config.feature_dim_s8, num_layers=6, radius=4)

        self.conv_s8 = backbone.ConvBlock(3, config.feature_dim_s8, kernel_size=8, stride=8, padding=0)

        self.upsample_s1 = upsample.UpSample(config.feature_dim_s1, upsample_factor=8)

        for p in self.parameters():
            if p.dim() > 1:
                torch.nn.init.xavier_uniform_(p)

    def init_bhw(self, batch_size, height, width):
        self.backbone.init_pos_12(batch_size*2, height, width)
        self.matching_s16.init_grid(batch_size, height//16, width//16)
        self.corr_block.init_grid(batch_size, height//8, width//8)
        self.img_mean = torch.tensor([0.485, 0.456, 0.406], dtype=torch.half, device='cuda').view(1, 3, 1, 1)
        self.img_std = torch.tensor([0.229, 0.224, 0.225], dtype=torch.half, device='cuda').view(1, 3, 1, 1)
        # self.init_flow = torch.zeros((batch_size, 2, height//8, width//8)).cuda()

    def forward(self, img0, img1, iters=4):

        flow_list = []

        imgs = torch.cat([img0, img1], dim=0)

        imgs = utils.normalize_img(imgs, self.img_mean, self.img_std)

        features_s16, features_s8 = self.backbone(imgs)

        features_s16 = self.cross_attn_s16(features_s16)

        feature0_s16, feature1_s16 = features_s16.chunk(chunks=2, dim=0)

        flow0, corr_conf = self.matching_s16.global_correlation_softmax(feature0_s16, feature1_s16)

        flow0 = self.flow_attn_s16(torch.cat([feature0_s16, corr_conf], dim=1), flow0)

        features_s16 = F.interpolate(features_s16, scale_factor=2, mode='nearest')

        features_s8 = self.merge_s8(torch.cat([features_s8, features_s16], dim=1))

        feature0_s8, feature1_s8 = features_s8.chunk(chunks=2, dim=0)

        flow0 = F.interpolate(flow0, scale_factor=2, mode='nearest') * 2

        # delta_flow = self.refine_s8(feature0_s8, utils.flow_warp(feature1_s8, flow0), flow0)
        # flow0 = flow0 + delta_flow

        # flow0 = self.init_flow

        corr_pyr_s8 = self.corr_block.init_corr_pyr(feature0_s8, feature1_s8)

        for i in range(iters):

            if i > 0:
                flow0 = flow0.detach()

            corrs = self.corr_block(corr_pyr_s8, flow0)

            delta_flow = self.refine_s8(corrs, feature0_s8, flow0)
            flow0 = flow0 + delta_flow

            if self.training or i == iters-1:
                feature0_s1 = self.conv_s8(img0)
                flow0_s1 = self.upsample_s1(feature0_s1, flow0)
                flow_list.append(flow0_s1)

        return flow_list
