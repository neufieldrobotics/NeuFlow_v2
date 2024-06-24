import torch
import torch.nn.functional as F

from NeuFlow import backbone_v6
from NeuFlow import transformer
from NeuFlow import matching
from NeuFlow import corr
from NeuFlow import refine
from NeuFlow import upsample
from NeuFlow import utils
from NeuFlow import config

import time


class NeuFlow(torch.nn.Module):
    def __init__(self):
        super(NeuFlow, self).__init__()

        self.backbone = backbone_v6.CNNEncoder(config.feature_dim_s16, config.feature_dim_s8)
        
        self.cross_attn_s16 = transformer.FeatureAttention(config.feature_dim_s16, num_layers=2, ffn=True, ffn_dim_expansion=1, post_norm=True)
        
        self.matching_s16 = matching.Matching()

        # self.flow_attn_s16 = transformer.FlowAttention(config.feature_dim_s16)
        
        self.merge_s8 = torch.nn.Sequential(torch.nn.Conv2d(config.feature_dim_s16 + config.feature_dim_s8 + 2, config.feature_dim_s8, kernel_size=3, stride=1, padding=1, bias=False),
                                              torch.nn.GELU(),
                                              torch.nn.Conv2d(config.feature_dim_s8, config.feature_dim_s8, kernel_size=3, stride=1, padding=1, bias=False),
                                              torch.nn.Tanh())

        self.corr_block_s16 = corr.CorrBlock(radius=4, levels=1)
        self.corr_block_s8 = corr.CorrBlock(radius=4, levels=1)

        self.context_s16 = torch.nn.Sequential(torch.nn.Conv2d(config.feature_dim_s16, config.hidden_dim_s16, kernel_size=3, stride=1, padding=1, bias=False),
                                            torch.nn.GELU(),
                                            torch.nn.Conv2d(config.hidden_dim_s16, config.hidden_dim_s16, kernel_size=3, stride=1, padding=1, bias=False))

        self.context_merge_s8 = torch.nn.Sequential(torch.nn.Conv2d(config.hidden_dim_s16 + config.feature_dim_s8, config.hidden_dim_s8, kernel_size=3, stride=1, padding=1, bias=False),
                                           torch.nn.GELU(),
                                           torch.nn.Conv2d(config.hidden_dim_s8, config.hidden_dim_s8, kernel_size=3, stride=1, padding=1, bias=False),
                                           torch.nn.Tanh())

        self.refine_s16 = refine.Refine(config.hidden_dim_s16, num_layers=6, levels=1, radius=4)
        self.refine_s8 = refine.Refine(config.hidden_dim_s8, num_layers=6, levels=1, radius=4)

        # self.conv_s16 = backbone_v6.ConvBlock(3, config.feature_dim_s1 * 2, kernel_size=16, stride=16, padding=0)
        # self.upsample_s16 = upsample.UpSample(config.feature_dim_s1 * 2, upsample_factor=16)

        self.conv_s8 = backbone_v6.ConvBlock(3, config.feature_dim_s1, kernel_size=8, stride=8, padding=0)
        self.upsample_s8 = upsample.UpSample(config.feature_dim_s1, upsample_factor=8)

        for p in self.parameters():
            if p.dim() > 1:
                torch.nn.init.xavier_uniform_(p)

    def init_bhwd(self, batch_size, height, width, device, amp=True):

        self.backbone.init_bhwd(batch_size*2, height, width, device, amp)

        self.matching_s16.init_bhwd(batch_size, height//16, width//16, device, amp)

        self.corr_block_s16.init_bhwd(batch_size, height//16, width//16, device, amp)
        self.corr_block_s8.init_bhwd(batch_size, height//8, width//8, device, amp)

        self.refine_s16.init_bhwd(batch_size, height//16, width//16, device, amp)
        self.refine_s8.init_bhwd(batch_size, height//8, width//8, device, amp)

    def forward(self, img0, img1, iters_s16=3, iters_s8=4):

        flow_list = []

        img0 /= 255.
        img1 /= 255.

        features_s16, features_s8 = self.backbone(torch.cat([img0, img1], dim=0))

        features_s16 = self.cross_attn_s16(features_s16)

        feature0_s16, feature1_s16 = features_s16.chunk(chunks=2, dim=0)

        flow0 = self.matching_s16.global_correlation_softmax(feature0_s16, feature1_s16)

        # flow0 = self.flow_attn_s16(feature0_s16, flow0)

        corr_pyr_s16 = self.corr_block_s16.init_corr_pyr(feature0_s16, feature1_s16)

        context_s16 = self.context_s16(feature0_s16)
        iter_context_s16 = context_s16.clone()

        for i in range(iters_s16):

            if self.training and i > 0:
                flow0 = flow0.detach()
                # iter_feature0_s16 = iter_feature0_s16.detach()

            corrs = self.corr_block_s16(corr_pyr_s16, flow0)

            iter_context_s16, delta_flow = self.refine_s16(corrs, iter_context_s16, flow0)

            flow0 = flow0 + delta_flow

            if self.training:
                up_flow0 = F.interpolate(flow0, scale_factor=16, mode='bilinear') * 16
                flow_list.append(up_flow0)

        flow0 = F.interpolate(flow0, scale_factor=2, mode='nearest') * 2

        features_s16 = F.interpolate(features_s16, scale_factor=2, mode='nearest')

        features_s8 = self.merge_s8(torch.cat([features_s8, features_s16], dim=1))

        feature0_s8, feature1_s8 = features_s8.chunk(chunks=2, dim=0)

        corr_pyr_s8 = self.corr_block_s8.init_corr_pyr(feature0_s8, feature1_s8)

        context_s16 = F.interpolate(context_s16, scale_factor=2, mode='nearest')

        context_s8 = self.context_merge_s8(torch.cat([feature0_s8, context_s16], dim=1))
        iter_context_s8 = context_s8.clone()

        for i in range(iters_s8):

            if self.training and i > 0:
                flow0 = flow0.detach()
                # iter_feature0_s8 = iter_feature0_s8.detach()

            corrs = self.corr_block_s8(corr_pyr_s8, flow0)

            iter_context_s8, delta_flow = self.refine_s8(corrs, iter_context_s8, flow0)

            flow0 = flow0 + delta_flow

            if self.training or i == iters_s8 - 1:

                feature0_s1 = self.conv_s8(img0)
                up_flow0 = self.upsample_s8(feature0_s1, flow0) * 8
                flow_list.append(up_flow0)

        return flow_list
