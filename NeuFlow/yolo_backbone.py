import torch
import torch.nn as nn
import cv2
import numpy as np


def autopad(k, p=None, d=1):  # kernel, padding, dilation
    """Pad to 'same' shape outputs."""
    if d > 1:
        k = d * (k - 1) + 1 if isinstance(k, int) else [d * (x - 1) + 1 for x in k]  # actual kernel-size
    if p is None:
        p = k // 2 if isinstance(k, int) else [x // 2 for x in k]  # auto-pad
    return p


class Conv(nn.Module):
    """Standard convolution with args(ch_in, ch_out, kernel, stride, padding, groups, dilation, activation)."""

    default_act = nn.SiLU(inplace=True)  # default activation

    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, d=1, act=True):
        """Initialize Conv layer with given arguments including activation."""
        super().__init__()
        self.conv = nn.Conv2d(c1, c2, k, s, autopad(k, p, d), groups=g, dilation=d, bias=False)
        self.bn = nn.BatchNorm2d(c2, eps=0.001, momentum=0.03)
        self.act = self.default_act if act is True else act if isinstance(act, nn.Module) else nn.Identity()

    def forward(self, x):
        """Apply convolution, batch normalization and activation to input tensor."""
        return self.act(self.bn(self.conv(x)))

    def forward_fuse(self, x):
        """Perform transposed convolution of 2D data."""
        return self.act(self.conv(x))


class Bottleneck(nn.Module):
    """Standard bottleneck."""

    def __init__(self, c1, c2, shortcut=True, g=1, k=(3, 3), e=0.5):
        """Initializes a bottleneck module with given input/output channels, shortcut option, group, kernels, and
        expansion.
        """
        super().__init__()
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, c_, k[0], 1)
        self.cv2 = Conv(c_, c2, k[1], 1, g=g)
        self.add = shortcut and c1 == c2

    def forward(self, x):
        """'forward()' applies the YOLO FPN to input data."""
        return x + self.cv2(self.cv1(x)) if self.add else self.cv2(self.cv1(x))


class C2f(nn.Module):
    """Faster Implementation of CSP Bottleneck with 2 convolutions."""

    def __init__(self, c1, c2, n=1, shortcut=False, g=1, e=0.5):
        """Initialize CSP bottleneck layer with two convolutions with arguments ch_in, ch_out, number, shortcut, groups,
        expansion.
        """
        super().__init__()
        self.c = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, 2 * self.c, 1, 1)
        self.cv2 = Conv((2 + n) * self.c, c2, 1)  # optional act=FReLU(c2)
        self.m = nn.ModuleList(Bottleneck(self.c, self.c, shortcut, g, k=((3, 3), (3, 3)), e=1.0) for _ in range(n))

    def forward(self, x):
        """Forward pass through C2f layer."""
        y = list(self.cv1(x).chunk(2, 1))
        y.extend(m(y[-1]) for m in self.m)
        return self.cv2(torch.cat(y, 1))


class SPPF(nn.Module):
    """Spatial Pyramid Pooling - Fast (SPPF) layer for YOLOv5 by Glenn Jocher."""

    def __init__(self, c1, c2, k=5):
        """
        Initializes the SPPF layer with given input/output channels and kernel size.

        This module is equivalent to SPP(k=(5, 9, 13)).
        """
        super().__init__()
        c_ = c1 // 2  # hidden channels
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c_ * 4, c2, 1, 1)
        self.m = nn.MaxPool2d(kernel_size=k, stride=1, padding=k // 2)

    def forward(self, x):
        """Forward pass through Ghost Convolution block."""
        y = [self.cv1(x)]
        y.extend(self.m(y[-1]) for _ in range(3))
        return self.cv2(torch.cat(y, 1))


class Concat(nn.Module):
    """Concatenate a list of tensors along dimension."""

    def __init__(self, dimension=1):
        """Concatenates a list of tensors along a specified dimension."""
        super().__init__()
        self.d = dimension

    def forward(self, x):
        """Forward pass for the YOLOv8 mask Proto module."""
        return torch.cat(x, self.d)




class YoloBackbone(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_0 = Conv(3, 16, 3, 2)
        self.conv_1 = Conv(16, 32, 3, 2)
        self.c2f_2 = C2f(32, 32, 1, True)
        self.conv_3 = Conv(32, 64, 3, 2)
        self.c2f_4 = C2f(64, 64, 2, True)
        self.conv_5 = Conv(64, 128, 3, 2)
        self.c2f_6 = C2f(128, 128, 2, True)
        self.conv_7 = Conv(128, 256, 3, 2)
        self.c2f_8 = C2f(256, 256, 1, True)
        self.sppf_9 = SPPF(256, 256)
        self.upsample_10 = nn.Upsample(None, 2, "nearest")
        self.concat_11 = Concat(1)
        self.c2f_12 = C2f(384, 128, 1, False)
        self.upsample_13 = nn.Upsample(None, 2, "nearest")
        self.concat_14 = Concat(1)
        self.c2f_15 = C2f(192, 64, 1, False)
        self.conv_16 = Conv(64, 64, 3, 2)
        self.concat_17 = Concat(1)
        self.c2f_18 = C2f(192, 128, 1, False)

    def init_pos(self, batch_size, height, width):

        ys, xs = torch.meshgrid(torch.arange(height), torch.arange(width), indexing='ij')
        ys = ys.half().cuda() / (height-1)
        xs = xs.half().cuda() / (width-1)
        pos = torch.stack([ys, xs])
        return pos[None].repeat(batch_size,1,1,1)

    def init_pos_12(self, batch_size, height, width):
        self.pos_1 = self.init_pos(batch_size, height, width)
        self.pos_2 = self.init_pos(batch_size, height//2, width//2)

    def forward(self, x):
        x = self.conv_0(x)
        x = self.conv_1(x)
        x = self.c2f_2(x)
        x = self.conv_3(x)
        x = self.c2f_4(x)
        x_4 = x.clone()
        x = self.conv_5(x)
        x = self.c2f_6(x)
        x_6 = x.clone()
        x = self.conv_7(x)
        x = self.c2f_8(x)
        x = self.sppf_9(x)
        x = self.upsample_10(x)
        x = self.concat_11([x, x_6])
        x = self.c2f_12(x)
        x_12 = x.clone()
        x = self.upsample_13(x)
        x = self.concat_14([x, x_4])
        x = self.c2f_15(x)
        x_15 = x.clone()
        x = self.conv_16(x)
        x = self.concat_17([x, x_12])
        x = self.c2f_18(x)

        x = torch.cat([x, self.pos_2], dim=1)
        x_15 = torch.cat([x_15, self.pos_1], dim=1)

        return x, x_15
        
# model = YoloBackbone().to('cuda')

# state_dict = {}

# # checkpoint = torch.load('/home/goku/zhiyongzhang/ultralytics/yolov8n-seg.pt', map_location='cuda')

# # for k, v in checkpoint['model'].model.state_dict().items():

# #     module_index, module_suffix = k.split('.', 1)

# #     if module_index == '19':
# #         break

# #     if module_index in ['0', '1', '3', '5', '7', '16']:
# #         module_prefix = 'conv_'
# #     elif module_index in ['2', '4', '6', '8', '12', '15', '18']:
# #         module_prefix = 'c2f_'
# #     elif module_index in ['9']:
# #         module_prefix = 'sppf_'

# #     k = module_prefix + module_index + '.' + module_suffix

# #     state_dict[k] = v

# # torch.save({'model': state_dict}, 'yolo_backbone.pth')

# checkpoint = torch.load('yolo_backbone.pth', map_location='cuda')

# for k, v in checkpoint['model'].items():

#     state_dict[k] = v

# model.load_state_dict(state_dict, strict=True)

# def fuse_conv_and_bn(conv, bn):
#     """Fuse Conv2d() and BatchNorm2d() layers https://tehnokv.com/posts/fusing-batchnorm-and-conv/."""
#     fusedconv = (
#         nn.Conv2d(
#             conv.in_channels,
#             conv.out_channels,
#             kernel_size=conv.kernel_size,
#             stride=conv.stride,
#             padding=conv.padding,
#             dilation=conv.dilation,
#             groups=conv.groups,
#             bias=True,
#         )
#         .requires_grad_(False)
#         .to(conv.weight.device)
#     )

#     # Prepare filters
#     w_conv = conv.weight.clone().view(conv.out_channels, -1)
#     w_bn = torch.diag(bn.weight.div(torch.sqrt(bn.eps + bn.running_var)))
#     fusedconv.weight.copy_(torch.mm(w_bn, w_conv).view(fusedconv.weight.shape))

#     # Prepare spatial bias
#     b_conv = torch.zeros(conv.weight.shape[0], device=conv.weight.device) if conv.bias is None else conv.bias
#     b_bn = bn.bias - bn.weight.mul(bn.running_mean).div(torch.sqrt(bn.running_var + bn.eps))
#     fusedconv.bias.copy_(torch.mm(w_bn, b_conv.reshape(-1, 1)).reshape(-1) + b_bn)

#     return fusedconv

# for m in model.modules():
#     if type(m) is Conv and hasattr(m, "bn"):
#         m.conv = fuse_conv_and_bn(m.conv, m.bn)  # update conv
#         delattr(m, "bn")  # remove batchnorm
#         m.forward = m.forward_fuse  # update forward

# model.eval()


# image = cv2.imread('/media/goku/data/zhiyongzhang/optical_flow/datasets/KITTI/testing/image_2/000000_10.png')

# image = cv2.resize(image, (512, 384))

# image = np.ascontiguousarray(image[..., ::-1])

# image = torch.from_numpy(image).permute(2, 0, 1).float()
# image = image[None].cuda()
# image /= 255

# # torch.cuda.synchronize()
# # start_time = time.time()
# with torch.no_grad():
#     feature_s16, feature_s8 = model(image)
#     print(feature_s16.shape)
#     print(feature_s8.shape)
# # print(feature_s16)
# # torch.cuda.synchronize()
# # end_time = time.time()
# # print('time:', end_time-start_time)
