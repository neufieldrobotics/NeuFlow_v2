import torch
from glob import glob
import os
import numpy as np
import cv2
from NeuFlow.neuflow import NeuFlow
from NeuFlow.backbone_v7 import ConvBlock
from data_utils import flow_viz


image_width = 768
image_height = 432

def get_cuda_image(image_path):
    image = cv2.imread(image_path)

    image = cv2.resize(image, (image_width, image_height))

    image = torch.from_numpy(image).permute(2, 0, 1).half()
    return image[None].cuda()


def fuse_conv_and_bn(conv, bn):
        """Fuse Conv2d() and BatchNorm2d() layers https://tehnokv.com/posts/fusing-batchnorm-and-conv/."""
        fusedconv = (
            torch.nn.Conv2d(
                conv.in_channels,
                conv.out_channels,
                kernel_size=conv.kernel_size,
                stride=conv.stride,
                padding=conv.padding,
                dilation=conv.dilation,
                groups=conv.groups,
                bias=True,
            )
            .requires_grad_(False)
            .to(conv.weight.device)
        )

        # Prepare filters
        w_conv = conv.weight.clone().view(conv.out_channels, -1)
        w_bn = torch.diag(bn.weight.div(torch.sqrt(bn.eps + bn.running_var)))
        fusedconv.weight.copy_(torch.mm(w_bn, w_conv).view(fusedconv.weight.shape))

        # Prepare spatial bias
        b_conv = torch.zeros(conv.weight.shape[0], device=conv.weight.device) if conv.bias is None else conv.bias
        b_bn = bn.bias - bn.weight.mul(bn.running_mean).div(torch.sqrt(bn.running_var + bn.eps))
        fusedconv.bias.copy_(torch.mm(w_bn, b_conv.reshape(-1, 1)).reshape(-1) + b_bn)

        return fusedconv


image_path_list = sorted(glob('test_images/*.jpg'))
vis_path = 'test_results/'

device = torch.device('cuda')

model = NeuFlow.from_pretrained("Study-is-happy/neuflow-v2").to(device)

for m in model.modules():
    if type(m) is ConvBlock:
        m.conv1 = fuse_conv_and_bn(m.conv1, m.norm1)  # update conv
        m.conv2 = fuse_conv_and_bn(m.conv2, m.norm2)  # update conv
        delattr(m, "norm1")  # remove batchnorm
        delattr(m, "norm2")  # remove batchnorm
        m.forward = m.forward_fuse  # update forward

model.eval()
model.half()

model.init_bhwd(1, image_height, image_width, 'cuda')

if not os.path.exists(vis_path):
    os.makedirs(vis_path)

for image_path_0, image_path_1 in zip(image_path_list[:-1], image_path_list[1:]):

    print(image_path_0)

    image_0 = get_cuda_image(image_path_0)
    image_1 = get_cuda_image(image_path_1)

    file_name = os.path.basename(image_path_0)

    with torch.no_grad():

        flow = model(image_0, image_1)[-1][0]

        flow = flow.permute(1,2,0).cpu().numpy()
        
        flow = flow_viz.flow_to_image(flow)

        image_0 = cv2.resize(cv2.imread(image_path_0), (image_width, image_height))

        cv2.imwrite(vis_path + file_name, np.vstack([image_0, flow]))

