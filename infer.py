import torch
from glob import glob
import os
import numpy as np
import cv2
from NeuFlow.neuflow import NeuFlow

from NeuFlow.backbone_v7 import ConvBlock
from data_utils import flow_viz
from fuse_conv_and_bn import fuse_conv_and_bn


image_width = 768
image_height = 432

def get_cuda_image(image_path):
    image = cv2.imread(image_path)

    image = cv2.resize(image, (image_width, image_height))

    image = torch.from_numpy(image).permute(2, 0, 1).half()
    return image[None].cuda()


def main(hugging_face: bool) -> None:
    image_path_list = sorted(glob('test_images/*.jpg'))
    vis_path = 'test_results/'

    device = torch.device('cuda')

    if hugging_face:
        print("from_pretrained: Study-is-happy/neuflow-v2")
        model = NeuFlow.from_pretrained("Study-is-happy/neuflow-v2").to(device)
    else:
        model = NeuFlow().to(device)
        print("load: neuflow_mixed.pth")
        checkpoint = torch.load('neuflow_mixed.pth', map_location='cuda')
        model.load_state_dict(checkpoint['model'], strict=True)

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


if __name__ == "__main__":
    from argparse import ArgumentParser

    parser = ArgumentParser()
    parser.add_argument(
        "--hugging-face", action="store_true",
        help="load model form hugging face (Study-is-happy/neuflow-v2)"
    )
    args = parser.parse_args()
    main(args.hugging_face)
