import numpy as np
import glob
import cv2
import json
import os

from data_utils import frame_utils

width = 960
height = 540

x_range = np.arange(width)
y_range = np.arange(height)
xs, ys = np.meshgrid(x_range, y_range)
coords = np.float32(np.dstack([xs, ys]))

root = 'datasets/FlyingThings3D/optical_flow/'

fw_flow_dirs = sorted(glob.glob(root + '*/*/*/into_future/*/'))
bw_flow_dirs = sorted(glob.glob(root + '*/*/*/into_past/*/'))

print(len(fw_flow_dirs))

flow_mag_dict = {}
index = 0

for fw_flow_dir, bw_flow_dir in zip(fw_flow_dirs, bw_flow_dirs):

    fw_flows = sorted(glob.glob(fw_flow_dir + '*.pfm'))[:-1]
    bw_flows = sorted(glob.glob(bw_flow_dir + '*.pfm'))[1:]

    for fw_flow_path, bw_flow_path in zip(fw_flows+bw_flows, bw_flows+fw_flows):

        occlusion_file_path = os.path.splitext(fw_flow_path)[0]+'.png'

        fw_flow = frame_utils.read_gen(fw_flow_path)
        bw_flow = frame_utils.read_gen(bw_flow_path)

        warp_flow = cv2.remap(coords, coords + bw_flow, None, interpolation=cv2.INTER_LINEAR)
        warp_flow = cv2.remap(warp_flow, coords + fw_flow, None, interpolation=cv2.INTER_LINEAR)

        warp_flow -= coords

        occlusion = np.sum(warp_flow**2, axis=-1) < 0.01

        # cv2.imwrite(occlusion_file_path, occlusion*255)
        index += 1
        print(index)
        