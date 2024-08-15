# Data loading based on https://github.com/NVIDIA/flownet2-pytorch

import numpy as np
import torch
import torch.utils.data as data

import os
import random
from glob import glob
import os.path as osp

from data_utils import frame_utils
from data_utils.transforms import FlowAugmentor, SparseFlowAugmentor


class FlowDataset(data.Dataset):
    def __init__(self, aug_params=None, sparse=False, virtual=False,
                 load_occlusion=False,
                 ):
        self.augmentor = None
        self.sparse = sparse
        self.virtual = virtual

        if aug_params is not None:
            if sparse:
                self.augmentor = SparseFlowAugmentor(**aug_params)
            else:
                self.augmentor = FlowAugmentor(**aug_params)

        self.is_test = False
        self.init_seed = False
        self.flow_list = []
        self.image_list = []
        self.extra_info = []

        self.load_occlusion = load_occlusion
        self.occ_list = []

    def __getitem__(self, index):

        if self.is_test:
            img1 = frame_utils.read_gen(self.image_list[index][0])
            img2 = frame_utils.read_gen(self.image_list[index][1])

            img1 = np.array(img1).astype(np.uint8)[..., :3]
            img2 = np.array(img2).astype(np.uint8)[..., :3]

            img1 = torch.from_numpy(img1).permute(2, 0, 1).float()
            img2 = torch.from_numpy(img2).permute(2, 0, 1).float()

            return img1, img2, self.extra_info[index]

        if not self.init_seed:
            worker_info = torch.utils.data.get_worker_info()
            if worker_info is not None:
                torch.manual_seed(worker_info.id)
                np.random.seed(worker_info.id)
                random.seed(worker_info.id)
                self.init_seed = True

        index = index % len(self.image_list)
        valid = None

        if self.sparse:
            if self.virtual:
                flow, valid = frame_utils.read_vkitti_png_flow(self.flow_list[index])  # [H, W, 2], [H, W]
            else:
                flow, valid = frame_utils.readFlowKITTI(self.flow_list[index])  # [H, W, 2], [H, W]
        else:
            flow = frame_utils.read_gen(self.flow_list[index])

        if self.load_occlusion:
            occlusion = frame_utils.read_gen(self.occ_list[index])  # [H, W], 0 or 255 (occluded)

        img1 = frame_utils.read_gen(self.image_list[index][0])
        img2 = frame_utils.read_gen(self.image_list[index][1])

        flow = np.array(flow).astype(np.float32)
        img1 = np.array(img1).astype(np.uint8)
        img2 = np.array(img2).astype(np.uint8)

        if self.load_occlusion:
            occlusion = np.array(occlusion).astype(np.float32)

        # grayscale images
        if len(img1.shape) == 2:
            img1 = np.tile(img1[..., None], (1, 1, 3))
            img2 = np.tile(img2[..., None], (1, 1, 3))
        else:
            img1 = img1[..., :3]
            img2 = img2[..., :3]

        if self.augmentor is not None:
            if self.sparse:
                img1, img2, flow, valid = self.augmentor(img1, img2, flow, valid)
            else:
                if self.load_occlusion:
                    img1, img2, flow, occlusion = self.augmentor(img1, img2, flow, occlusion=occlusion)
                else:
                    img1, img2, flow = self.augmentor(img1, img2, flow)

        img1 = torch.from_numpy(img1).permute(2, 0, 1).float()
        img2 = torch.from_numpy(img2).permute(2, 0, 1).float()
        flow = torch.from_numpy(flow).permute(2, 0, 1).float()

        if self.load_occlusion:
            occlusion = torch.from_numpy(occlusion)  # [H, W]

        if valid is not None:
            valid = torch.from_numpy(valid)
        else:
            valid = (flow[0].abs() < 1000) & (flow[1].abs() < 1000)

        # mask out occluded pixels
        if self.load_occlusion:
            # non-occlusion: 0, occlusion: 255
            noc_valid = 1 - occlusion / 255.  # 0 or 1

            return img1, img2, flow, noc_valid.float()

        return img1, img2, flow, valid.float()

    def __rmul__(self, v):
        self.flow_list = v * self.flow_list
        self.image_list = v * self.image_list
        self.occ_list = v * self.occ_list

        return self

    def __len__(self):
        return len(self.image_list)


class MpiSintel(FlowDataset):
    def __init__(self, aug_params=None, split='training',
                 root='datasets/Sintel',
                 dstype='clean',
                 load_occlusion=False,
                 ):
        super(MpiSintel, self).__init__(aug_params,
                                        load_occlusion=load_occlusion,
                                        )

        flow_root = osp.join(root, split, 'flow')
        image_root = osp.join(root, split, dstype)

        if load_occlusion:
            occlusion_root = osp.join(root, split, 'occlusions')

        if split == 'test':
            self.is_test = True

        for scene in os.listdir(image_root):
            image_list = sorted(glob(osp.join(image_root, scene, '*.npy')))
            for i in range(len(image_list) - 1):
                self.image_list += [[image_list[i], image_list[i + 1]]]
                self.extra_info += [(scene, i)]  # scene and frame_id

            if split != 'test':
                self.flow_list += sorted(glob(osp.join(flow_root, scene, '*.npy')))

                if load_occlusion:
                    self.occ_list += sorted(glob(osp.join(occlusion_root, scene, '*.png')))


class FlyingChairs(FlowDataset):
    def __init__(self, aug_params=None, split='train',
                 root='datasets/FlyingChairs_release/data',
                 ):
        super(FlyingChairs, self).__init__(aug_params)

        images = sorted(glob(osp.join(root, '*.ppm')))
        flows = sorted(glob(osp.join(root, '*.flo')))
        assert (len(images) // 2 == len(flows))

        split_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'chairs_split.txt')
        split_list = np.loadtxt(split_file, dtype=np.int32)
        for i in range(len(flows)):
            xid = split_list[i]
            if (split == 'training' and xid == 1) or (split == 'validation' and xid == 2):
                self.flow_list += [flows[i]]
                self.image_list += [[images[2 * i], images[2 * i + 1]]]


class FlyingThings3D(FlowDataset):
    def __init__(self, aug_params=None,
                 root='datasets/FlyingThings3D',
                 dstype='frames_cleanpass',
                 test_set=False,
                 validate_subset=False,
                 only_left=True,
                 ):
        super(FlyingThings3D, self).__init__(aug_params)

        img_dir = root
        flow_dir = root

        if only_left:
            cam_list = ['left']
        else: 
            cam_list = ['left', 'right']

        for cam in cam_list:
            for direction in ['into_future', 'into_past']:
                if test_set:
                    image_dirs = sorted(glob(osp.join(img_dir, dstype, 'TEST/*/*')))
                else:
                    image_dirs = sorted(glob(osp.join(img_dir, dstype, 'TRAIN/*/*')))
                image_dirs = sorted([osp.join(f, cam) for f in image_dirs])

                if test_set:
                    flow_dirs = sorted(glob(osp.join(flow_dir, 'optical_flow/TEST/*/*')))
                else:
                    flow_dirs = sorted(glob(osp.join(flow_dir, 'optical_flow/TRAIN/*/*')))
                flow_dirs = sorted([osp.join(f, direction, cam) for f in flow_dirs])

                for idir, fdir in zip(image_dirs, flow_dirs):
                    images = sorted(glob(osp.join(idir, '*.npy')))
                    flows = sorted(glob(osp.join(fdir, '*.npy')))
                    for i in range(len(flows) - 1):
                        if direction == 'into_future':
                            self.image_list += [[images[i], images[i + 1]]]
                            self.flow_list += [flows[i]]
                        elif direction == 'into_past':
                            self.image_list += [[images[i + 1], images[i]]]
                            self.flow_list += [flows[i + 1]]

        # if test_set and validate_subset:
        if validate_subset:
            num_val_samples = 1024
            all_test_samples = len(self.image_list)  # 7866

            stride = all_test_samples // num_val_samples
            remove = all_test_samples % num_val_samples

            self.image_list = self.image_list[:-remove][::stride]
            self.flow_list = self.flow_list[:-remove][::stride]


class KITTI(FlowDataset):
    def __init__(self, aug_params=None, split='training',
                 root='datasets/KITTI',
                 ):
        super(KITTI, self).__init__(aug_params, sparse=True,
                                    )
        if split == 'testing':
            self.is_test = True

        root = osp.join(root, split)
        images1 = sorted(glob(osp.join(root, 'image_2/*_10.npy')))
        images2 = sorted(glob(osp.join(root, 'image_2/*_11.npy')))

        for img1, img2 in zip(images1, images2):
            frame_id = img1.split('/')[-1]
            self.extra_info += [[frame_id]]
            self.image_list += [[img1, img2]]

        if split == 'training':
            self.flow_list = sorted(glob(osp.join(root, 'flow_occ/*_10.png')))


class HD1K(FlowDataset):
    def __init__(self, aug_params=None, root='datasets/HD1K'):
        super(HD1K, self).__init__(aug_params, sparse=True)

        seq_ix = 0
        while 1:
            flows = sorted(glob(os.path.join(root, 'hd1k_flow_gt', 'flow_occ/%06d_*.png' % seq_ix)))
            images = sorted(glob(os.path.join(root, 'hd1k_input', 'image_2/%06d_*.npy' % seq_ix)))

            if len(flows) == 0:
                break

            for i in range(len(flows) - 1):
                self.flow_list += [flows[i]]
                self.image_list += [[images[i], images[i + 1]]]

            seq_ix += 1

class VIPER(FlowDataset):
    def __init__(self, aug_params=None,
                 root='datasets/VIPER',
                 splits=['train', 'train_2'],
                 load_occlusion=True,
                 only_left=False
                 ):
        super(VIPER, self).__init__(aug_params, load_occlusion=load_occlusion)

        empty_occ_path = root + '/empty_mask.png'

        for split in splits:

            image_dirs = sorted(glob(osp.join(root, split, 'img/*')))
            
            for image_dir in image_dirs:

                seq = os.path.basename(image_dir)
                print(seq)

                image_paths = sorted(glob(osp.join(image_dir, '*.npy')))
                
                for image_0_path, image_1_path in zip(image_paths[:-1],image_paths[1:]):

                    file_0_name = os.path.splitext(os.path.basename(image_0_path))[0]
                    flow_0_path = osp.join(root, split, 'flow', seq, file_0_name + '.npz')

                    if os.path.exists(flow_0_path):
                        self.image_list += [[image_0_path, image_1_path]]
                        self.flow_list += [flow_0_path]
                        if load_occlusion:
                            occ_0_path = osp.join(root, split, 'flow_mask', seq, file_0_name + '.png')
                            if os.path.exists(occ_0_path):
                                self.occ_list += [occ_0_path]
                            else:
                                self.occ_list += [empty_occ_path]

                    if not only_left:

                        file_1_name = os.path.splitext(os.path.basename(image_1_path))[0]
                        flow_1_path = osp.join(root, split, 'flowbw', seq, file_1_name + '.npz')

                        if os.path.exists(flow_1_path):
                            self.image_list += [[image_1_path, image_0_path]]
                            self.flow_list += [flow_1_path]
                            if load_occlusion:
                                occ_1_path = osp.join(root, split, 'flowbw_mask', seq, file_1_name + '.png')
                                if os.path.exists(occ_1_path):
                                    self.occ_list += [occ_1_path]
                                else:
                                    self.occ_list += [empty_occ_path]


class NeuSim(FlowDataset):
    def __init__(self, aug_params=None,
                 root='datasets/NeuSim'
                 ):
        super(NeuSim, self).__init__(aug_params)

        image_dirs = sorted(glob(osp.join(root, '*/image')))

        fw_flow_dirs = sorted(glob(osp.join(root, '*/forward_flow')))
        bw_flow_dirs = sorted(glob(osp.join(root, '*/backward_flow')))

        for image_dir, fw_flow_dir, bw_flow_dir in zip(image_dirs, fw_flow_dirs, bw_flow_dirs):
            images = sorted(glob(osp.join(image_dir, '*.png')))
            fw_flows = sorted(glob(osp.join(fw_flow_dir, '*.npy')))
            bw_flows = sorted(glob(osp.join(bw_flow_dir, '*.npy')))
            for i in range(len(fw_flows) - 1):
                self.image_list += [[images[i], images[i + 1]]]
                self.flow_list += [fw_flows[i]]
                self.image_list += [[images[i + 1], images[i]]]
                self.flow_list += [bw_flows[i]]


def build_train_dataset(stage):
    if stage == 'chairs':
        aug_params = {'crop_size': (384, 512), 'min_scale': -0.1, 'max_scale': 1.0, 'do_flip': True}

        train_dataset = FlyingChairs(aug_params, split='training')

    elif stage == 'things':
        aug_params = {'crop_size': (384, 768), 'min_scale': -0.4, 'max_scale': 0.8, 'do_flip': True}

        clean_dataset = FlyingThings3D(aug_params, dstype='frames_cleanpass')
        final_dataset = FlyingThings3D(aug_params, dstype='frames_finalpass')
        train_dataset = clean_dataset + final_dataset

    elif stage == 'sintel':
        crop_size = (368, 768)
        aug_params = {'crop_size': crop_size, 'min_scale': -0.2, 'max_scale': 0.6, 'do_flip': True}

        things_clean = FlyingThings3D(aug_params, dstype='frames_cleanpass')

        sintel_clean = MpiSintel(aug_params, split='training', dstype='clean')
        sintel_final = MpiSintel(aug_params, split='training', dstype='final')

        aug_params = {'crop_size': crop_size, 'min_scale': -0.3, 'max_scale': 0.5, 'do_flip': True}

        kitti = KITTI(aug_params=aug_params)

        aug_params = {'crop_size': crop_size, 'min_scale': -0.5, 'max_scale': 0.2, 'do_flip': True}

        hd1k = HD1K(aug_params=aug_params)

        train_dataset = 20 * sintel_clean + 20 * sintel_final + 200 * kitti + 5 * hd1k + things_clean
        print(len(sintel_clean), len(sintel_final), len(kitti), len(hd1k), len(things_clean))

    elif stage == 'viper':
        crop_size = (368, 768)
        aug_params = {'crop_size': crop_size, 'min_scale': -0.2, 'max_scale': 0.6, 'do_flip': True}

        things_clean = FlyingThings3D(aug_params, dstype='frames_cleanpass')

        sintel_clean = MpiSintel(aug_params, split='training', dstype='clean')
        sintel_final = MpiSintel(aug_params, split='training', dstype='final')

        aug_params = {'crop_size': crop_size, 'min_scale': -0.3, 'max_scale': 0.5, 'do_flip': True}

        kitti = KITTI(aug_params=aug_params)

        aug_params = {'crop_size': crop_size, 'min_scale': -0.5, 'max_scale': 0.2, 'do_flip': True}

        hd1k = HD1K(aug_params=aug_params)

        aug_params = {'crop_size': crop_size, 'min_scale': -0.2, 'max_scale': 0.6, 'do_flip': True}

        viper = VIPER(aug_params=aug_params)

        # train_dataset = 80 * sintel_clean + 80 * sintel_final + 400 * kitti + 20 * hd1k + 4 * things_clean + viper
        train_dataset = 40 * sintel_clean + 40 * sintel_final + 20 * hd1k + 4 * things_clean + viper + 400 * kitti
        print(len(sintel_clean), len(sintel_final), len(kitti), len(hd1k), len(things_clean), len(viper))

    elif stage == 'kitti':
        aug_params = {'crop_size': (320, 1152), 'min_scale': -0.2, 'max_scale': 0.4, 'do_flip': False}

        train_dataset = KITTI(aug_params, split='training')

    elif stage == 'neusim':
        crop_size = (320, 896)
        aug_params = {'crop_size': crop_size, 'min_scale': -0.4, 'max_scale': 0.8, 'do_flip': True}
        things_clean = FlyingThings3D(aug_params, dstype='frames_cleanpass')
        things_final = FlyingThings3D(aug_params, dstype='frames_finalpass')

        aug_params = {'crop_size': crop_size, 'min_scale': -1, 'max_scale': 0, 'do_flip': False}
        neu_dataset = NeuSim(aug_params)

        train_dataset = things_clean + things_final + 2 * neu_dataset

    return train_dataset
