import torch
import torch.nn.functional as F


class ConvBlock(torch.nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, stride, padding):
        super(ConvBlock, self).__init__()

        self.conv1 = torch.nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=padding, padding_mode='zeros', bias=False)

        self.conv2 = torch.nn.Conv2d(out_planes, out_planes, kernel_size=3, stride=1, padding=1, bias=False)

        self.relu = torch.nn.LeakyReLU(negative_slope=0.1, inplace=True)

        self.norm = torch.nn.BatchNorm2d(out_planes, affine=False)

    def forward(self, x):

        x = self.relu(self.conv1(x))
        x1 = self.relu(self.conv2(x))

        return self.norm(x + x1)

class DownDimBlock(torch.nn.Module):
    def __init__(self, in_planes, out_planes):
        super(DownDimBlock, self).__init__()

        self.relu = torch.nn.LeakyReLU(negative_slope=0.1, inplace=True)

        self.conv_block = ConvBlock(in_planes, out_planes, kernel_size=1, stride=1, padding=0)

    def forward(self, x):

        return self.conv_block(self.relu(x))

class CNNEncoder(torch.nn.Module):
    def __init__(self, feature_dim_s16, feature_dim_s8):
        super(CNNEncoder, self).__init__()

        self.block1_2 = ConvBlock(3, feature_dim_s8, kernel_size=8, stride=4, padding=2) # 1/2
        self.block1_2_extra = ConvBlock(feature_dim_s8, feature_dim_s8, kernel_size=3, stride=1, padding=1) # 1/2

        self.block1_3 = ConvBlock(3, feature_dim_s8, kernel_size=8, stride=2, padding=3) # 1/4
        self.block1_3_extra = ConvBlock(feature_dim_s8, feature_dim_s8, kernel_size=3, stride=1, padding=1) # 1/4

        self.block1_dd = DownDimBlock(feature_dim_s8 * 4, feature_dim_s8) # pick features
        self.block1_ds = ConvBlock(feature_dim_s8, feature_dim_s16, kernel_size=2, stride=2, padding=0)

        self.block2 = ConvBlock(3, feature_dim_s16, kernel_size=8, stride=2, padding=3) # 1/16
        self.block2_dd = DownDimBlock(feature_dim_s16 * 2, feature_dim_s16 - 2) # pick features

    def init_pos(self, batch_size, height, width, device, amp):
        ys, xs = torch.meshgrid(torch.arange(height, dtype=torch.half if amp else torch.float, device=device), torch.arange(width, dtype=torch.half if amp else torch.float, device=device), indexing='ij')
        ys = ys / (height-1)
        xs = xs / (width-1)
        pos = torch.stack([ys, xs])
        return pos[None].repeat(batch_size,1,1,1)

    def init_bhwd(self, batch_size, height, width, device, amp):
        self.pos_1 = self.init_pos(batch_size, height//8, width//8, device, amp)
        self.pos_2 = self.init_pos(batch_size, height//16, width//16, device, amp)

    def forward(self, img):

        img = F.avg_pool2d(img, kernel_size=2, stride=2)

        x1_2 = self.block1_2(img)
        x1_2_extra = self.block1_2_extra(x1_2)

        img = F.avg_pool2d(img, kernel_size=2, stride=2)

        x1_3 = self.block1_3(img)
        x1_3_extra = self.block1_3_extra(x1_3)

        x1 = torch.cat([x1_2, x1_3, x1_2_extra, x1_3_extra], dim=1)
        x1 = self.block1_dd(x1)

        img = F.avg_pool2d(img, kernel_size=2, stride=2)
        x2 = self.block2(img)

        x2 = torch.cat([self.block1_ds(x1), x2], dim=1)
        x2 = self.block2_dd(x2)

        x1 = torch.cat([x1, self.pos_1], dim=1)
        x2 = torch.cat([x2, self.pos_2], dim=1)

        return [x2, x1]
