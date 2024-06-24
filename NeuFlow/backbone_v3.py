import torch
import torch.nn.functional as F


class ConvBlock(torch.nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, stride, padding):
        super(ConvBlock, self).__init__()

        self.conv1 = torch.nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=padding, padding_mode='zeros', bias=False)
        self.conv2 = torch.nn.Conv2d(out_planes, out_planes, kernel_size=3, stride=1, padding=1, padding_mode='zeros', bias=False)

        self.relu = torch.nn.LeakyReLU(negative_slope=0.1, inplace=True)

        self.norm1 = torch.nn.BatchNorm2d(out_planes, affine=False)
        self.norm2 = torch.nn.BatchNorm2d(out_planes, affine=False)

    def forward(self, x):

        x = self.norm1(self.relu(self.conv1(x)))
        x = self.norm2(self.relu(self.conv2(x)))

        return x

class CNNEncoder(torch.nn.Module):
    def __init__(self, feature_dim_s16, feature_dim_s8, feature_dim_s4):
        super(CNNEncoder, self).__init__()

        inter_dim1 = 32
        inter_dim2 = 64

        self.block_s4_1 = ConvBlock(3, inter_dim1, kernel_size=8, stride=4, padding=2)

        self.block_s8_1 = ConvBlock(inter_dim1, inter_dim2, kernel_size=6, stride=2, padding=2)
        self.block_s8_2 = ConvBlock(3, inter_dim2, kernel_size=6, stride=2, padding=2)

        self.block_cat_s8_3 = ConvBlock(inter_dim2 + inter_dim2, inter_dim2, kernel_size=3, stride=1, padding=1)

        self.block_s8_4 = ConvBlock(inter_dim2, inter_dim2, kernel_size=3, stride=1, padding=1)

        self.block_cat_s8_5 = ConvBlock(inter_dim2 + inter_dim2, feature_dim_s8-2, kernel_size=3, stride=1, padding=1)

        self.block_s16_1 = ConvBlock(feature_dim_s8-2, feature_dim_s16-2, kernel_size=6, stride=2, padding=2)

    def init_pos(self, batch_size, height, width):

        ys, xs = torch.meshgrid(torch.arange(height, dtype=torch.half, device='cuda'), torch.arange(width, dtype=torch.half, device='cuda'), indexing='ij')
        ys = ys / (height-1)
        xs = xs / (width-1)
        pos = torch.stack([ys, xs])
        return pos[None].repeat(batch_size,1,1,1)

    def init_pos_scales(self, batch_size, height, width):
        self.pos_s8 = self.init_pos(batch_size, height//8, width//8)
        self.pos_s16 = self.init_pos(batch_size, height//16, width//16)

    def forward(self, img):

        x_s8 = self.block_s4_1(img)
        img = F.avg_pool2d(img, kernel_size=2, stride=2)

        x_s8 = self.block_s8_1(x_s8)
        img = F.avg_pool2d(img, kernel_size=2, stride=2)

        x_s8_2 = self.block_s8_2(img)

        x_s8_3 = self.block_cat_s8_3(torch.cat([x_s8, x_s8_2], dim=1))

        x_s8 = self.block_s8_4(x_s8_3)

        x_s8 = self.block_cat_s8_5(torch.cat([x_s8, x_s8_3], dim=1))

        x_s16 = self.block_s16_1(x_s8)

        x_s8 = torch.cat([x_s8, self.pos_s8], dim=1)
        x_s16 = torch.cat([x_s16, self.pos_s16], dim=1)

        return x_s16, x_s8
