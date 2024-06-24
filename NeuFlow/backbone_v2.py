import torch
import torch.nn.functional as F


class ConvBlock(torch.nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, stride, padding):
        super(ConvBlock, self).__init__()

        self.conv = torch.nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=padding, padding_mode='zeros', bias=False)

        self.relu = torch.nn.LeakyReLU(negative_slope=0.1, inplace=True)

        self.norm = torch.nn.BatchNorm2d(out_planes, affine=False)

        # self.dropout = torch.nn.Dropout(p=0.1)

    def forward(self, x):

        # x = self.dropout(x)

        x = self.norm(self.relu(self.conv(x)))

        return x

class CNNEncoder(torch.nn.Module):
    def __init__(self, feature_dim_s16, feature_dim_s8, feature_dim_s4):
        super(CNNEncoder, self).__init__()

        inter_dim = 64

        self.block_4_1 = ConvBlock(3, feature_dim_s4, kernel_size=8, stride=4, padding=2)
        # self.block_4_2 = ConvBlock(3, inter_dim, kernel_size=6, stride=2, padding=2)
        # self.block_cat_4 = ConvBlock(inter_dim * 2, feature_dim_s4 - 2, kernel_size=3, stride=1, padding=1)

        self.block_8_1 = ConvBlock(feature_dim_s4, feature_dim_s8, kernel_size=6, stride=2, padding=2)
        self.block_8_2 = ConvBlock(3, inter_dim, kernel_size=6, stride=2, padding=2)
        self.block_8_2_extra = ConvBlock(inter_dim, feature_dim_s8, kernel_size=3, stride=1, padding=1)
        self.block_cat_8 = ConvBlock(feature_dim_s8 * 2, feature_dim_s8, kernel_size=3, stride=1, padding=1)

        self.block_16_1 = ConvBlock(feature_dim_s8, feature_dim_s16, kernel_size=6, stride=2, padding=2)
        self.block_16_2 = ConvBlock(3, inter_dim, kernel_size=6, stride=2, padding=2)
        self.block_cat_16 = ConvBlock(inter_dim + feature_dim_s16, feature_dim_s16-2, kernel_size=3, stride=1, padding=1)

    def init_pos(self, batch_size, height, width):

        ys, xs = torch.meshgrid(torch.arange(height, dtype=torch.half, device='cuda'), torch.arange(width, dtype=torch.half, device='cuda'), indexing='ij')
        ys = ys / (height-1)
        xs = xs / (width-1)
        pos = torch.stack([ys, xs])
        return pos[None].repeat(batch_size,1,1,1)

    def init_pos_scales(self, batch_size, height, width):
        self.pos_s16 = self.init_pos(batch_size, height//16, width//16)
        self.pos_s8 = self.init_pos(batch_size, height//8, width//8)
        # self.pos_s4 = self.init_pos(batch_size, height//4, width//4)

    def forward(self, img):

        x_4 = self.block_4_1(img)
        img = F.avg_pool2d(img, kernel_size=2, stride=2)
        # x_4_2 = self.block_4_2(img)
        # x_4 = self.block_cat_4(torch.cat([x_4, x_4_2], dim=1))

        x_8 = self.block_8_1(x_4)
        img = F.avg_pool2d(img, kernel_size=2, stride=2)
        x_8_2 = self.block_8_2_extra(self.block_8_2(img))
        x_8 = self.block_cat_8(torch.cat([x_8, x_8_2], dim=1))

        x_16 = self.block_16_1(x_8)
        img = F.avg_pool2d(img, kernel_size=2, stride=2)
        x_16_2 = self.block_16_2(img)
        # x_16_2 = torch.zeros_like(x_16_2)
        x_16 = self.block_cat_16(torch.cat([x_16, x_16_2], dim=1))

        x_16 = torch.cat([x_16, self.pos_s16], dim=1)
        x_8 = torch.cat([x_8, self.pos_s8], dim=1)
        # x_4 = torch.cat([x_4, self.pos_s4], dim=1)

        return x_16, x_8
