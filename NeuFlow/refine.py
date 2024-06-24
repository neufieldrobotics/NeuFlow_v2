import torch


class ConvBlock(torch.nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, stride, padding):
        super(ConvBlock, self).__init__()

        self.conv = torch.nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=padding, padding_mode='zeros', bias=True)
        self.relu = torch.nn.LeakyReLU(negative_slope=0.1, inplace=False)

    def forward(self, x):
        return self.relu(self.conv(x))

class Refine(torch.nn.Module):
    def __init__(self, feature_dim, num_layers, levels, radius):
        super(Refine, self).__init__()

        self.radius = radius

        self.conv1 = ConvBlock((radius*2+1)**2*levels+feature_dim+2+1, feature_dim, kernel_size=3, stride=1, padding=1)

        self.conv_layers = torch.nn.ModuleList([ConvBlock(feature_dim, feature_dim, kernel_size=3, stride=1, padding=1)
                                                for i in range(num_layers)])

        self.conv2 = torch.nn.Conv2d(feature_dim, feature_dim+2, kernel_size=3, stride=1, padding=1, padding_mode='zeros', bias=True)

    def init_bhwd(self, batch_size, height, width, device, amp):
        self.radius_emb = torch.tensor(self.radius, dtype=torch.half if amp else torch.float, device=device).view(1,-1,1,1).expand([batch_size,1,height,width])

    def forward(self, corrs, feature0, flow0):

        x = torch.cat([corrs, feature0, flow0, self.radius_emb], dim=1)

        x = self.conv1(x)

        for layer in self.conv_layers:
            x = layer(x)

        x = self.conv2(x)

        return torch.tanh(x[:,2:]), x[:,:2]
