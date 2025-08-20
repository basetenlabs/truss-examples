import torch
import torch.nn as nn
import torch.nn.functional as F


class WaveletEncoder(nn.Module):
    def __init__(self, ae_input_channel, args):
        super(WaveletEncoder, self).__init__()
        self.ef_dim = args.ae_ef_dim
        self.z_dim = args.ae_z_dim
        self.slope = args.negative_slope
        self.conv_1 = nn.Conv3d(
            ae_input_channel, self.ef_dim, 4, stride=2, padding=1, bias=True
        )
        self.in_1 = nn.InstanceNorm3d(self.ef_dim)
        self.conv_2 = nn.Conv3d(
            self.ef_dim, self.ef_dim * 2, 4, stride=2, padding=1, bias=True
        )
        self.in_2 = nn.InstanceNorm3d(self.ef_dim * 2)
        self.conv_3 = nn.Conv3d(
            self.ef_dim * 2, self.ef_dim * 4, 4, stride=2, padding=1, bias=True
        )
        self.in_3 = nn.InstanceNorm3d(self.ef_dim * 4)
        self.conv_4 = nn.Conv3d(
            self.ef_dim * 4, self.z_dim, 4, stride=2, padding=1, bias=True
        )
        # self.in_4 = nn.InstanceNorm3d(self.ef_dim * 8)
        # self.conv_5 = nn.Conv3d(self.ef_dim * 8, self.z_dim, 4, stride=1, padding=0, bias=True)
        nn.init.xavier_uniform_(self.conv_1.weight)
        nn.init.xavier_uniform_(self.conv_2.weight)
        nn.init.xavier_uniform_(self.conv_3.weight)
        nn.init.xavier_uniform_(self.conv_4.weight)
        # nn.init.xavier_uniform_(self.conv_5.weight)
        nn.init.constant_(self.conv_1.bias, 0)
        nn.init.constant_(self.conv_2.bias, 0)
        nn.init.constant_(self.conv_3.bias, 0)
        nn.init.constant_(self.conv_4.bias, 0)
        # nn.init.constant_(self.conv_5.bias, 0)

    def forward(self, inputs):
        batch_size = inputs.size(0)
        d_1 = self.in_1(self.conv_1(inputs))
        d_1 = F.leaky_relu(d_1, negative_slope=self.slope, inplace=True)

        d_2 = self.in_2(self.conv_2(d_1))
        d_2 = F.leaky_relu(d_2, negative_slope=self.slope, inplace=True)

        d_3 = self.in_3(self.conv_3(d_2))
        d_3 = F.leaky_relu(d_3, negative_slope=self.slope, inplace=True)

        # d_4 = self.in_4(self.conv_4(d_3))
        # d_4 = F.leaky_relu(d_4, negative_slope=self.slope, inplace=True)

        d_5 = self.conv_4(d_3)

        ## reshape
        d_5 = torch.permute(d_5, (0, 2, 3, 4, 1))
        d_5 = d_5.reshape((batch_size, -1, self.z_dim))

        d_5 = torch.mean(d_5, dim=1, keepdim=True)  # B * 1 * z_dim

        # # positional embeding
        # position_indices = torch.arange(d_4.size(1)).to(d_4.device).unsqueeze(0).unsqueeze(2).repeat(batch_size, 1, 1) / d_4.size(1)
        # d_4 = torch.cat((position_indices, d_4), dim=2)

        ### average pooling
        # d_4 = torch.nn.functional.avg_pool3d(d_4, (d_4.size(2), d_4.size(3), d_4.size(4)))
        # d_4 = d_4.squeeze(4).squeeze(3).squeeze(2)

        return d_5
