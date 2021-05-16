import torch
import torch.nn as nn
import torch.nn.functional as F


class SingleConv(nn.Module):
    def __init__(self, in_ch, out_ch, kernel_size, stride, padding):
        super(SingleConv, self).__init__()

        # Sequential容器，通过Squential将网络层和激活函数结合起来
        self.single_conv = nn.Sequential(
            nn.Conv3d(in_ch, out_ch, kernel_size=kernel_size, padding=padding, stride=stride, bias=True),
            nn.InstanceNorm3d(out_ch, affine=True),  # 归一化层
            nn.ReLU(inplace=True)  # 激活函数，inplace为True，将会改变输入的数据 ，否则不会改变原输入，只会产生新的输出
        )

    def forward(self, x):
        return self.single_conv(x)


class DenseConv(nn.Module):
    def __init__(self, in_ch, list_ch):
        super(DenseConv, self).__init__()
        self.densecov = SingleConv(in_ch, list_ch[1], kernel_size=3, stride=1, padding=1)

    def forward(self, x):
        densecov_out = torch.cat((x, self.densecov(x)), dim=1)
        return densecov_out


class Upsample(nn.Module):
    def __init__(self, in_ch, scale_factor, list_ch):
        super(Upsample, self).__init__()

        self.upsample = nn.Upsample(scale_factor=scale_factor, mode='trilinear', align_corners=True)
        self.cov = SingleConv(in_ch, list_ch[4], kernel_size=3, stride=1, padding=1)

    def forward(self, x):
        out_upsample = self.upsample(x)
        output = self.cov(out_upsample)

        return output


class Downsample(nn.Module):
    def __init__(self, in_ch, stride, list_ch):
        super(Downsample, self).__init__()
        self.maxpooling = nn.MaxPool3d(3, stride=stride, padding=1)
        self.newfeature = SingleConv(in_ch, list_ch[1], kernel_size=3, stride=stride, padding=1)

    def forward(self, x):
        maxpooling_out = self.maxpooling(x)
        print('maxpooling_out:')
        print(maxpooling_out.shape)
        newfeature_out = self.newfeature(x)
        print('newfeature_out:')
        print(newfeature_out.shape)
        downsample_out = torch.cat((maxpooling_out, newfeature_out), dim=1)
        return downsample_out


class Encoder(nn.Module):
    def __init__(self, in_ch, list_ch):
        super(Encoder, self).__init__()

        self.encoder_1 = nn.Sequential(
            DenseConv(in_ch, list_ch),  # DenseConv(in_ch, in_ch + 16),
            DenseConv(in_ch + list_ch[1], list_ch)  # DenseConv(in_ch + 16, in_ch + 32)
        )
        self.downsample_1 = Downsample(in_ch + list_ch[2], (2, 2, 2),  list_ch)
        self.encoder_2 = nn.Sequential(
            DenseConv(in_ch + list_ch[3], list_ch),  # DenseConv(in_ch + 48, in_ch + 64),
            DenseConv(in_ch + list_ch[4], list_ch)  # DenseConv(in_ch + 64, in_ch + 80)
        )
        self.downsample_2 = Downsample(in_ch + list_ch[5], (2, 2, 2), list_ch)
        self.encoder_3 = nn.Sequential(
            DenseConv(in_ch + list_ch[6], list_ch),  # DenseConv(in_ch + 96, in_ch + 112),
            DenseConv(in_ch + list_ch[7], list_ch)  # DenseConv(in_ch + 112, in_ch + 128)
        )
        self.downsample_3 = Downsample(in_ch + list_ch[8], (1, 2, 2), list_ch)
        self.encoder_4 = nn.Sequential(
            DenseConv(in_ch + list_ch[9], list_ch),  # DenseConv(in_ch + 144, in_ch + 160),
            DenseConv(in_ch + list_ch[10], list_ch)  # DenseConv(in_ch + 160, in_ch + 176)
        )
        self.downsample_4 = Downsample(in_ch + list_ch[11], (1, 2, 2),  list_ch)
        self.encoder_5 = nn.Sequential(
            DenseConv(in_ch + list_ch[12], list_ch),  # DenseConv(in_ch + 192, in_ch + 208),
            DenseConv(in_ch + list_ch[13], list_ch),  # DenseConv(in_ch + 208, in_ch + 224),
            DenseConv(in_ch + list_ch[14], list_ch),  # DenseConv(in_ch + 224, in_ch + 240),
            DenseConv(in_ch + list_ch[15], list_ch)  # DenseConv(in_ch + 240, in_ch + 256),
        )

    def forward(self, x):
        out_encoder_1 = self.encoder_1(x)  # 42
        #print('out_encoder_1:')
        #print(out_encoder_1.shape)
        out_downsample_1 = self.downsample_1(out_encoder_1)  # 58
        #print('out_downsample_1:')
        #print(out_downsample_1.shape)
        out_encoder_2 = self.encoder_2(out_downsample_1)
        #print('out_encoder_2:')
        #print(out_encoder_2.shape)
        out_downsample_2 = self.downsample_2(out_encoder_2)
        #print('out_downsample_2:')
        #print(out_downsample_2.shape)
        out_encoder_3 = self.encoder_3(out_downsample_2)
        #print('out_encoder_3:')
        #print(out_encoder_3.shape)
        out_downsample_3 = self.downsample_3(out_encoder_3)
        #print('out_downsample_3:')
        #print(out_downsample_3.shape)
        out_encoder_4 = self.encoder_4(out_downsample_3)
        #print('out_encoder_4:')
        #print(out_encoder_4.shape)
        out_downsample_4 = self.downsample_4(out_encoder_4)
        #print('out_downsample_4:')
        #print(out_downsample_4.shape)
        out_encoder_5 = self.encoder_5(out_downsample_4)
        #print('out_encoder_5:')
       # print(out_encoder_5.shape)

        return [out_encoder_1, out_encoder_2, out_encoder_3, out_encoder_4, out_encoder_5]  # 返回一个列表


class Decoder(nn.Module):
    def __init__(self, in_ch, list_ch):
        super(Decoder, self).__init__()

        self.upsample_4 = Upsample(in_ch + list_ch[16], (1, 2, 2), list_ch)
        self.decoder_4 = nn.Sequential(
            DenseConv(in_ch + list_ch[15], list_ch),  # DenseConv(in_ch + 240, in_ch + 256),
            DenseConv(in_ch + list_ch[16], list_ch)  # DenseConv(in_ch + 256, in_ch + 272)
        )
        self.upsample_3 = Upsample(in_ch + list_ch[17], (1, 2, 2), list_ch)
        self.decoder_3 = nn.Sequential(
            DenseConv(in_ch + list_ch[12], list_ch),  # DenseConv(in_ch + 192, in_ch + 208),
            DenseConv(in_ch + list_ch[13], list_ch)  # DenseConv(in_ch + 208, in_ch + 224)
        )
        self.upsample_2 = Upsample(in_ch + list_ch[14], 2, list_ch)
        self.decoder_2 = nn.Sequential(
            DenseConv(in_ch + list_ch[9], list_ch),  # DenseConv(in_ch + 144, in_ch + 160),
            DenseConv(in_ch + list_ch[10], list_ch)  # DenseConv(in_ch + 160, in_ch + 176)
        )
        self.upsample_1 = Upsample(in_ch + list_ch[11], 2, list_ch)
        self.decoder_1 = nn.Sequential(
            DenseConv(in_ch + list_ch[6], list_ch),  # DenseConv(in_ch + 96, in_ch + 112),
            DenseConv(in_ch + list_ch[7], list_ch),  # DenseConv(in_ch + 112, in_ch + 128)
        )

    def forward(self, out_encoder):
        # out_encoder_1, out_encoder_2, out_encoder_3, out_encoder_4, out_encoder_5 = out_encoder

        out_upsample_4 = torch.cat((out_encoder[3], self.upsample_4(out_encoder[4])), dim=1)  # 250
        print('out_upsample_4:')
        print(out_upsample_4.shape)
        out_decoder_4 = self.decoder_4(out_upsample_4)  # 282
        print('out_decoder_4:')
        print(out_decoder_4.shape)
        out_upsample_3 = torch.cat((out_encoder[2], self.upsample_3(out_decoder_4)), dim=1)  # 202
        print('out_upsample_3:')
        print(out_upsample_3.shape)
        out_decoder_3 = self.decoder_3(out_upsample_3)  # 234
        print('out_decoder_3:')
        print(out_decoder_3.shape)
        out_upsample_2 = torch.cat((out_encoder[1], self.upsample_2(out_decoder_3)), dim=1)  # 154
        print('out_upsample_2:')
        print(out_upsample_2.shape)
        out_decoder_2 = self.decoder_2(out_upsample_2)  # 186
        print('out_decoder_2:')
        print(out_decoder_2.shape)
        out_upsample_1 = torch.cat((out_encoder[0], self.upsample_1(out_decoder_2)), dim=1)  # 106
        print('out_upsample_1:')
        print(out_upsample_1.shape)
        out_decoder_1 = self.decoder_1(out_upsample_1)  # 138

        return out_decoder_1


class Net(nn.Module):
    def __init__(self, in_ch, list_ch):
        super(Net, self).__init__()
        self.encoder = Encoder(in_ch, list_ch)
        self.decoder = Decoder(in_ch, list_ch)

        # init
        self.initialize()

    @staticmethod  # 可以BaseUNet.init_conv_IN调用#
    def init_conv_IN(modules):
        for m in modules():
            if isinstance(m, nn.Conv3d):
                nn.init.kaiming_uniform_(m.weight, mode='fan_in', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.)
            elif isinstance(m, nn.InstanceNorm3d):
                nn.init.constant_(m.weight, 1.)
                nn.init.constant_(m.bias, 0.)

    def initialize(self):
        print('# random init encoder weight using nn.init.kaiming_uniform !')
        self.init_conv_IN(self.decoder.modules)
        print('# random init decoder weight using nn.init.kaiming_uniform !')
        self.init_conv_IN(self.encoder.modules)

    def forward(self, x):
        print('Doing encoder..........')
        out_encoder = self.encoder(x)
        print('out_encoder:')
        print(out_encoder[0].shape)
        print(out_encoder[1].shape)
        print(out_encoder[2].shape)
        print(out_encoder[3].shape)
        print(out_encoder[4].shape)
        print('Doing decoder..........')
        out_decoder = self.decoder(out_encoder)
        print('out_decoder:')
        print(out_decoder.shape)

        return out_decoder


class Model(nn.Module):
    def __init__(self, in_ch, out_ch, list_ch_A):
        super(Model, self).__init__()
        # list_ch_A=[-1, 16, 32, 48, 64, 80, 96, 112, 128, 144, 160, 176, 192, 208, 224, 240, 256, 272]
        # list_ch_B=[-1, 96, 112, 144, 160, 176, 192, 208, 224, 240, 256, 272]
        self.net_A = Net(in_ch, list_ch_A)
        self.net_B = Net(2 * in_ch + list_ch_A[8], list_ch_A)

        self.cov_out_A = SingleConv(in_ch + list_ch_A[8], out_ch, kernel_size=3, stride=1, padding=1)
        self.cov_out_B = SingleConv(2 * in_ch + list_ch_A[16], out_ch, kernel_size=3, stride=1, padding=1)

    def forward(self, x):
        # input_: (b, num_classes, D, H, W)
        print('Doing Net_A')
        out_net_A = self.net_A(x)
        print('out_net_A:')
        print(out_net_A.shape)
        print('Doing Net_B')
        out_net_B = self.net_B(torch.cat((out_net_A, x), dim=1))
        print('out_net_B:')
        print(out_net_B.shape)

        output_A = self.cov_out_A(out_net_A)
        output_B = self.cov_out_B(out_net_B)
        print('output_A:')
        print(output_A.shape)
        print('output_B:')
        print(output_B.shape)

        return [output_A, output_B]
