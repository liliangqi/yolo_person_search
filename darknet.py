# -----------------------------------------------------
# Person Search Architecture -- Darknet
#
# Author: Liangqi Li, BinWang Shu and Ayoosh Kathuria
# Creating Date: May 17, 2018
# Latest rectifying: May 17, 2018
# -----------------------------------------------------
import torch
import torch.nn as nn
import numpy as np
import yaml


class Conv(nn.Module):
    """Convolutional layer with BN and LeakyReLU"""

    def __init__(self, in_c, out_c, k_s=3, stride=1, pad=1):
        super().__init__()
        self.conv = nn.Conv2d(in_c, out_c, k_s, stride, pad, bias=False)
        self.batchnorm = nn.BatchNorm2d(out_c)
        self.relu = nn.LeakyReLU(0.1, inplace=True)

    def forward(self, x):
        out = self.relu(self.batchnorm(self.conv(x)))
        return out

    def __iter__(self):
        return (i for i in (self.conv, self.batchnorm, self.relu))


class Residual(nn.Module):
    """Residual unit"""

    def __init__(self, in_c):
        super().__init__()
        self.conv1 = Conv(in_c, in_c//2, 1, 1, 0)
        self.conv2 = Conv(in_c//2, in_c, 3, 1, 1)

    def forward(self, x):
        out = self.conv2(self.conv1(x))
        return x + out


class Reorg(nn.Module):
    """Reshape layer used in YOLOv2"""

    def __init__(self, stride):
        super().__init__()
        self.stride = stride

    def forward(self, x):

        assert x.data.dim() == 4
        bs = x.data.size(0)
        ch = x.data.size(1)
        h = x.data.size(2)
        w = x.data.size(3)
        assert h % self.stride == 0
        assert w % self.stride == 0
        hs = self.stride
        ws = self.stride
        x = x.view(bs, ch, h//hs, hs, w//ws, ws).transpose(3, 4).contiguous()
        x = x.view(bs, ch, h//hs*w//ws, hs*ws).transpose(2, 3).contiguous()
        x = x.view(bs, ch, hs*ws, h//hs, w//ws).transpose(1, 2).contiguous()
        x = x.view(bs, hs*ws*ch, h//hs, w//ws)

        return x


class Darknet19(nn.Module):
    """Darknet19 for Person Search"""

    def __init__(self, pre_model=None):
        super().__init__()
        self.conv0 = Conv(3, 32)
        self.pool1 = nn.MaxPool2d(2, 2)
        self.conv2 = Conv(32, 64)
        self.pool3 = nn.MaxPool2d(2, 2)
        self.conv4 = Conv(64, 128)
        self.conv5 = Conv(128, 64, 1, 1, 0)
        self.conv6 = Conv(64, 128)
        self.pool7 = nn.MaxPool2d(2, 2)
        self.conv8 = Conv(128, 256)
        self.conv9 = Conv(256, 128, 1, 1, 0)
        self.conv10 = Conv(128, 256)
        self.pool11 = nn.MaxPool2d(2, 2)
        self.conv12 = Conv(256, 512)
        self.conv13 = Conv(512, 256, 1, 1, 0)
        self.conv14 = Conv(256, 512)
        self.conv15 = Conv(512, 256, 1, 1, 0)
        self.conv16 = Conv(256, 512)
        self.pool17 = nn.MaxPool2d(2, 2)
        self.conv18 = Conv(512, 1024)
        self.conv19 = Conv(1024, 512, 1, 1, 0)
        self.conv20 = Conv(512, 1024)
        self.conv21 = Conv(1024, 512, 1, 1, 0)
        self.conv22 = Conv(512, 1024)

        self.conv23 = Conv(1024, 1024)
        self.conv24 = Conv(1024, 1024)

        self.conv26 = Conv(512, 64, 1, 1, 0)
        self.reorg27 = Reorg(2)

        self.conv29 = Conv(1280, 1024)
        self.conv30 = nn.Conv2d(1024, 125, 1, 1)  # TODO: change 125 to 25

        if pre_model is not None:
            self.load_dark_weights(pre_model)

    def forward(self, x):

        x = self.conv0(x)
        x = self.conv2(self.pool1(x))
        x = self.conv6(self.conv5(self.conv4(self.pool3(x))))
        x = self.conv10(self.conv9(self.conv8(self.pool7(x))))
        x = self.conv16(self.conv15(self.conv14(self.conv13(self.conv12(
            self.pool11(x))))))
        route_1 = x

        x = self.conv22(self.conv21(self.conv20(self.conv19(self.conv18(
            self.pool17(x))))))

        x = self.conv24(self.conv23(x))
        route_2 = x

        route_3 = self.reorg27(self.conv26(route_1))
        x = torch.cat((route_3, route_2), 1)

        x = self.conv29(x)
        output = self.conv30(x)

        return output

    def load_trained_model(self, state_dict):
        nn.Module.load_state_dict(
            self, {k: state_dict[k] for k in list(self.state_dict())})

    def load_dark_weights(self, weight_file):

        conv_list = []
        conv_list.append(self.conv0)
        conv_list.append(self.conv2)
        conv_list.append(self.conv4)
        conv_list.append(self.conv5)
        conv_list.append(self.conv6)
        conv_list.append(self.conv8)
        conv_list.append(self.conv9)
        conv_list.append(self.conv10)
        conv_list.append(self.conv12)
        conv_list.append(self.conv13)
        conv_list.append(self.conv14)
        conv_list.append(self.conv15)
        conv_list.append(self.conv16)
        conv_list.append(self.conv18)
        conv_list.append(self.conv19)
        conv_list.append(self.conv20)
        conv_list.append(self.conv21)
        conv_list.append(self.conv22)

        # Open the weights file
        fp = open(weight_file, 'rb')
        # The first 3 values are header information
        header = np.fromfile(fp, dtype=np.int32, count=4)
        # The rest of the values are the weights
        weights = np.fromfile(fp, dtype=np.float32)
        fp.close()

        ptr = 0
        conv_num = 0
        for model in conv_list:
            conv, bn, _ = model
            conv_num += 1

            # Get the number of weights of BN layer
            num_bn_biases = bn.bias.numel()
            # Load the weights
            bn_biases = torch.from_numpy(weights[ptr: ptr + num_bn_biases])
            ptr += num_bn_biases
            bn_weights = torch.from_numpy(weights[ptr: ptr + num_bn_biases])
            ptr += num_bn_biases
            bn_running_mean = torch.from_numpy(
                weights[ptr: ptr + num_bn_biases])
            ptr += num_bn_biases
            bn_running_var = torch.from_numpy(
                weights[ptr: ptr + num_bn_biases])
            ptr += num_bn_biases
            # Cast the loaded weights into dims of the model weights
            bn_biases = bn_biases.view_as(bn.bias.data)
            bn_weights = bn_weights.view_as(bn.weight.data)
            bn_running_mean = bn_running_mean.view_as(bn.running_mean)
            bn_running_var = bn_running_var.view_as(bn.running_var)
            # Copy the data to the model
            bn.bias.data.copy_(bn_biases)
            bn.weight.data.copy_(bn_weights)
            bn.running_mean.copy_(bn_running_mean)
            bn.running_var.copy_(bn_running_var)

            # Load the weights for the Conv layer
            num_weights = conv.weight.numel()
            # Do the same as above for weights
            conv_weights = torch.from_numpy(weights[ptr: ptr + num_weights])
            ptr = ptr + num_weights

            conv_weights = conv_weights.view_as(conv.weight.data)
            conv.weight.data.copy_(conv_weights)


class Darknet53(nn.Module):
    """Darknet53 for SIPN"""

    def __init__(self, pre_model=None, is_train=True):
        super().__init__()
        self.conv0 = Conv(3, 32, 3, 1, 1)
        self.down1 = Conv(32, 64, 3, 2, 1)  # downsample1
        self.res1 = Residual(64)
        self.down2 = Conv(64, 128, 3, 2, 1)
        self.res2 = nn.Sequential(
            Residual(128),
            Residual(128))
        self.down3 = Conv(128, 256, 3, 2, 1)
        self.res3 = nn.Sequential(
            Residual(256),
            Residual(256),
            Residual(256),
            Residual(256),
            Residual(256),
            Residual(256),
            Residual(256),
            Residual(256))
        self.down4 = Conv(256, 512, 3, 2, 1)  # downsample4 1/16
        self.res4 = nn.Sequential(
            Residual(512),
            Residual(512),
            Residual(512),
            Residual(512),
            Residual(512),
            Residual(512),
            Residual(512),
            Residual(512))
        self.conv5 = Conv(512, 1024, 3, 1, 1)  # different from original one
        self.res5 = nn.Sequential(
            Residual(1024),
            Residual(1024),
            Residual(1024),
            Residual(1024))

        self.is_train = is_train
        self.net_conv_channels = 512
        self.fc7_channels = 1024

        if self.is_train:
            self.load_dark_weights(pre_model)

        with open('config.yml', 'r') as f:
            config = yaml.load(f)

        self.head, self.tail = self.initialize(config['darknet_fix_bn'])

    def load_dark_weights(self, weight_file):

        conv_list = []
        conv_list.append(self.conv0)
        conv_list.append(self.down1)
        conv_list.append(self.res1.conv1)
        conv_list.append(self.res1.conv2)
        conv_list.append(self.down2)
        for res in self.res2:
            conv_list.append(res.conv1)
            conv_list.append(res.conv2)
        conv_list.append(self.down3)
        for res in self.res3:
            conv_list.append(res.conv1)
            conv_list.append(res.conv2)
        conv_list.append(self.down4)
        for res in self.res4:
            conv_list.append(res.conv1)
            conv_list.append(res.conv2)
        conv_list.append(self.conv5)
        for res in self.res5:
            conv_list.append(res.conv1)
            conv_list.append(res.conv2)

        # Open the weights file
        fp = open(weight_file, 'rb')
        # The first 4 values are header information
        header = np.fromfile(fp, dtype=np.int32, count=5)
        # The rest of the values are the weights
        weights = np.fromfile(fp, dtype=np.float32)
        fp.close()

        ptr = 0
        conv_num = 0
        for model in conv_list:
            conv, bn, _ = model
            conv_num += 1

            # Get the number of weights of BN layer
            num_bn_biases = bn.bias.numel()
            # Load the weights
            bn_biases = torch.from_numpy(weights[ptr: ptr+num_bn_biases])
            ptr += num_bn_biases
            bn_weights = torch.from_numpy(weights[ptr: ptr+num_bn_biases])
            ptr += num_bn_biases
            bn_running_mean = torch.from_numpy(weights[ptr: ptr+num_bn_biases])
            ptr += num_bn_biases
            bn_running_var = torch.from_numpy(weights[ptr: ptr+num_bn_biases])
            ptr += num_bn_biases
            # Cast the loaded weights into dims of the model weights
            bn_biases = bn_biases.view_as(bn.bias.data)
            bn_weights = bn_weights.view_as(bn.weight.data)
            bn_running_mean = bn_running_mean.view_as(bn.running_mean)
            bn_running_var = bn_running_var.view_as(bn.running_var)
            # Copy the data to the model
            bn.bias.data.copy_(bn_biases)
            bn.weight.data.copy_(bn_weights)
            bn.running_mean.copy_(bn_running_mean)
            bn.running_var.copy_(bn_running_var)

            # Load the weights for the Conv layer
            num_weights = conv.weight.numel()
            # Do the same as above for weights
            conv_weights = torch.from_numpy(weights[ptr: ptr+num_weights])
            ptr = ptr + num_weights

            conv_weights = conv_weights.view_as(conv.weight.data)
            conv.weight.data.copy_(conv_weights)

    def initialize(self, fix):

        def set_bn_fix(m):
            class_name = m.__class__.__name__
            if class_name.find('BatchNorm') != -1:
                for p in m.parameters():
                    p.requires_grad = False

        # TODO: Whether or not fix the first conv layer
        if fix:
            # TODO: Check if bn is fixed after model.train()
            self.apply(set_bn_fix)

        head = nn.Sequential(self.conv0, self.down1, self.res1,
                                  self.down2, self.res2, self.down3,
                                  self.res3, self.down4, self.res4)
        tail = nn.Sequential(self.conv5, self.res5)

        return head, tail


if __name__ == '__main__':

    dk = Darknet19('darknet19_448.conv.23')
