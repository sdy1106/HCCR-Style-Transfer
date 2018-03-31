import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable


class ContentEncoder(nn.Module):
    def __init__(self, batch_size):
        super(ContentEncoder, self).__init__()
        self.conv1 = nn.Conv2d(batch_size, 64, kernel_size=5, stride=1, padding=2, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(128)
        self.conv3 = nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1, bias=False)
        self.bn3 = nn.BatchNorm2d(256)
        self.conv4 = nn.Conv2d(256, 512, kernel_size=3, stride=2, padding=1, bias=False)
        self.bn4 = nn.BatchNorm2d(512)
        self.conv5 = nn.Conv2d(512, 512, kernel_size=3, stride=2, padding=1, bias=False)
        self.bn5 = nn.BatchNorm2d(512)
        self.conv6 = nn.Conv2d(512, 512, kernel_size=3, stride=2, padding=1, bias=False)
        self.bn6 = nn.BatchNorm2d(512)
        self.conv7 = nn.Conv2d(512, 512, kernel_size=3, stride=2, padding=1, bias=False)
        self.bn7 = nn.BatchNorm2d(512)
        self.weight_init()

    def weight_init(self):
        nn.init.xavier_uniform(self.conv1.weight)
        nn.init.xavier_uniform(self.conv2.weight)
        nn.init.xavier_uniform(self.conv3.weight)
        nn.init.xavier_uniform(self.conv4.weight)
        nn.init.xavier_uniform(self.conv5.weight)
        nn.init.xavier_uniform(self.conv6.weight)
        nn.init.xavier_uniform(self.conv7.weight)

    def forward(self, input):
        x = F.leaky_relu(self.bn1(self.conv1(input)))
        x = F.leaky_relu(self.bn1(self.conv1(input)), negative_slope=0.2)
        x = F.leaky_relu(self.bn2(self.conv2(x)), negative_slope=0.2)
        x = F.leaky_relu(self.bn3(self.conv3(x)), negative_slope=0.2)
        x = F.leaky_relu(self.bn4(self.conv4(x)), negative_slope=0.2)
        x = F.leaky_relu(self.bn5(self.conv5(x)), negative_slope=0.2)
        x = F.leaky_relu(self.bn6(self.conv6(x)), negative_slope=0.2)
        output = F.leaky_relu(self.bn7(self.conv7(x)), negative_slope=0.2)

        return output


class StyleEncoder(nn.Module):
    def __init__(self, style_batch):
        super(StyleEncoder, self).__init__()
        self.conv1 = nn.Conv2d(style_batch, 64, kernel_size=5, stride=1, padding=2, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(128)
        self.conv3 = nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1, bias=False)
        self.bn3 = nn.BatchNorm2d(256)
        self.conv4 = nn.Conv2d(256, 512, kernel_size=3, stride=2, padding=1, bias=False)
        self.bn4 = nn.BatchNorm2d(512)
        self.conv5 = nn.Conv2d(512, 512, kernel_size=3, stride=2, padding=1, bias=False)
        self.bn5 = nn.BatchNorm2d(512)
        self.conv6 = nn.Conv2d(512, 512, kernel_size=3, stride=2, padding=1, bias=False)
        self.bn6 = nn.BatchNorm2d(512)
        self.conv7 = nn.Conv2d(512, 512, kernel_size=3, stride=2, padding=1, bias=False)
        self.bn7 = nn.BatchNorm2d(512)
        self.weight_init()

    def weight_init(self):
        nn.init.xavier_uniform(self.conv1.weight)
        nn.init.xavier_uniform(self.conv2.weight)
        nn.init.xavier_uniform(self.conv3.weight)
        nn.init.xavier_uniform(self.conv4.weight)
        nn.init.xavier_uniform(self.conv5.weight)
        nn.init.xavier_uniform(self.conv6.weight)
        nn.init.xavier_uniform(self.conv7.weight)

    def forward(self, input):
        x = F.leaky_relu(self.bn1(self.conv1(input)), negative_slope=0.2)
        x = F.leaky_relu(self.bn2(self.conv2(x)), negative_slope=0.2)
        x = F.leaky_relu(self.bn3(self.conv3(x)), negative_slope=0.2)
        x = F.leaky_relu(self.bn4(self.conv4(x)), negative_slope=0.2)
        x = F.leaky_relu(self.bn5(self.conv5(x)), negative_slope=0.2)
        x = F.leaky_relu(self.bn6(self.conv6(x)), negative_slope=0.2)
        output = F.leaky_relu(self.bn7(self.conv7(x)), negative_slope=0.2)

        return output


class Mixer(nn.Module):
    def __init__(self):
        super(Mixer, self).__init__()
        #self.mix = torch.nn.Parameter(torch.zeros(512, 512, 512)).cuda()
        self.mix = nn.Bilinear(512, 512, 512, bias=False)
        #self.weight_init()

    def weight_init(self):
        nn.init.xavier_uniform(self.mix.data)

    def forward(self, style, content):
        #print self.mix
        # batch_size = style.size(0)
        # out = Variable(torch.zeros((batch_size, style.size(1)))).cuda()
        # for i in range(batch_size):
        #     tmp = torch.matmul(style[i], self.mix)
        #     out[i] = torch.matmul(tmp, content[i])
        # return out
        out = self.mix(style, content)
        return out


class Decoder(nn.Module):
    def __init__(self):
        super(Decoder, self).__init__()
        self.Dconv1 = nn.ConvTranspose2d(512, 512, kernel_size=3, stride=2, output_padding=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(512)
        self.Dconv2 = nn.ConvTranspose2d(512, 512, kernel_size=3, stride=2, output_padding=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(512)
        self.Dconv3 = nn.ConvTranspose2d(512, 512, kernel_size=3, stride=2, output_padding=1, padding=1, bias=False)
        self.bn3 = nn.BatchNorm2d(512)
        self.Dconv4 = nn.ConvTranspose2d(512, 256, kernel_size=3, stride=2, output_padding=1, padding=1, bias=False)
        self.bn4 = nn.BatchNorm2d(256)
        self.Dconv5 = nn.ConvTranspose2d(256, 128, kernel_size=3, stride=2, output_padding=1, padding=1, bias=False)
        self.bn5 = nn.BatchNorm2d(128)
        self.Dconv6 = nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2, output_padding=1, padding=1, bias=False)
        self.bn6 = nn.BatchNorm2d(64)
        self.Dconv7 = nn.ConvTranspose2d(64, 1, kernel_size=5, stride=1, padding=2, bias=False)
        self.bn7 = nn.BatchNorm2d(1)

        self.weight_init()

    def weight_init(self):
        nn.init.xavier_uniform(self.Dconv1.weight)
        nn.init.xavier_uniform(self.Dconv2.weight)
        nn.init.xavier_uniform(self.Dconv3.weight)
        nn.init.xavier_uniform(self.Dconv4.weight)
        nn.init.xavier_uniform(self.Dconv5.weight)
        nn.init.xavier_uniform(self.Dconv6.weight)
        nn.init.xavier_uniform(self.Dconv7.weight)

    def forward(self, input):
        x = F.relu(self.bn1(self.Dconv1(input)))
        x = F.relu(self.bn2(self.Dconv2(x)))
        x = F.relu(self.bn3(self.Dconv3(x)))
        x = F.relu(self.bn4(self.Dconv4(x)))
        x = F.relu(self.bn5(self.Dconv5(x)))
        x = F.relu(self.bn6(self.Dconv6(x)))
        output = F.relu(self.bn7(self.Dconv7(x)))

        return output


class FullTransfer(nn.Module):
    def __init__(self, content_batch, style_batch):
        super(FullTransfer, self).__init__()
        self.Style = StyleEncoder(style_batch)
        self.Content = ContentEncoder(content_batch)
        self.Mixer = Mixer()
        self.Decoder = Decoder()

    def forward(self, style_input, content_input):
        style = self.Style.forward(style_input).squeeze()  # batch_size * 512
        content = self.Content.forward(content_input).squeeze()  # batch_size * 512
        mix = self.Mixer.forward(style, content)
        mix = mix.view(mix.size(0), mix.size(1), 1, 1)
        output = self.Decoder.forward(mix)

        return output


