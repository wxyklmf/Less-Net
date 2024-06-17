import torch
import torch.nn as nn
from nets.MobileViT import *


class CA_Block(nn.Module):
    def __init__(self, channel, reduction=16):
        super(CA_Block, self).__init__()

        self.conv_1x1 = nn.Conv2d(in_channels=channel, out_channels=channel // reduction, kernel_size=1, stride=1,
                                  bias=False)

        self.relu = nn.ReLU()
        self.bn = nn.BatchNorm2d(channel // reduction)

        self.F_h = nn.Conv2d(in_channels=channel // reduction, out_channels=channel, kernel_size=1, stride=1,
                             bias=False)
        self.F_w = nn.Conv2d(in_channels=channel // reduction, out_channels=channel, kernel_size=1, stride=1,
                             bias=False)

        self.sigmoid_h = nn.Sigmoid()
        self.sigmoid_w = nn.Sigmoid()

    def forward(self, x):
        _, _, h, w = x.size()

        x_h = torch.mean(x, dim=3, keepdim=True).permute(0, 1, 3, 2)
        x_w = torch.mean(x, dim=2, keepdim=True)

        x_cat_conv_relu = self.relu(self.bn(self.conv_1x1(torch.cat((x_h, x_w), 3))))

        x_cat_conv_split_h, x_cat_conv_split_w = x_cat_conv_relu.split([h, w], 3)

        s_h = self.sigmoid_h(self.F_h(x_cat_conv_split_h.permute(0, 1, 3, 2)))
        s_w = self.sigmoid_w(self.F_w(x_cat_conv_split_w))

        out = x * s_h.expand_as(x) * s_w.expand_as(x)
        return out


class MultiScaleFeatureAggregation(nn.Module):
    def __init__(self, channel_sizes, output_size):
        super(MultiScaleFeatureAggregation, self).__init__()

        # 定义增强层，用于融合所有特征
        self.enhancer = nn.Sequential(
            nn.Conv2d(480, 256, kernel_size=1, bias=False),
            CA_Block(256),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True)
        )

        # 指定输出尺寸
        self.output_size = output_size

    def forward(self, *features):
        # 将所有特征调整到相同的空间尺寸
        resized_features = [F.interpolate(feat, size=self.output_size, mode='bilinear', align_corners=False) for feat in features]
        # 按通道连接所有特征
        concatenated_features = torch.cat(resized_features, dim=1)
        # print(concatenated_features.shape)
        # 应用增强层
        enhanced_features = self.enhancer(concatenated_features)
        return enhanced_features

class FeatureFusionModule(nn.Module):
    def __init__(self, global_channels, feature_channels, output_size):
        super(FeatureFusionModule, self).__init__()
        self.resize = nn.Upsample(size=output_size, mode='bilinear', align_corners=False)
        self.conv = nn.Conv2d(global_channels, feature_channels, kernel_size=1, stride=1, padding=0)

    def forward(self, global_feature, feature):
        resized_global = self.resize(global_feature)
        adapted_global = self.conv(resized_global)
        # print(adapted_global.shape)
        # print(feature.shape)
        return adapted_global + feature

class unetUp(nn.Module):
    def __init__(self, in_size, out_size):
        super(unetUp, self).__init__()
        self.conv1 = nn.Conv2d(in_size, out_size, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(out_size, out_size, kernel_size=3, padding=1)
        self.up = nn.UpsamplingBilinear2d(scale_factor=2)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, inputs1, inputs2):
        outputs = torch.cat([inputs1, self.up(inputs2)], 1)
        outputs = self.conv1(outputs)
        outputs = self.relu(outputs)
        outputs = self.conv2(outputs)
        outputs = self.relu(outputs)
        return outputs

class Unet(nn.Module):
    def __init__(self, num_classes=21, pretrained=False, backbone='mobilevit'):
        super(Unet, self).__init__()
        self.backbone = mobile_vit_small()
        in_filters = [160, 320, 608, 288]
        out_filters = [64, 128, 256, 512]

        feature_sizes = [(256, 256), (128, 128), (64, 64), (32, 32), (16, 16)]  # 特征图的尺寸

        self.feature_aggregator = MultiScaleFeatureAggregation(in_filters, 256)
        a_filters = [32, 64, 96, 128, 160]

        self.fusion_modules = nn.ModuleList([
            FeatureFusionModule(256, a_filters[i], feature_sizes[i]) for i in range(len(a_filters))
        ])
        # upsampling
        # 64,64,512
        self.up_concat4 = unetUp(in_filters[3], out_filters[3])
        # 128,128,256
        self.up_concat3 = unetUp(in_filters[2], out_filters[2])
        # 256,256,128
        self.up_concat2 = unetUp(in_filters[1], out_filters[1])
        # 512,512,64
        self.up_concat1 = unetUp(in_filters[0], out_filters[0])
        self.up_conv = nn.Sequential(
            nn.UpsamplingBilinear2d(scale_factor=2),
            nn.Conv2d(out_filters[0], out_filters[0], kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(out_filters[0], out_filters[0], kernel_size=3, padding=1),
            nn.ReLU(),
        )
        self.final = nn.Conv2d(out_filters[0], num_classes, 1)

    def forward(self, inputs):
        [feat1, feat2, feat3, feat4, feat5] = self.backbone.forward(inputs)
        feats = self.backbone.forward(inputs)
        global_feature = self.feature_aggregator(*feats)

        fused_feats = [module(global_feature, feat) for module, feat in zip(self.fusion_modules, feats)]

        up4 = self.up_concat4(fused_feats[3], fused_feats[4])
        up3 = self.up_concat3(fused_feats[2], up4)
        up2 = self.up_concat2(fused_feats[1], up3)
        up1 = self.up_concat1(fused_feats[0], up2)


        up1 = self.up_conv(up1)
        final = self.final(up1)

        return final

if __name__ == '__main__':
    model = Unet(2, False)
    random_input = torch.randn(1, 3, 512, 512)
    x1 = model(random_input)
    print(x1.shape)

