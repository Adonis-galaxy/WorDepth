import torch
import torch.nn as nn
import torch.nn.functional as F

from .swin_transformer import SwinTransformer
from .loss import SILogLoss
import random
########################################################################################################################

class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()

        if not mid_channels:
            mid_channels = out_channels

        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, groups=4),
            nn.InstanceNorm2d(mid_channels),
            nn.LeakyReLU(),
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1),
            nn.InstanceNorm2d(out_channels),
            nn.LeakyReLU(),
        )

        self.bt = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)

    def forward(self, x):
        skip = self.bt(x)

        x = self.channel_shuffle(x, 4)

        x = self.conv1(x)

        x = self.conv2(x)

        return x + skip

    def channel_shuffle(self, x, groups):
        batchsize, num_channels, height, width = x.shape

        channels_per_group = num_channels // groups
        # reshape
        x = x.view(batchsize, groups, channels_per_group, height, width)

        x = torch.transpose(x, 1, 2).contiguous()

        # flatten
        x = x.view(batchsize, -1, height, width)

        return x


class Up(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()

        self.up = nn.Upsample(
            scale_factor=2, mode='bilinear', align_corners=True)
        self.conv = DoubleConv(
            in_channels, out_channels, in_channels)

    def forward(self, x1, x2=None):
        x1 = self.up(x1)
        # input is CHW
        if x2 is not None:
            diffY = x2.size()[2] - x1.size()[2]
            diffX = x2.size()[3] - x1.size()[3]
            if diffX > 0 or diffY > 0:
                x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                                diffY // 2, diffY - diffY // 2])
            x = torch.cat([x2, x1], dim=1)
        else:
            x = x1
        return self.conv(x)

class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels, prior_mean=1.54):
        super(OutConv, self).__init__()

        self.prior_mean = prior_mean
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)

    def forward(self, x):
        return torch.exp(self.conv(x) + self.prior_mean)


class EpsLayer(nn.Module):
    def __init__(self, in_channels, h, w):
        super(EpsLayer, self).__init__()

        self.gr = 16

        self.post = nn.Conv2d(512, 8*self.gr, kernel_size=3, padding=1)

        self.eps_net = nn.Conv2d(512+128+128, 512, kernel_size=1, padding=0)

    def forward(self, x, mean, std):

        B, hidden_dim, H, W = x.shape

        mean = F.interpolate(mean.unsqueeze(-1).unsqueeze(-1), size=(H, W))  # B*128 -> B*128*30*40
        std = F.interpolate(std.unsqueeze(-1).unsqueeze(-1), size=(H, W))  # B*128 -> B*128*30*40
        x = self.eps_net(torch.cat((x, mean, std), dim=1))

        x = self.post(x)

        return x


class Refine(nn.Module):
    def __init__(self, c1, c2):
        super(Refine, self).__init__()

        s = c1 + c2
        self.fw = nn.Sequential(
            nn.Conv2d(s, s, kernel_size=3, padding=1),
            nn.LeakyReLU(),
            nn.Conv2d(s, c1, kernel_size=3, padding=1)
        )

        self.dw = nn.Sequential(
            nn.Conv2d(s, s, kernel_size=3, padding=1),
            nn.LeakyReLU(),
            nn.Conv2d(s, c2, kernel_size=3, padding=1)
        )

    def forward(self, feat, depth):
        cc = torch.cat([feat, depth], 1)
        feat_new = self.fw(cc)
        depth_new = self.dw(cc)
        return feat_new, depth_new


class MetricLayer(nn.Module):
    def __init__(self, c):
        super(MetricLayer, self).__init__()

        self.ln = nn.Sequential(
            nn.Linear(c, c//4),
            nn.LeakyReLU(),
            nn.Linear(c//4, 2)
        )

    def forward(self, x):

        x = x.squeeze(-1).squeeze(-1)
        x = self.ln(x)
        x = x.unsqueeze(-1).unsqueeze(-1)

        return x

class Text_Encoder(nn.Module):
    def __init__(self, hidden_dim=128):
        super().__init__()
        self.mean = nn.Sequential(
            nn.Linear(1024, 512),
            nn.LeakyReLU(),
            nn.Linear(512, 256),
            nn.LeakyReLU(),
            nn.Linear(256, hidden_dim)
        )

        self.deviation = nn.Sequential(
            nn.Linear(1024, 512),
            nn.LeakyReLU(),
            nn.Linear(512, 256),
            nn.LeakyReLU(),
            nn.Linear(256, hidden_dim)
        )

    def forward(self, text_feature_list):
        text_feature_list = text_feature_list.to(torch.float32)

        mean = self.mean(text_feature_list)
        logvar = self.deviation(text_feature_list)
        std = torch.exp(0.5 * logvar)

        return mean, std, logvar

class WorDepth(nn.Module):
    def __init__(self, pretrained=None, max_depth=10.0, prior_mean=1.54, si_lambda=0.85, img_size=(480, 640), weight_kld=1e-3, alter_prob=0.1):
        '''
        WorDepth Model Network class

        Arg(s):
            pretrained: bool
                whether the image encoder is pre-trained
            max_depth: int
                max depth in this dataset
            prior_mean: int
                prior depth mean for this dataset
            si_lambda: int
                lambda for loss scale invariant factor
            img_size: (int, int)
                input image size
        '''
        super().__init__()
        self.prior_mean = prior_mean
        self.SI_loss_lambda = si_lambda
        self.max_depth = max_depth

        pretrain_img_size = img_size
        patch_size = (4, 4)
        embed_dim = 192
        depths = [2, 2, 18, 2]
        num_heads = [6, 12, 24, 48]
        window_size = 12

        backbone_cfg = dict(
            pretrain_img_size=pretrain_img_size,
            patch_size=patch_size,
            embed_dim=embed_dim,
            depths=depths,
            num_heads=num_heads,
            window_size=window_size,
            ape=True,
            drop_rate=0.
        )

        self.backbone = SwinTransformer(**backbone_cfg)

        self.backbone.init_weights(pretrained=pretrained)

        self.up_4 = Up(1536 + 768, 512)
        self.up_3 = Up(512 + 384, 256)
        self.up_2 = Up(256 + 192, 64)

        self.outc = OutConv(128, 1, self.prior_mean)

        self.eps_layer = EpsLayer(512, img_size[0]//16, img_size[1]//16)

        self.ref_4 = Refine(512, 128)
        self.ref_3 = Refine(256, 128)
        self.ref_2 = Refine(64, 128)

        self.si_loss = SILogLoss(self.SI_loss_lambda, self.max_depth)

        self.mlayer = nn.Sequential(
            nn.AdaptiveMaxPool2d((1, 1)),
            MetricLayer(1536)
        )

        self.text_encoder = Text_Encoder(hidden_dim=128)
        self.weight_kld = weight_kld
        self.alter_prob = alter_prob

    def forward(self, image, text_feature_list, depth_gt=None, sample_from_gaussian_eval=None):
        '''
        Forwards the inputs through the network

        Arg(s):
            image: torch.Tensor[float32]
                N x 3 x Height (480 for nyu) x Width (640 for nyu)
            text_feature_list: torch.Tensor[float32]
                N x text_feat_dim(1024 by default)
            depth_gt: torch.Tensor[float32]
                N x 1 x Height (480 for nyu) x Width (640 for nyu)
        Returns:
            depth_pred: torch.Tensor[float32]
                N x 1 x Height (480 for nyu) x Width (640 for nyu)
        '''
        if random.random()<self.alter_prob:
            sample_from_gaussian = True
        else:
            sample_from_gaussian = False
        # For vis gaussian
        if sample_from_gaussian_eval is not None:
            sample_from_gaussian = sample_from_gaussian_eval
        x2, x3, x4, x5 = self.backbone(image)

        metric = self.mlayer(x5)

        x = self.up_4(x5, x4)
        B, _, H, W = x.shape
        hidden_dim = 128

        # Mean and Std generated by Text
        mean_txt, std_txt, log_var_txt = self.text_encoder(text_feature_list)  # B*128, global text feature for mean and std
        kld_text_mean_std_loss = torch.mean(-0.5 * torch.sum(1 + log_var_txt - mean_txt ** 2 - log_var_txt.exp(), dim=1), dim=0)

        if sample_from_gaussian is True:
            eps_img = torch.normal(mean=0, std=1, size=(B, hidden_dim, H, W)).to(x.device)
        else:
            eps_img = self.eps_layer(x, mean_txt.clone().detach(), std_txt.clone().detach())  # ([B, 128, 30, 40])
        mean_txt_mul = F.interpolate(mean_txt.unsqueeze(-1).unsqueeze(-1), size=(H, W))  # B*128 -> B*128*30*40
        std_txt_mul = F.interpolate(std_txt.unsqueeze(-1).unsqueeze(-1), size=(H, W))  # B*128 -> B*128*30*40

        # KLD for eps (n=B*30*40, dim=128)
        if sample_from_gaussian is True:
            kld_image_eps_loss = 0
        else:
            eps_for_cal_loss = eps_img.permute(0, 2, 3, 1)  # [B, 30, 40, 128])
            eps_for_cal_loss = eps_for_cal_loss.reshape(B * H * W, hidden_dim)  # n=B*30*40, num for eps instance, dim=128
            eps_mean = torch.mean(eps_for_cal_loss, dim=0, keepdim=True)
            eps_std = torch.std(eps_for_cal_loss, dim=0, keepdim=True)
            eps_logvar = 2 * torch.log(eps_std)
            kld_image_eps_loss = torch.mean(-0.5 * torch.sum(1 + eps_logvar - eps_mean ** 2 - eps_logvar.exp(), dim=1), dim=0)

        # The intuition is to use regional img feat to predict eps for each patch
        # Then sample the maximum likely layout for this patch from language generation
        d_feat = mean_txt_mul + std_txt_mul * eps_img  # B*128*30*40, upsample to be the depth feature
        #
        # Upsampling depth feature, get rid of visual signal when sample eps from Gaussian
        if sample_from_gaussian is True:
            x = torch.zeros_like(x)
            x3 = torch.zeros_like(x3)
            x2 = torch.zeros_like(x2)
        x, d_feat  = self.ref_4(x, d_feat)

        d_u4 = F.interpolate(d_feat, scale_factor=16, mode='bilinear', align_corners=True)

        x = self.up_3(x, x3)

        x, d_feat = self.ref_3(x, F.interpolate(d_feat, scale_factor=2, mode='bilinear', align_corners=True))

        d_u3 = F.interpolate(d_feat, scale_factor=8, mode='bilinear', align_corners=True)

        x = self.up_2(x, x2)

        x, d_feat = self.ref_2(x, F.interpolate(d_feat, scale_factor=2, mode='bilinear', align_corners=True))

        d_u2 = F.interpolate(d_feat, scale_factor=4, mode='bilinear', align_corners=True)

        d_feat = d_u2 + d_u3 + d_u4

        depth_pred = torch.sigmoid(metric[:, 0:1]) * (self.outc(d_feat) + torch.exp(metric[:, 1:2]))

        if self.training:
            loss = self.si_loss(depth_pred, depth_gt) + self.weight_kld * kld_image_eps_loss + self.weight_kld * kld_text_mean_std_loss
            # loss = self.si_loss(depth_pred, depth_gt)
            return depth_pred, loss
        else:
            return depth_pred
