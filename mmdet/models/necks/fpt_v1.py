# ---------------------------------------------------------------------------
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F

from mmcv.cnn import ConvModule, caffe2_xavier_init, constant_init, is_norm

BatchNorm2d = nn.BatchNorm2d

from dropblock import DropBlock2D
from ..builder import NECKS

# factorized attention !!!!!!!

@NECKS.register_module()
class FPT_v(nn.Module):
    """"https://arxiv.org/abs/2007.09451"""

    def __init__(self,
                 in_channels,
                 out_channels,
                 start_level=1,
                 end_level=-1,
                 add_extra_convs=True,  # use P6, P7
                 extra_convs_on_inputs=False,
                 num_outs=5,  # in = out
                 with_norm='none',
                 upsample_method='bilinear'):  #
        super(FPT_v, self).__init__()
        assert isinstance(in_channels, list)
        self.in_channels = in_channels
        self.feature_dim = out_channels
        self.num_ins = len(in_channels)
        self.num_outs = num_outs
        self.start_level = start_level
        self.end_level = end_level
        self.add_extra_convs = add_extra_convs
        self.extra_convs_on_inputs = extra_convs_on_inputs
        assert upsample_method in ['nearest', 'bilinear']

        def interpolate(input):
            return F.interpolate(input, scale_factor=2, mode=upsample_method,
                                 align_corners=False if upsample_method == 'bilinear' else None)

        self.fpn_upsample = interpolate

        assert with_norm in ['group_norm', 'batch_norm', 'none']
        if with_norm == 'batch_norm':
            norm = nn.BatchNorm2d
        elif with_norm == 'group_norm':
            def group_norm(num_channels):
                return nn.GroupNorm(32, num_channels)
            norm = group_norm

        # self-transformer
        self.st_p5 = SelfTrans(n_head=1, n_mix=2, d_model=out_channels, d_k=out_channels, d_v=out_channels)
        self.st_p4 = SelfTrans(n_head=1, n_mix=2, d_model=out_channels, d_k=out_channels, d_v=out_channels)
        self.st_p3 = SelfTrans(n_head=1, n_mix=2, d_model=out_channels, d_k=out_channels, d_v=out_channels)
        # self.st_p2 = SelfTrans(n_head=1, n_mix=2, d_model=out_channels, d_k=out_channels, d_v=out_channels)

        # grounding transformer
        self.gt_p4_p5 = GroundTrans(in_channels=out_channels, inter_channels=None, mode='dot', dimension=2,
                                    bn_layer=True)
        self.gt_p3_p4 = GroundTrans(in_channels=out_channels, inter_channels=None, mode='dot', dimension=2,
                                    bn_layer=True)
        self.gt_p3_p5 = GroundTrans(in_channels=out_channels, inter_channels=None, mode='dot', dimension=2,
                                    bn_layer=True)
        # self.gt_p2_p3 = GroundTrans(in_channels=out_channels, inter_channels=None, mode='dot', dimension=2,
        #                             bn_layer=True)
        # self.gt_p2_p4 = GroundTrans(in_channels=out_channels, inter_channels=None, mode='dot', dimension=2,
        #                             bn_layer=True)
        # self.gt_p2_p5 = GroundTrans(in_channels=out_channels, inter_channels=None, mode='dot', dimension=2,
        #                             bn_layer=True)

        # rendering transformer
        self.rt_p5_p4 = RenderTrans(channels_high=out_channels, channels_low=out_channels, upsample=False, layer_stride=1)
        self.rt_p5_p3 = RenderTrans(channels_high=out_channels, channels_low=out_channels, upsample=False, layer_stride=2)
        # self.rt_p5_p2 = RenderTrans(channels_high=out_channels, channels_low=out_channels, upsample=False, layer_stride=3)
        self.rt_p4_p3 = RenderTrans(channels_high=out_channels, channels_low=out_channels, upsample=False, layer_stride=1)
        # self.rt_p4_p2 = RenderTrans(channels_high=out_channels, channels_low=out_channels, upsample=False, layer_stride=2)
        # self.rt_p3_p2 = RenderTrans(channels_high=out_channels, channels_low=out_channels, upsample=False, layer_stride=1)
        drop_block = DropBlock2D(block_size=3, drop_prob=0.2)

        if with_norm != 'none':
            self.fpn_p5_1x1 = nn.Sequential(*[nn.Conv2d(in_channels[3], out_channels, 1, bias=False), norm(out_channels)])
            self.fpn_p4_1x1 = nn.Sequential(*[nn.Conv2d(in_channels[2], out_channels, 1, bias=False), norm(out_channels)])
            self.fpn_p3_1x1 = nn.Sequential(*[nn.Conv2d(in_channels[1], out_channels, 1, bias=False), norm(out_channels)])
            # self.fpn_p2_1x1 = nn.Sequential(*[nn.Conv2d(in_channels[0], out_channels, 1, bias=False), norm(out_channels)])

            self.fpt_p5 = nn.Sequential(
                *[nn.Conv2d(out_channels * 4, out_channels, 3, padding=1, bias=False), norm(out_channels)])  # *5 because 1(orignal) + 4 (results of FPT)
            self.fpt_p4 = nn.Sequential(
                *[nn.Conv2d(out_channels * 4, out_channels, 3, padding=1, bias=False), norm(out_channels)])
            self.fpt_p3 = nn.Sequential(
                *[nn.Conv2d(out_channels * 4, out_channels, 3, padding=1, bias=False), norm(out_channels)])
            # self.fpt_p2 = nn.Sequential(
            #     *[nn.Conv2d(out_channels * 5, out_channels, 3, padding=1, bias=False), norm(out_channels)])
        else:
            self.fpn_p5_1x1 = nn.Conv2d(in_channels[3], out_channels, 1)
            self.fpn_p4_1x1 = nn.Conv2d(in_channels[2], out_channels, 1)
            self.fpn_p3_1x1 = nn.Conv2d(in_channels[1], out_channels, 1)
            # self.fpn_p2_1x1 = nn.Conv2d(in_channels[0], out_channels, 1)

            self.fpt_p5 = nn.Conv2d(out_channels * 4, out_channels, 3, padding=1)  # *5 ecause 1+4
            self.fpt_p4 = nn.Conv2d(out_channels * 4, out_channels, 3, padding=1)
            self.fpt_p3 = nn.Conv2d(out_channels * 4, out_channels, 3, padding=1)
            # self.fpt_p2 = nn.Conv2d(out_channels * 5, out_channels, 3, padding=1)
            # add P6,P7
            self.fpt_p6 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=2, padding=1)
            self.fpt_p7 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=2, padding=1)

        # self.initialize()

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_uniform_(m.weight.data, a=1)  # caffe2_xavier_init(m)
                if m.bias is not None:
                    m.bias.data.zero_()

    def forward(self, inputs):
        fpn_p5_1 = self.fpn_p5_1x1(inputs[self.end_level])
        fpn_p4_1 = self.fpn_p4_1x1(inputs[self.start_level + 1])
        fpn_p3_1 = self.fpn_p3_1x1(inputs[self.start_level])
        # fpn_p2_1 = self.fpn_p2_1x1(inputs[self.start_level - 1])
        fpt_p5_out = torch.cat((self.st_p5(fpn_p5_1), self.rt_p5_p4(fpn_p5_1, fpn_p4_1),
                                self.rt_p5_p3(fpn_p5_1, fpn_p3_1), fpn_p5_1), 1)
        fpt_p4_out = torch.cat((self.st_p4(fpn_p4_1), self.rt_p4_p3(fpn_p4_1, fpn_p3_1),
                                self.gt_p4_p5(fpn_p4_1, fpn_p5_1), fpn_p4_1), 1)
        fpt_p3_out = torch.cat((self.st_p3(fpn_p3_1), self.gt_p3_p4(fpn_p3_1, fpn_p4_1),
                                self.gt_p3_p5(fpn_p3_1, fpn_p5_1), fpn_p3_1), 1)
        #fpt_p2_out = torch.cat((self.st_p2(fpn_p2_1), self.gt_p2_p3(fpn_p2_1, fpn_p3_1),
        #                         self.gt_p2_p4(fpn_p2_1, fpn_p4_1), self.gt_p2_p5(fpn_p2_1, fpn_p5_1), fpn_p2_1), 1)
        fpt_p5 = self.fpt_p5(fpt_p5_out)
        fpt_p4 = self.fpt_p4(fpt_p4_out)
        fpt_p3 = self.fpt_p3(fpt_p3_out)
        # fpt_p2 = self.fpt_p2(fpt_p2_out)
        fpt_p6 = self.fpt_p6(fpt_p5)
        fpt_p7 = self.fpt_p7(fpt_p6)
        '''
        fpt_p5 = drop_block(self.fpt_p5(fpt_p5_out))
        fpt_p4 = drop_block(self.fpt_p4(fpt_p4_out))
        fpt_p3 = drop_block(self.fpt_p3(fpt_p3_out))
        fpt_p2 = drop_block(self.fpt_p2(fpt_p2_out))
        '''
        return fpt_p3, fpt_p4, fpt_p5, fpt_p6, fpt_p7


class SelfTrans(nn.Module):
    def __init__(self, n_head, n_mix, d_model, d_k, d_v,
                 norm_layer=BatchNorm2d, kq_transform='conv', value_transform='conv',
                 pooling=True, concat=False, dropout=0.1):
        super(SelfTrans, self).__init__()

        self.n_head = n_head
        self.n_mix = n_mix
        self.d_k = d_k
        self.d_v = d_v

        self.pooling = pooling
        self.concat = concat

        if self.pooling:
            self.pool = nn.AvgPool2d(3, 2, 1, count_include_pad=False)
        if kq_transform == 'conv':
            self.conv_qs = nn.Conv2d(d_model, n_head * d_k, 1)
            nn.init.normal_(self.conv_qs.weight, mean=0, std=np.sqrt(2.0 / (d_model + d_k)))
        elif kq_transform == 'ffn':
            self.conv_qs = nn.Sequential(
                nn.Conv2d(d_model, n_head * d_k, 3, padding=1, bias=False),
                norm_layer(n_head * d_k),
                nn.ReLU(True),
                nn.Conv2d(n_head * d_k, n_head * d_k, 1),
            )
            nn.init.normal_(self.conv_qs[-1].weight, mean=0, std=np.sqrt(1.0 / d_k))
        elif kq_transform == 'dffn':
            self.conv_qs = nn.Sequential(
                nn.Conv2d(d_model, n_head * d_k, 3, padding=4, dilation=4, bias=False),
                norm_layer(n_head * d_k),
                nn.ReLU(True),
                nn.Conv2d(n_head * d_k, n_head * d_k, 1),
            )
            nn.init.normal_(self.conv_qs[-1].weight, mean=0, std=np.sqrt(1.0 / d_k))
        else:
            raise NotImplemented

        self.conv_ks = self.conv_qs
        if value_transform == 'conv':
            self.conv_vs = nn.Conv2d(d_model, n_head * d_v, 1)
        else:
            raise NotImplemented

        nn.init.normal_(self.conv_vs.weight, mean=0, std=np.sqrt(2.0 / (d_model + d_v)))

        self.attention = MixtureOfSoftMax(n_mix=n_mix, d_k=d_k)

        self.conv = nn.Conv2d(n_head * d_v, d_model, 1, bias=False)
        self.norm_layer = norm_layer(d_model)

    def forward(self, x):
        residual = x
        d_k, d_v, n_head = self.d_k, self.d_v, self.n_head
        b_, c_, h_, w_ = x.size()
        if self.pooling:
            # qt = self.conv_ks(x).view(b_ * n_head, d_k, h_ * w_)
            """
            qt = self.conv_qs(x).view(b_ * n_head, d_k, h_ * w_)
            kt = self.conv_ks(self.pool(x)).view(b_ * n_head, d_k, -1)
            vt = self.conv_vs(self.pool(x)).view(b_ * n_head, d_v, -1)
            """
            kt = self.conv_ks(x).view(b_ * n_head, d_k, h_ * w_)
            qt = self.conv_qs(self.pool(x)).view(b_ * n_head, d_k, -1)
            vt = self.conv_vs(self.pool(x)).view(b_ * n_head, d_v, -1)
        else:
            kt = self.conv_ks(x).view(b_ * n_head, d_k, h_ * w_)
            # qt = kt
            qt = self.conv_qs(x).view(b_ * n_head, d_k, h_ * w_)
            vt = self.conv_vs(x).view(b_ * n_head, d_v, h_ * w_)

        output, attn = self.attention(qt, kt, vt)

        output = output.transpose(1, 2).contiguous().view(b_, n_head * d_v, h_, w_)

        output = self.conv(output)
        if self.concat:
            output = torch.cat((self.norm_layer(output), residual), 1)
        else:
            output = self.norm_layer(output) + residual
        return output


class MixtureOfSoftMax(nn.Module):
    """"https://arxiv.org/pdf/1711.03953.pdf"""

    def __init__(self, n_mix, d_k, attn_dropout=0.1):
        super(MixtureOfSoftMax, self).__init__()
        self.temperature = np.power(d_k, 0.5)
        self.n_mix = n_mix
        self.att_drop = attn_dropout
        self.dropout = nn.Dropout(attn_dropout)
        self.softmax1 = nn.Softmax(dim=1)  # the sum of each tensor's column equal to 1
        self.softmax2 = nn.Softmax(dim=2)  # the sum of each tensor's row equal to 1
        self.d_k = d_k
        if n_mix > 1:
            self.weight = nn.Parameter(torch.Tensor(n_mix, d_k))
            std = np.power(n_mix, -0.5)
            self.weight.data.uniform_(-std, std)

    def forward(self, qt, kt, vt):
        # B, d_k, N = qt.size()
        B, d_k, N = kt.size()
        m = self.n_mix
        assert d_k == self.d_k
        d = d_k // m
        if m > 1:
            bar_kt = torch.mean(kt, 2, True)
            pi = self.softmax1(torch.matmul(self.weight, bar_kt)).view(B * m, 1, 1)
        """
        q = qt.view(B * m, d, N).transpose(1, 2)
        N2 = kt.size(2)
        kt = kt.view(B * m, d, N2)
        v = vt.transpose(1, 2)
        
        attn = torch.bmm(q, kt)  # bmm
        attn = attn / self.temperature
        attn = self.softmax2(attn)
        attn = self.dropout(attn)
        if m > 1:
            attn = (attn * pi).view(B, m, N, N2).sum(1)
        output = torch.bmm(attn, v)
        """
        kt = kt.view(B * m, d, N).transpose(1, 2)
        N2 = vt.size(2)
        vt = vt.view(B * m, d, N2)
        q = qt.transpose(1, 2)

        attn = self.softmax2(kt)
        attn = torch.bmm(attn, vt)
        attn = self.dropout(attn)
        if m > 1:
            attn = (attn * pi).view(B, m, N, N2).sum(1)
        attn = attn / self.temperature
        output = torch.bmm(attn, q)
        return output, attn


class GroundTrans(nn.Module):
    def __init__(self, in_channels, inter_channels=None, mode='dot', dimension=2, bn_layer=True):
        super(GroundTrans, self).__init__()
        assert dimension in [1, 2, 3]
        if mode not in ['gaussian', 'embedded', 'dot', 'concatenate']:
            raise ValueError('`mode` must be one of `gaussian`, `embedded`, `dot` or `concatenate`')

        self.mode = mode
        self.dimension = dimension

        self.in_channels = in_channels
        self.inter_channels = inter_channels

        if self.inter_channels is None:
            self.inter_channels = in_channels // 2

        if dimension == 3:
            conv_nd = nn.Conv3d
            max_pool_layer = nn.MaxPool3d(kernel_size=(1, 2, 2))
            bn = nn.BatchNorm3d

        elif dimension == 2:
            conv_nd = nn.Conv2d
            max_pool_layer = nn.MaxPool2d(kernel_size=(2, 2))
            bn = nn.BatchNorm2d

        else:
            conv_nd = nn.Conv1d
            max_pool_layer = nn.MaxPool1d(kernel_size=(2))
            bn = nn.BatchNorm1d

        self.g = conv_nd(in_channels=self.in_channels, out_channels=self.inter_channels, kernel_size=1)

        if bn_layer:
            self.W_z = nn.Sequential(
                conv_nd(in_channels=self.inter_channels, out_channels=self.in_channels, kernel_size=1),
                bn(self.in_channels)
            )
            nn.init.constant_(self.W_z[1].weight, 0)
            nn.init.constant_(self.W_z[1].bias, 0)
        else:
            self.W_z = conv_nd(in_channels=self.inter_channels, out_channels=self.in_channels, kernel_size=1)

            nn.init.constant_(self.W_z.weight, 0)
            nn.init.constant_(self.W_z.bias, 0)

        if self.mode == "embedded" or self.mode == "dot" or self.mode == "concatenate":
            self.theta = conv_nd(in_channels=self.in_channels, out_channels=self.inter_channels, kernel_size=1)
            self.phi = conv_nd(in_channels=self.in_channels, out_channels=self.inter_channels, kernel_size=1)

        if self.mode == "concatenate":
            self.W_f = nn.Sequential(
                nn.Conv2d(in_channels=self.inter_channels * 2, out_channels=1, kernel_size=1),
                nn.ReLU()
            )
        self.softmax1 = nn.Softmax(dim=1)  # the sum of each tensor's row equal to 1 [ADD]

    def forward(self, x_low, x_high):
        """
        args
            x: (N, C, T, H, W) for dimension=3; (N, C, H, W) for dimension 2; (N, C, T) for dimension 1
        """
        batch_size = x_low.size(0)
        """
        g_x = self.g(x_high).view(batch_size, self.inter_channels, -1)  # V
        g_x = g_x.permute(0, 2, 1)
        """
        g_x = self.g(x_low).view(batch_size, self.inter_channels, -1)  # Q
        g_x = g_x.permute(0, 2, 1)


        if self.mode == "gaussian":
            theta_x = x_low.view(batch_size, self.in_channels, -1)  # Q
            phi_x = x_high.view(batch_size, self.in_channels, -1)  # K
            theta_x = theta_x.permute(0, 2, 1)
            f = torch.matmul(theta_x, phi_x)  # W=Q.K /

        elif self.mode == "embedded" or self.mode == "dot":
            # theta_x = self.theta(x_low).view(batch_size, self.inter_channels, -1)  # Q
            phi_x = self.phi(x_high).view(batch_size, self.inter_channels, -1)  # K
            phi_x = self.softmax1(phi_x)
            phi_x = phi_x.permute(0, 2, 1)
            theta_x = self.theta(x_high).view(batch_size, self.inter_channels, -1)  # V
            # theta_x = theta_x.permute(0, 2, 1)
            f = torch.matmul(theta_x, phi_x)  # W=Q.K

        elif self.mode == "concatenate":
            theta_x = self.theta(x_low).view(batch_size, self.inter_channels, -1, 1)
            phi_x = self.phi(x_high).view(batch_size, self.inter_channels, 1, -1)

            h = theta_x.size(2)
            w = phi_x.size(3)
            theta_x = theta_x.repeat(1, 1, 1, w)
            phi_x = phi_x.repeat(1, 1, h, 1)
            concat = torch.cat([theta_x, phi_x], dim=1)
            f = self.W_f(concat)
            f = f.view(f.size(0), f.size(2), f.size(3))

        if self.mode == "gaussian" or self.mode == "embedded":
            f_div_C = F.softmax(f, dim=-1)
        elif self.mode == "dot" or self.mode == "concatenate":
            N = f.size(-1)  # number of position in x
            f_div_C = f / N
        y = torch.matmul(g_x, f_div_C)  # W.V/ W.Q

        y = y.permute(0, 2, 1).contiguous()
        h, w = x_low.size()[2:]
        y = y.view(batch_size, self.inter_channels, h, w)

        z = self.W_z(y)
        return z


class RenderTrans(nn.Module):
    def __init__(self, channels_high, channels_low, upsample=True, layer_stride=1):
        super(RenderTrans, self).__init__()
        self.upsample = upsample

        self.conv3x3 = nn.Conv2d(channels_high, channels_high, kernel_size=3, padding=1, bias=False)
        self.bn_low = nn.BatchNorm2d(channels_high)

        self.conv1x1 = nn.Conv2d(channels_low, channels_high, kernel_size=1, padding=0, bias=False)
        self.bn_high = nn.BatchNorm2d(channels_high)

        if upsample:
            self.conv_upsample = nn.ConvTranspose2d(channels_low, channels_high, kernel_size=4, stride=2, padding=1, bias=False)
            self.bn_upsample = nn.BatchNorm2d(channels_high)
        else:
            self.conv_reduction = nn.Conv2d(channels_low, channels_high, kernel_size=1, padding=0, bias=False)
            self.bn_reduction = nn.BatchNorm2d(channels_high)

        if layer_stride==1:
            self.str_conv3x3_s2 = nn.Conv2d(channels_low, channels_high, kernel_size=3, stride=2, padding=1, bias=False)  # stride should be change
        if layer_stride==2:
            self.str_conv3x3_s4 = nn.Sequential(nn.Conv2d(channels_low, channels_high, kernel_size=3, stride=2, padding=1, bias=False),
                                                nn.Conv2d(channels_low, channels_high, kernel_size=3, stride=2, padding=1, bias=False))
        if layer_stride==3:
            self.str_conv3x3_s8 = nn.Sequential(nn.Conv2d(channels_low, channels_high, kernel_size=3, stride=2, padding=1, bias=False),
                                                nn.Conv2d(channels_low, channels_high, kernel_size=3, stride=2, padding=1, bias=False),
                                                nn.Conv2d(channels_low, channels_high, kernel_size=3, stride=2, padding=1, bias=False))

        self.relu = nn.ReLU(inplace=True)
        self.conv_cat = nn.Conv2d(channels_high*2, channels_high, kernel_size=1, padding=0, bias=False)
        self.coorattention = cross_scale_CoordAtt(channels_low, channels_low)

    def forward(self, x_high, x_low):
        b, c, h, w = x_low.shape
        x_low_gp = nn.AvgPool2d(x_low.shape[2:])(x_low).view(len(x_low), c, 1, 1)
        x_low_gp = self.conv1x1(x_low_gp)
        x_low_gp = self.bn_low(x_low_gp)
        x_low_gp = self.relu(x_low_gp)

        x_high_mask = self.conv3x3(x_high)
        x_high_mask = self.bn_high(x_high_mask)

        x_att = x_high_mask * x_low_gp
        # x_att = self.coorattention(x_low, x_high_mask)
        b1, c1, h1, w1 = x_high.shape
        s = h // h1
        if self.upsample:
            if s==2:
                out = self.relu(self.bn_upsample(self.str_conv3x3_s2(x_low)) + x_att)
                # self.conv_cat(torch.cat([self.bn_upsample(self.str_conv3x3(x_low)), x_att], dim=1))
            elif s==4:
                out = self.relu(self.bn_upsample(self.str_conv3x3_s4(x_low)) + x_att)
            else:
                out = self.relu(self.bn_upsample(self.str_conv3x3_s8(x_low)) + x_att)
        else:
            if s==2:
                out = self.relu(self.bn_reduction(self.str_conv3x3_s2(x_low)) + x_att)
            elif s==4:
                out = self.relu(self.bn_reduction(self.str_conv3x3_s4(x_low)) + x_att)
            else:
                out = self.relu(self.bn_reduction(self.str_conv3x3_s8(x_low)) + x_att)
                # self.conv_cat(torch.cat([self.bn_reduction(self.str_conv3x3(x_low)), x_att], dim=1))
        return out


# coordAttention !!!
class h_sigmoid(nn.Module):
    def __init__(self, inplace=True):
        super(h_sigmoid, self).__init__()
        self.relu = nn.ReLU6(inplace=inplace)

    def forward(self, x):
        return self.relu(x + 3) / 6


class h_swish(nn.Module):
    def __init__(self, inplace=True):
        super(h_swish, self).__init__()
        self.sigmoid = h_sigmoid(inplace=inplace)

    def forward(self, x):
        return x * self.sigmoid(x)


class cross_scale_CoordAtt(nn.Module):
    def __init__(self, inp, oup, reduction=32):
        super(cross_scale_CoordAtt, self).__init__()
        self.pool_h = nn.AdaptiveAvgPool2d((None, 1))
        self.pool_w = nn.AdaptiveAvgPool2d((1, None))

        mip = max(8, inp // reduction)

        self.conv1_s2 = nn.Conv2d(inp, mip, kernel_size=3, stride=2, padding=1)  # change k=1,s=1
        self.conv1_s4 = nn.Sequential(nn.Conv2d(inp, mip, kernel_size=3, stride=2, padding=1, bias=False),
                                      nn.Conv2d(mip, mip, kernel_size=3, stride=2, padding=1, bias=False))
        self.conv1_s8 = nn.Sequential(nn.Conv2d(inp, mip, kernel_size=3, stride=2, padding=1, bias=False),
                                      nn.Conv2d(mip, mip, kernel_size=3, stride=2, padding=1, bias=False),
                                      nn.Conv2d(mip, mip, kernel_size=3, stride=2, padding=1, bias=False))
        self.bn1 = nn.BatchNorm2d(mip)
        self.act = h_swish()

        self.conv_h = nn.Conv2d(mip, oup, kernel_size=1, stride=1, padding=0)
        self.conv_w = nn.Conv2d(mip, oup, kernel_size=1, stride=1, padding=0)

    def forward(self, x_low, x_high):
        identity = x_high
        n, c, h, w = x_low.size()
        n1, c1, h1, w1 = x_high.size()
        stride = h // h1

        x_h = self.pool_h(x_low)
        x_w = self.pool_w(x_low).permute(0, 1, 3, 2)

        y = torch.cat([x_h, x_w], dim=2)
        if stride == 2:
            y = self.conv1_s2(y)
        elif stride == 4:
            y = self.conv1_s4(y)
        else:
            y = self.conv1_s8(y)
        y = self.bn1(y)
        y = self.act(y)

        x_h, x_w = torch.split(y, [h1, w1], dim=2)
        x_w = x_w.permute(0, 1, 3, 2)

        a_h = self.conv_h(x_h).sigmoid()
        a_w = self.conv_w(x_w).sigmoid()

        out = identity * a_w * a_h

        return out