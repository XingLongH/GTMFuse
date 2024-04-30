import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy  as np
#from fairscale.nn import checkpoint_wrapper
from omegaconf import OmegaConf
from upsample import Upsample, UpsampleOneStep
from mixed_attn_block_efficient import (
    _get_stripe_info,
    EfficientMixAttnTransformerBlock,
)
from ops import (
    bchw_to_blc,
    blc_to_bchw,
    calculate_mask,
    calculate_mask_all,
    get_relative_coords_table_all,
    get_relative_position_index_simple,
)
from swin_v1_block import (
    build_last_conv,
)
from timm.models.layers import to_2tuple, trunc_normal_


class TransformerStage(nn.Module):
    def __init__(
        self,
        dim,
        input_resolution,
        depth,
        num_heads_window,
        num_heads_stripe,
        window_size,
        stripe_size,
        stripe_groups,
        stripe_shift,
        mlp_ratio=4.0,
        qkv_bias=True,
        qkv_proj_type="linear",
        anchor_proj_type="avgpool",
        anchor_one_stage=True,
        anchor_window_down_factor=1,
        drop=0.0,
        attn_drop=0.0,
        drop_path=0.0,
        norm_layer=nn.LayerNorm,
        pretrained_window_size=[0, 0],
        pretrained_stripe_size=[0, 0],
        conv_type="1conv",
        init_method="",
        fairscale_checkpoint=False,
        offload_to_cpu=False,
        args=None,
    ):
        super().__init__()

        self.dim = dim
        self.input_resolution = input_resolution
        self.init_method = init_method

        self.blocks = nn.ModuleList()
        for i in range(depth):
            block = EfficientMixAttnTransformerBlock(
                dim=dim,
                input_resolution=input_resolution,
                num_heads_w=num_heads_window,
                num_heads_s=num_heads_stripe,
                window_size=window_size,
                window_shift=i % 2 == 0,
                stripe_size=stripe_size,
                stripe_groups=stripe_groups,
                stripe_type="H" if i % 2 == 0 else "W",
                stripe_shift=i % 4 in [2, 3] if stripe_shift else False,
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                qkv_proj_type=qkv_proj_type,
                anchor_proj_type=anchor_proj_type,
                anchor_one_stage=anchor_one_stage,
                anchor_window_down_factor=anchor_window_down_factor,
                drop=drop,
                attn_drop=attn_drop,
                drop_path=drop_path[i] if isinstance(drop_path, list) else drop_path,
                norm_layer=norm_layer,
                pretrained_window_size=pretrained_window_size,
                pretrained_stripe_size=pretrained_stripe_size,
                res_scale=0.1 if init_method == "r" else 1.0,
                args=args,
            )

            self.blocks.append(block)

        self.conv = build_last_conv(conv_type, dim)

    def _init_weights(self):
        for n, m in self.named_modules():
            if self.init_method == "w":
                if isinstance(m, (nn.Linear, nn.Conv2d)) and n.find("cpb_mlp") < 0:
                    print("nn.Linear and nn.Conv2d weight initilization")
                    m.weight.data *= 0.1
            elif self.init_method == "l":
                if isinstance(m, nn.LayerNorm):
                    print("nn.LayerNorm initialization")
                    nn.init.constant_(m.bias, 0)
                    nn.init.constant_(m.weight, 0)
            elif self.init_method.find("t") >= 0:
                scale = 0.1 ** (len(self.init_method) - 1) * int(self.init_method[-1])
                if isinstance(m, nn.Linear) and n.find("cpb_mlp") < 0:
                    trunc_normal_(m.weight, std=scale)
                elif isinstance(m, nn.Conv2d):
                    m.weight.data *= 0.1
                print(
                    "Initialization nn.Linear - trunc_normal; nn.Conv2d - weight rescale."
                )
            else:
                raise NotImplementedError(
                    f"Parameter initialization method {self.init_method} not implemented in TransformerStage."
                )

    def forward(self, x, x_size, table_index_mask):
        res = x
        for blk in self.blocks:
            res = blk(res, x_size, table_index_mask)
        res = bchw_to_blc(self.conv(blc_to_bchw(res, x_size)))

        return res + x

    def flops(self):
        pass


class GTM(nn.Module):
    def __init__(
        self,
        img_size=256,
        in_channels=1,
        out_channels=None,
        embed_dim=96,
        upscale=2,
        img_range=1.0,
        upsampler="none",
        depths=[6, 6, 6, 6],
        num_heads_window=[3, 3, 3, 3],
        num_heads_stripe=[3, 3, 3, 3],
        window_size=8,
        stripe_size=[8, 8],  # used for stripe window attention
        stripe_groups=[None, None],
        stripe_shift=False,
        mlp_ratio=4.0,
        qkv_bias=True,
        qkv_proj_type="linear",
        anchor_proj_type="avgpool",
        anchor_one_stage=True,
        anchor_window_down_factor=1,
        out_proj_type="linear",
        local_connection=False,
        drop_rate=0.0,
        attn_drop_rate=0.0,
        drop_path_rate=0.1,
        norm_layer=nn.LayerNorm,
        pretrained_window_size=[0, 0],
        pretrained_stripe_size=[0, 0],
        conv_type="1conv",
        init_method="n",  # initialization method of the weight parameters used to train large scale models.
        fairscale_checkpoint=False,  # fairscale activation checkpointing
        offload_to_cpu=False,
        euclidean_dist=False,
        **kwargs,
    ):
        super(GTM, self).__init__()
        # Process the input arguments
        self.Maxpool = nn.MaxPool2d(kernel_size = 2, stride = 2)
        out_channels = out_channels or in_channels
        self.in_channels = in_channels
        self.out_channels = out_channels
        num_out_feats = 64
        self.embed_dim = embed_dim
        self.upscale = upscale
        self.upsampler = upsampler
        self.img_range = img_range
        if in_channels == 3:
            rgb_mean = (0.4488, 0.4371, 0.4040)
            self.mean = torch.Tensor(rgb_mean).view(1, 3, 1, 1)
        else:
            self.mean = torch.zeros(1, 1, 1, 1)

        max_stripe_size = max([0 if s is None else s for s in stripe_size])
        max_stripe_groups = max([0 if s is None else s for s in stripe_groups])
        max_stripe_groups *= anchor_window_down_factor
        self.pad_size = max(window_size, max_stripe_size, max_stripe_groups)

        self.input_resolution = to_2tuple(img_size)
        self.window_size = to_2tuple(window_size)
        self.shift_size = [w // 2 for w in self.window_size]
        self.stripe_size = stripe_size
        self.stripe_groups = stripe_groups
        self.pretrained_window_size = pretrained_window_size
        self.pretrained_stripe_size = pretrained_stripe_size
        self.anchor_window_down_factor = anchor_window_down_factor

        self.conv_first = nn.Conv2d(in_channels, embed_dim, 3, 1, 1)

        # Body of the network
        self.norm_start = norm_layer(embed_dim)
        self.pos_drop = nn.Dropout(p=drop_rate)

        # stochastic depth
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]
        # stochastic depth decay rule
        args = OmegaConf.create(
            {
                "out_proj_type": out_proj_type,
                "local_connection": local_connection,
                "euclidean_dist": euclidean_dist,
            }
        )
        for k, v in self.set_table_index_mask(self.input_resolution).items():
            self.register_buffer(k, v)

        self.layers = nn.ModuleList()
        for i in range(len(depths)):
            layer = TransformerStage(
                dim=embed_dim,
                input_resolution=self.input_resolution,
                depth=depths[i],
                num_heads_window=num_heads_window[i],
                num_heads_stripe=num_heads_stripe[i],
                window_size=self.window_size,
                stripe_size=stripe_size,
                stripe_groups=stripe_groups,
                stripe_shift=stripe_shift,
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                qkv_proj_type=qkv_proj_type,
                anchor_proj_type=anchor_proj_type,
                anchor_one_stage=anchor_one_stage,
                anchor_window_down_factor=anchor_window_down_factor,
                drop=drop_rate,
                attn_drop=attn_drop_rate,
                drop_path=dpr[
                    sum(depths[:i]) : sum(depths[: i + 1])
                ],  # no impact on SR results
                norm_layer=norm_layer,
                pretrained_window_size=pretrained_window_size,
                pretrained_stripe_size=pretrained_stripe_size,
                conv_type=conv_type,
                init_method=init_method,
                fairscale_checkpoint=fairscale_checkpoint,
                offload_to_cpu=offload_to_cpu,
                args=args,
            )
            self.layers.append(layer)
        self.norm_end = norm_layer(embed_dim)

        # Tail of the network
        self.conv_after_body = build_last_conv(conv_type, embed_dim)
        self.conv_last = nn.Conv2d(embed_dim, out_channels, 3, 1, 1)

        self.apply(self._init_weights)
        if init_method in ["l", "w"] or init_method.find("t") >= 0:
            for layer in self.layers:
                layer._init_weights()

    def set_table_index_mask(self, x_size):

        ss, sss = _get_stripe_info(self.stripe_size, self.stripe_groups, True, x_size)
        df = self.anchor_window_down_factor

        table_w = get_relative_coords_table_all(
            self.window_size, self.pretrained_window_size
        )
        table_sh = get_relative_coords_table_all(ss, self.pretrained_stripe_size, df)
        table_sv = get_relative_coords_table_all(
            ss[::-1], self.pretrained_stripe_size, df
        )

        index_w = get_relative_position_index_simple(self.window_size)
        index_sh_a2w = get_relative_position_index_simple(ss, df, False)
        index_sh_w2a = get_relative_position_index_simple(ss, df, True)
        index_sv_a2w = get_relative_position_index_simple(ss[::-1], df, False)
        index_sv_w2a = get_relative_position_index_simple(ss[::-1], df, True)

        mask_w = calculate_mask(x_size, self.window_size, self.shift_size)
        mask_sh_a2w = calculate_mask_all(x_size, ss, sss, df, False)
        mask_sh_w2a = calculate_mask_all(x_size, ss, sss, df, True)
        mask_sv_a2w = calculate_mask_all(x_size, ss[::-1], sss[::-1], df, False)
        mask_sv_w2a = calculate_mask_all(x_size, ss[::-1], sss[::-1], df, True)
        return {
            "table_w": table_w,
            "table_sh": table_sh,
            "table_sv": table_sv,
            "index_w": index_w,
            "index_sh_a2w": index_sh_a2w,
            "index_sh_w2a": index_sh_w2a,
            "index_sv_a2w": index_sv_a2w,
            "index_sv_w2a": index_sv_w2a,
            "mask_w": mask_w,
            "mask_sh_a2w": mask_sh_a2w,
            "mask_sh_w2a": mask_sh_w2a,
            "mask_sv_a2w": mask_sv_a2w,
            "mask_sv_w2a": mask_sv_w2a,
        }

    def get_table_index_mask(self, device=None, input_resolution=None):
        # Used during forward pass
        if input_resolution == self.input_resolution:
            return {
                "table_w": self.table_w,
                "table_sh": self.table_sh,
                "table_sv": self.table_sv,
                "index_w": self.index_w,
                "index_sh_a2w": self.index_sh_a2w,
                "index_sh_w2a": self.index_sh_w2a,
                "index_sv_a2w": self.index_sv_a2w,
                "index_sv_w2a": self.index_sv_w2a,
                "mask_w": self.mask_w,
                "mask_sh_a2w": self.mask_sh_a2w,
                "mask_sh_w2a": self.mask_sh_w2a,
                "mask_sv_a2w": self.mask_sv_a2w,
                "mask_sv_w2a": self.mask_sv_w2a,
            }
        else:
            table_index_mask = self.set_table_index_mask(input_resolution)
            for k, v in table_index_mask.items():
                table_index_mask[k] = v.to(device)
            return table_index_mask

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=0.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {"absolute_pos_embed"}

    @torch.jit.ignore
    def no_weight_decay_keywords(self):
        return {"relative_position_bias_table"}

    def check_image_size(self, x):
        _, _, h, w = x.size()
        mod_pad_h = (self.pad_size - h % self.pad_size) % self.pad_size
        mod_pad_w = (self.pad_size - w % self.pad_size) % self.pad_size

        try:
            x = F.pad(x, (0, mod_pad_w, 0, mod_pad_h), "reflect")
        except BaseException:
            x = F.pad(x, (0, mod_pad_w, 0, mod_pad_h), "constant")
        return x

    def forward(self, x):
        x_list = []
        H, W = x.shape[2:]
        x = self.check_image_size(x) 

        x_first = self.conv_first(x)   
        x_size = (x_first.shape[2], x_first.shape[3])
        x = bchw_to_blc(x_first)    
        x = self.norm_start(x)
        x = self.pos_drop(x)


        table_index_mask = self.get_table_index_mask(x.device, x_size)

        for layer in self.layers:
           
            x = layer(x, x_size, table_index_mask)

            x = self.norm_end(x) 
            x = blc_to_bchw(x, x_size) 
           
           
            x_list.append(x)
            x = self.Maxpool(x)
            x_size = (x.shape[2], x.shape[3])
            x = bchw_to_blc(x)   
            x = self.norm_start(x)
            x = self.pos_drop(x)

            table_index_mask = self.get_table_index_mask(x.device, x_size)

           
        return x_list

    def flops(self):
        pass

    def convert_checkpoint(self, state_dict):
        for k in list(state_dict.keys()):
            if (
                k.find("relative_coords_table") >= 0
                or k.find("relative_position_index") >= 0
                or k.find("attn_mask") >= 0
                or k.find("model.table_") >= 0
                or k.find("model.index_") >= 0
                or k.find("model.mask_") >= 0
                
            ):
                state_dict.pop(k)
                print(k)
        return state_dict


def model(**kwargs):
    model = GTM(
    upscale=4,
    img_size=256,
    in_channels=1,
    out_channels=32,
    window_size=8,
    depths=[4, 4, 8, 8],
    embed_dim=96,
    num_heads_window=[3, 3, 3, 3],
    num_heads_stripe=[3, 3, 3, 3],
    mlp_ratio=2,
    qkv_proj_type="linear",
    anchor_proj_type="avgpool",
    anchor_window_down_factor=2,
    out_proj_type="linear",
    conv_type="1conv",
    upsampler="pixe",
    local_connection=True,
    **kwargs)

    return model
