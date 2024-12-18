from functools import partial

import torch
import torch.nn as nn
import torch.utils.checkpoint as cp
from timm.models.layers import trunc_normal_ as __call_trunc_normal_
from timm.models.registry import register_model

from .modeling_finetune import (
    Block,
    PatchEmbed,
    get_sinusoid_encoding_table,
)


def trunc_normal_(tensor, mean=0., std=1.):
    __call_trunc_normal_(tensor, mean=mean, std=std, a=-std, b=std)


class PixelPretrainVisionTransformerEncoder(nn.Module):
    """ Vision Transformer with support for patch or hybrid CNN input stage
    """

    def __init__(self,
                 img_size=224,
                 patch_size=16,
                 in_chans=3,
                 num_classes=0,
                 embed_dim=768,
                 depth=12,
                 num_heads=12,
                 mlp_ratio=4.,
                 qkv_bias=False,
                 qk_scale=None,
                 drop_rate=0.,
                 attn_drop_rate=0.,
                 drop_path_rate=0.,
                 norm_layer=nn.LayerNorm,
                 init_values=None,
                 tubelet_size=2,
                 use_learnable_pos_emb=False,
                 with_cp=False,
                 all_frames=16,
                 cos_attn=False):
        super().__init__()
        self.num_classes = num_classes
        # num_features for consistency with other models
        self.num_features = self.embed_dim = embed_dim
        self.patch_embed = PatchEmbed(
            img_size=img_size,
            patch_size=patch_size,
            in_chans=in_chans,
            embed_dim=embed_dim,
            num_frames=all_frames,
            tubelet_size=tubelet_size)
        num_patches = self.patch_embed.num_patches
        self.with_cp = with_cp

        if use_learnable_pos_emb:
            self.pos_embed = nn.Parameter(torch.zeros(1, num_patches, embed_dim))
        else:
            pos_embed = get_sinusoid_encoding_table(
                n_position=num_patches, 
                d_hid=embed_dim)
            self.register_buffer("pos_embed", pos_embed)
            
        # stochastic depth decay rule
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)] 
        self.blocks = nn.ModuleList([
            Block(
                dim=embed_dim,
                num_heads=num_heads,
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                qk_scale=qk_scale,
                drop=drop_rate,
                attn_drop=attn_drop_rate,
                drop_path=dpr[i],
                norm_layer=norm_layer,
                init_values=init_values,
                cos_attn=cos_attn) for i in range(depth)
        ])

        self.norm = norm_layer(embed_dim)
        self.head = nn.Linear(
            embed_dim, num_classes) if num_classes > 0 else nn.Identity()
        
        if use_learnable_pos_emb:
            trunc_normal_(self.pos_embed, std=.02)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def get_num_layers(self):
        return len(self.blocks)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'pos_embed', 'cls_token'}

    def random_masking(self, x, mask_ratio,
                    batch, embed_dim, sequence_frames, sequence_height, sequence_width):
        """
        Perform shared random masking for multiple frames in each sample.
        All frames in a sample share the same mask pattern.
        """
        N = batch
        L = sequence_height * sequence_width
        D = embed_dim
        len_keep = int(L * (1 - mask_ratio))

        # Generate a single noise vector for one sample, shared across all frames
        noise = torch.rand(N, 1, L, device=x.device)  # noise in [0, 1]

        # Sort noise to create a shared mask pattern for all frames in a sample
        ids_shuffle = torch.argsort(noise, dim=2)  # Ascend: small values are kept, large are removed
        ids_restore = torch.argsort(ids_shuffle, dim=2).repeat(1, sequence_frames, 1)

        # Keep the first subset, extending the mask pattern to all frames in each sample
        ids_keep = ids_shuffle[:, :, :len_keep]
        ids_keep = ids_keep.repeat(1, sequence_frames, 1)  # Repeat the same indices for all frames in each sample
        x = x.view(N, sequence_frames, L, D)  # Reshape x to separate frames
        x_masked = torch.gather(x, dim=2, index=ids_keep.unsqueeze(-1).repeat(1, 1, 1, D))
        x_masked = x_masked.view(N, sequence_frames * len_keep, D)  # Reshape back to original dimensions

        # Generate the binary mask: 0 is keep, 1 is remove, shared for all frames in each sample
        mask = torch.ones([N, 1, L], device=x.device)
        mask[:, :, :len_keep] = 0
        mask = mask.repeat(1, sequence_frames, 1)  # Repeat the same mask for all frames in each sample
        # Unshuffle to get the binary mask according to the original order
        mask = torch.gather(mask, dim=2, index=ids_restore)
        mask = mask.view(N, sequence_frames * L).to(torch.bool)

        return x_masked, mask, ids_restore


    def forward(self, x, mask_ratio):
        x = self.patch_embed(x)
        N, D, T, L_h, L_w = x.shape
        
        # N,D,L -> N,L,D
        x = x.flatten(2).transpose(1, 2)
        x = x + self.pos_embed
        
        # N,L,D -> N,(L*mask_ratio),D 
        x_vis, mask, ids_restore = self.random_masking(x, mask_ratio, 
                                                   N, D, T, L_h, L_w)

        for blk in self.blocks:
            if self.with_cp:
                x_vis = cp.checkpoint(blk, x_vis)
            else:
                x_vis = blk(x_vis)

        x_vis = self.norm(x_vis)
        x_vis = self.head(x_vis)
        return x_vis, mask, ids_restore


class PixelPretrainVisionTransformerDecoder(nn.Module):
    """ Vision Transformer with support for patch or hybrid CNN input stage
    """

    def __init__(self,
                 patch_size=16,
                 num_classes=768,
                 embed_dim=768,
                 depth=12,
                 num_heads=12,
                 mlp_ratio=4.,
                 qkv_bias=False,
                 qk_scale=None,
                 drop_rate=0.,
                 attn_drop_rate=0.,
                 drop_path_rate=0.,
                 norm_layer=nn.LayerNorm,
                 init_values=None,
                 num_patches=196,
                 tubelet_size=2,
                 with_cp=False,
                 all_frames=16,
                 vit_down_frames=None,
                 vit_down_h=None,
                 vit_down_w=None,
                 cos_attn=False):
        super().__init__()

        self.num_classes = num_classes
        self.num_patches = num_patches
        self.all_frames = all_frames
        self.tubelet_size = tubelet_size
        self.vit_down_frames = vit_down_frames
        self.vit_down_h = vit_down_h
        self.vit_down_w = vit_down_w
        self.patch_size = patch_size
        self.with_cp = with_cp

        # num_features for consistency with other models
        self.num_features = self.embed_dim = embed_dim

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)
               ]  # stochastic depth decay rule
        self.blocks = nn.ModuleList([
            Block(
                dim=embed_dim,
                num_heads=num_heads,
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                qk_scale=qk_scale,
                drop=drop_rate,
                attn_drop=attn_drop_rate,
                drop_path=dpr[i],
                norm_layer=norm_layer,
                init_values=init_values,
                cos_attn=cos_attn) for i in range(depth)
        ])
        self.norm = norm_layer(embed_dim)
        self.head = nn.Linear(
            embed_dim, num_classes) if num_classes > 0 else nn.Identity()
        
        self.apply(self._init_weights)

        self.mask_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        trunc_normal_(self.mask_token, std=.02)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def get_num_layers(self):
        return len(self.blocks)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'pos_embed', 'cls_token', 'mask_token'}

    def forward(self, x, 
                pos_emd_vis,
                pos_emd_mask,
                return_token_num):

        x = torch.cat(
            [x + pos_emd_vis, self.mask_token + pos_emd_mask], dim=1)

        for blk in self.blocks:
            if self.with_cp:
                x = cp.checkpoint(blk, x)
            else:
                x = blk(x)
        if return_token_num > 0:
            # only return the mask tokens predict pixels
            x = self.head(self.norm(x[:, -return_token_num:]))
        else:
            # [B, N, 3*16^2]
            x = self.head(self.norm(x))
        return x


class PixelPretrainVisionTransformer(nn.Module):
    """ Vision Transformer with support for patch or hybrid CNN input stage
    """

    def __init__(
        self,
        img_size=224,
        patch_size=16,
        encoder_in_chans=3,
        encoder_num_classes=0,
        encoder_embed_dim=768,
        encoder_depth=12,
        encoder_num_heads=12,
        decoder_num_classes=1536,
        decoder_embed_dim=512,
        decoder_depth=8,
        decoder_num_heads=8,
        mlp_ratio=4.,
        qkv_bias=False,
        qk_scale=None,
        drop_rate=0.,
        attn_drop_rate=0.,
        drop_path_rate=0.,
        norm_layer=nn.LayerNorm,
        init_values=0.,
        use_learnable_pos_emb=False,
        tubelet_size=2,
        num_classes=0,  # avoid the error from create_fn in timm
        in_chans=0,  # avoid the error from create_fn in timm
        with_cp=False,
        all_frames=16,
        cos_attn=False
    ):
        super().__init__()

        self.tubelet_size = tubelet_size
        self.patch_size = patch_size
        self.encoder_in_chans = encoder_in_chans

        self.vit_down_frames = int(all_frames//tubelet_size)
        self.vit_down_h = int(img_size//patch_size)
        self.vit_down_w = int(img_size//patch_size)

        assert(decoder_num_classes == self.encoder_in_chans * self.patch_size * self.patch_size * self.tubelet_size)

        self.encoder = PixelPretrainVisionTransformerEncoder(
            img_size=img_size,
            patch_size=patch_size,
            in_chans=encoder_in_chans,
            num_classes=encoder_num_classes,
            embed_dim=encoder_embed_dim,
            depth=encoder_depth,
            num_heads=encoder_num_heads,
            mlp_ratio=mlp_ratio,
            qkv_bias=qkv_bias,
            qk_scale=qk_scale,
            drop_rate=drop_rate,
            attn_drop_rate=attn_drop_rate,
            drop_path_rate=drop_path_rate,
            norm_layer=norm_layer,
            init_values=init_values,
            tubelet_size=tubelet_size,
            use_learnable_pos_emb=use_learnable_pos_emb,
            with_cp=with_cp,
            all_frames=all_frames,
            cos_attn=cos_attn)

        self.decoder = PixelPretrainVisionTransformerDecoder(
                    patch_size=patch_size,
                    num_patches=self.encoder.patch_embed.num_patches,
                    num_classes=decoder_num_classes,
                    embed_dim=decoder_embed_dim,
                    depth=decoder_depth,
                    num_heads=decoder_num_heads,
                    mlp_ratio=mlp_ratio,
                    qkv_bias=qkv_bias,
                    qk_scale=qk_scale,
                    drop_rate=drop_rate,
                    attn_drop_rate=attn_drop_rate,
                    drop_path_rate=drop_path_rate,
                    norm_layer=norm_layer,
                    init_values=init_values,
                    tubelet_size=tubelet_size,
                    with_cp=with_cp,
                    all_frames=all_frames,
                    vit_down_frames = self.vit_down_frames,
                    vit_down_h = self.vit_down_h,
                    vit_down_w = self.vit_down_w,
                    cos_attn=cos_attn)
        
        self.encoder_to_decoder = nn.Linear(encoder_embed_dim, decoder_embed_dim, bias=False)

        pos_embed = get_sinusoid_encoding_table(
            n_position=self.encoder.patch_embed.num_patches, 
            d_hid=decoder_embed_dim)
        self.register_buffer("pos_embed", pos_embed)


    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def get_num_layers(self):
        return len(self.blocks)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'pos_embed', 'cls_token'}

    @torch.no_grad()
    def target_pixel(self, vid):
        batch, _, _, _, _ = vid.shape
        # B,C,T,H,W -- > B, C, vit_down_frames, tubelet_size, vit_down_h, patch_size, vit_down_w, patch_size 
        vid = vid.view(batch, 
                    self.encoder_in_chans, 
                    self.vit_down_frames, self.tubelet_size, 
                    self.vit_down_h, self.patch_size, 
                    self.vit_down_w, self.patch_size)
        # B, C, T//tubelet_size, tubelet_size, 14, 16, 14, 16
        # --->
        # B, T//tubelet_size, 14, 14, tubelet_size, 16, 16, 3
        vid = vid.permute(0, 2, 4, 6, 3, 5, 7, 1).contiguous()
        vid = vid.view(batch, 
                        self.vit_down_frames * self.vit_down_h * self.vit_down_w,
                        self.tubelet_size * self.patch_size * self.patch_size,
                        self.encoder_in_chans)
        # videomae -> unbiased=True
        vid = (vid - vid.mean(dim=-2, keepdim=True)) / (vid.var(dim=-2, unbiased=True, keepdim=True).sqrt() + 1e-6)
        vid = vid.view(batch, 
                        self.vit_down_frames * self.vit_down_h * self.vit_down_w,
                        self.tubelet_size * self.patch_size * self.patch_size * self.encoder_in_chans)
        return vid

    def forward(self, x, mask_ratio):
                
        x_vis, mask, ids_restore = self.encoder(x, mask_ratio)
        x_vis = self.encoder_to_decoder(x_vis)
        
        N, L_vis, D = x_vis.shape
        expand_pos_embed = self.pos_embed.expand(N, -1, -1)
        pos_emd_vis = expand_pos_embed[~mask].reshape(N, -1, D)
        pos_emd_mask = expand_pos_embed[mask].reshape(N, -1, D)
        _, L_mask, _ = pos_emd_mask.shape

        pred = self.decoder(x_vis, pos_emd_vis, pos_emd_mask, L_mask)
        _, _, target_D = pred.shape
        target = self.target_pixel(x)[mask].reshape(N, -1, target_D)
        
        return pred, target, mask


@register_model
def pixel_pretrain_videomae_small_patch16_224(pretrained=False, **kwargs):
    model = PixelPretrainVisionTransformer(
        img_size=224,
        patch_size=16,
        encoder_embed_dim=384,
        encoder_depth=12,
        encoder_num_heads=6,
        encoder_num_classes=0,
        decoder_num_classes=1536,  # 16 * 16 * 3 * 2
        decoder_embed_dim=192,
        decoder_num_heads=3,
        mlp_ratio=4,
        qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
        **kwargs)
    if pretrained:
        checkpoint = torch.load(kwargs["init_ckpt"], map_location="cpu")
        model.load_state_dict(checkpoint["model"])
    return model


@register_model
def pixel_pretrain_videomae_base_patch16_224(pretrained=False, **kwargs):
    model = PixelPretrainVisionTransformer(
        img_size=224,
        patch_size=16,
        encoder_embed_dim=768,
        encoder_depth=12,
        encoder_num_heads=12,
        encoder_num_classes=0,
        decoder_num_classes=1536,  # 16 * 16 * 3 * 2
        decoder_embed_dim=384,
        decoder_num_heads=6,
        mlp_ratio=4,
        qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
        **kwargs)
    if pretrained:
        checkpoint = torch.load(kwargs["init_ckpt"], map_location="cpu")
        model.load_state_dict(checkpoint["model"])
    return model


@register_model
def pixel_pretrain_videomae_large_patch16_224(pretrained=False, **kwargs):
    model = PixelPretrainVisionTransformer(
        img_size=224,
        patch_size=16,
        encoder_embed_dim=1024,
        encoder_depth=24,
        encoder_num_heads=16,
        encoder_num_classes=0,
        decoder_num_classes=1536,  # 16 * 16 * 3 * 2
        decoder_embed_dim=512,
        decoder_num_heads=8,
        mlp_ratio=4,
        qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
        **kwargs)
    if pretrained:
        checkpoint = torch.load(kwargs["init_ckpt"], map_location="cpu")
        model.load_state_dict(checkpoint["model"])
    return model


@register_model
def pixel_pretrain_videomae_huge_patch16_224(pretrained=False, **kwargs):
    model = PixelPretrainVisionTransformer(
        img_size=224,
        patch_size=16,
        encoder_embed_dim=1280,
        encoder_depth=32,
        encoder_num_heads=16,
        encoder_num_classes=0,
        decoder_num_classes=1536,  # 16 * 16 * 3 * 2
        decoder_embed_dim=512,
        decoder_num_heads=8,
        mlp_ratio=4,
        qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
        **kwargs)
    if pretrained:
        checkpoint = torch.load(kwargs["init_ckpt"], map_location="cpu")
        model.load_state_dict(checkpoint["model"])
    return model


@register_model
def pixel_pretrain_videomae_giant_patch14_224(pretrained=False, **kwargs):
    model = PixelPretrainVisionTransformer(
        img_size=224,
        patch_size=14,
        encoder_embed_dim=1408,
        encoder_depth=40,
        encoder_num_heads=16,
        encoder_num_classes=0,
        decoder_num_classes=1176,  # 14 * 14 * 3 * 2,
        decoder_embed_dim=512,
        decoder_num_heads=8,
        mlp_ratio=48 / 11,
        qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
        **kwargs)
    if pretrained:
        checkpoint = torch.load(kwargs["init_ckpt"], map_location="cpu")
        model.load_state_dict(checkpoint["model"])
    return model



if __name__ == '__main__':
    import time
    import numpy as np
    import random
    from fvcore.nn import FlopCountAnalysis
    from fvcore.nn import flop_count_table

    seed = 10086
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    batch_size = 1
    num_frames = 16
    tubelet_size = 2
    img_size = 224
    mask_ratio = 0.9
    use_bf16 = True
    if use_bf16:
        datatype = torch.bfloat16
    else:
        datatype = torch.float16
        
    model = pixel_pretrain_videomae_base_patch16_224(pretrained=False,
                                                    tubelet_size=tubelet_size).cuda()
    print(model)
    
    videos = torch.rand(batch_size, 3, num_frames, img_size, img_size).cuda()
    print(videos.shape)
    
    if use_bf16:
        videos = videos.bfloat16()
        model = model.bfloat16()
        with torch.cuda.amp.autocast(dtype=datatype):
            pred, target, mask = model(videos, mask_ratio)
            B, _, _ = pred.shape
            cal_loss_mask = mask[mask].reshape(B, -1)
            # mae-loss(nn.MSELoss())
            loss = ((((pred - target)**2).mean(dim=-1)) * cal_loss_mask).sum() / cal_loss_mask.sum()
            print(loss)
    else:
        videos = videos.half()
        model = model.half()
        with torch.cuda.amp.autocast(dtype=datatype):
            pred, target, mask = model(videos, mask_ratio)
            B, _, _ = pred.shape
            cal_loss_mask = mask[mask].reshape(B, -1)
            # mae-loss(nn.MSELoss())
            loss = ((((pred - target)**2).mean(dim=-1)) * cal_loss_mask).sum() / cal_loss_mask.sum()
            print(loss)

    flops = FlopCountAnalysis(model, (videos, mask_ratio))
    print(flop_count_table(flops, max_depth=1))
    # batch_size=1 vit-b16 #flops 53.433G

    from timm.models import create_model
    resnet_model = create_model('resnet50', pretrained=False).cuda()
    images = torch.rand(batch_size, 3, img_size, img_size).cuda()
    resnet_flops = FlopCountAnalysis(resnet_model, images)
    print(flop_count_table(resnet_flops, max_depth=1))
    # batch_size=1 resnet50 #flops 4.145G