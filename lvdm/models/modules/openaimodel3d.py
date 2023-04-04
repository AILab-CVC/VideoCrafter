from abc import abstractmethod
import math
from einops import rearrange
from functools import partial
import numpy as np
import torch as th
import torch.nn as nn
import torch.nn.functional as F
from omegaconf.listconfig import ListConfig

from lvdm.models.modules.util import (
    checkpoint,
    conv_nd,
    linear,
    avg_pool_nd,
    zero_module,
    normalization,
    timestep_embedding,
    nonlinearity,
)

# dummy replace
def convert_module_to_f16(x):
    pass

def convert_module_to_f32(x):
    pass

## go
# ---------------------------------------------------------------------------------------------------
class TimestepBlock(nn.Module):
    """
    Any module where forward() takes timestep embeddings as a second argument.
    """

    @abstractmethod
    def forward(self, x, emb):
        """
        Apply the module to `x` given `emb` timestep embeddings.
        """


# ---------------------------------------------------------------------------------------------------
class TimestepEmbedSequential(nn.Sequential, TimestepBlock):
    """
    A sequential module that passes timestep embeddings to the children that
    support it as an extra input.
    """

    def forward(self, x, emb, context, **kwargs):
        for layer in self:
            if isinstance(layer, TimestepBlock):
                x = layer(x, emb, **kwargs)
            elif isinstance(layer, STTransformerClass):
                x = layer(x, context, **kwargs)
            else:
                x = layer(x)
        return x


# ---------------------------------------------------------------------------------------------------
class Upsample(nn.Module):
    """
    An upsampling layer with an optional convolution.
    :param channels: channels in the inputs and outputs.
    :param use_conv: a bool determining if a convolution is applied.
    :param dims: determines if the signal is 1D, 2D, or 3D. If 3D, then
                 upsampling occurs in the inner-two dimensions.
    """

    def __init__(self, channels, use_conv, dims=2, out_channels=None, 
        kernel_size_t=3,
        padding_t=1,
    ):
        super().__init__()
        self.channels = channels
        self.out_channels = out_channels or channels
        self.use_conv = use_conv
        self.dims = dims
        if use_conv:
            self.conv = conv_nd(dims, self.channels, self.out_channels, (kernel_size_t, 3,3), padding=(padding_t, 1,1))

    def forward(self, x):
        assert x.shape[1] == self.channels
        if self.dims == 3:
            x = F.interpolate(
                x, (x.shape[2], x.shape[3] * 2, x.shape[4] * 2), mode="nearest"
            )
        else:
            x = F.interpolate(x, scale_factor=2, mode="nearest")
        if self.use_conv:
            x = self.conv(x)
        return x


# ---------------------------------------------------------------------------------------------------
class TransposedUpsample(nn.Module):
    'Learned 2x upsampling without padding'
    def __init__(self, channels, out_channels=None, ks=5):
        super().__init__()
        self.channels = channels
        self.out_channels = out_channels or channels

        self.up = nn.ConvTranspose2d(self.channels,self.out_channels,kernel_size=ks,stride=2)

    def forward(self,x):
        return self.up(x)


# ---------------------------------------------------------------------------------------------------
class Downsample(nn.Module):
    """
    A downsampling layer with an optional convolution.
    :param channels: channels in the inputs and outputs.
    :param use_conv: a bool determining if a convolution is applied.
    :param dims: determines if the signal is 1D, 2D, or 3D. If 3D, then
                 downsampling occurs in the inner-two dimensions.
    """

    def __init__(self, channels, use_conv, dims=2, out_channels=None,
        kernel_size_t=3,
        padding_t=1,
    ):
        super().__init__()
        self.channels = channels
        self.out_channels = out_channels or channels
        self.use_conv = use_conv
        self.dims = dims
        stride = 2 if dims != 3 else (1, 2, 2)
        if use_conv:
            self.op = conv_nd(
                dims, self.channels, self.out_channels, (kernel_size_t, 3,3), stride=stride, padding=(padding_t, 1,1)
            )
        else:
            assert self.channels == self.out_channels
            self.op = avg_pool_nd(dims, kernel_size=stride, stride=stride)

    def forward(self, x):
        assert x.shape[1] == self.channels
        return self.op(x)


# ---------------------------------------------------------------------------------------------------
class ResBlock(TimestepBlock):
    """
    A residual block that can optionally change the number of channels.
    :param channels: the number of input channels.
    :param emb_channels: the number of timestep embedding channels.
    :param dropout: the rate of dropout.
    :param out_channels: if specified, the number of out channels.
    :param use_conv: if True and out_channels is specified, use a spatial
        convolution instead of a smaller 1x1 convolution to change the
        channels in the skip connection.
    :param dims: determines if the signal is 1D, 2D, or 3D.
    :param use_checkpoint: if True, use gradient checkpointing on this module.
    :param up: if True, use this block for upsampling.
    :param down: if True, use this block for downsampling.
    """

    def __init__(
        self,
        channels,
        emb_channels,
        dropout,
        out_channels=None,
        use_conv=False,
        use_scale_shift_norm=False,
        dims=2,
        use_checkpoint=False,
        up=False,
        down=False,
        # temporal
        kernel_size_t=3,
        padding_t=1,
        nonlinearity_type='silu',
        **kwargs
    ):
        super().__init__()
        self.channels = channels
        self.emb_channels = emb_channels
        self.dropout = dropout
        self.out_channels = out_channels or channels
        self.use_conv = use_conv
        self.use_checkpoint = use_checkpoint
        self.use_scale_shift_norm = use_scale_shift_norm
        self.nonlinearity_type = nonlinearity_type

        self.in_layers = nn.Sequential(
            normalization(channels),
            nonlinearity(nonlinearity_type),
            conv_nd(dims, channels, self.out_channels, (kernel_size_t, 3,3), padding=(padding_t, 1,1)),
        )

        self.updown = up or down

        if up:
            self.h_upd = Upsample(channels, False, dims, kernel_size_t=kernel_size_t, padding_t=padding_t)
            self.x_upd = Upsample(channels, False, dims, kernel_size_t=kernel_size_t, padding_t=padding_t)
        elif down:
            self.h_upd = Downsample(channels, False, dims, kernel_size_t=kernel_size_t, padding_t=padding_t)
            self.x_upd = Downsample(channels, False, dims, kernel_size_t=kernel_size_t, padding_t=padding_t)
        else:
            self.h_upd = self.x_upd = nn.Identity()

        self.emb_layers = nn.Sequential(
            nonlinearity(nonlinearity_type),
            linear(
                emb_channels,
                2 * self.out_channels if use_scale_shift_norm else self.out_channels,
            ),
        )
        self.out_layers = nn.Sequential(
            normalization(self.out_channels),
            nonlinearity(nonlinearity_type),
            nn.Dropout(p=dropout),
            zero_module(
                conv_nd(dims, self.out_channels, self.out_channels, (kernel_size_t, 3,3), padding=(padding_t, 1,1))
            ),
        )

        if self.out_channels == channels:
            self.skip_connection = nn.Identity()
        elif use_conv:
            self.skip_connection = conv_nd(
                dims, channels, self.out_channels, (kernel_size_t, 3,3), padding=(padding_t, 1,1)
            )
        else:
            self.skip_connection = conv_nd(dims, channels, self.out_channels, 1)
        

    def forward(self, x, emb, **kwargs):
        """
        Apply the block to a Tensor, conditioned on a timestep embedding.
        :param x: an [N x C x ...] Tensor of features.
        :param emb: an [N x emb_channels] Tensor of timestep embeddings.
        :return: an [N x C x ...] Tensor of outputs.
        """
        return checkpoint(self._forward, 
                          (x, emb), 
                          self.parameters(), 
                          self.use_checkpoint
                          )

    def _forward(self, x, emb,):
        if self.updown:
            in_rest, in_conv = self.in_layers[:-1], self.in_layers[-1]
            h = in_rest(x)
            h = self.h_upd(h)
            x = self.x_upd(x)
            h = in_conv(h)
        else:
            h = self.in_layers(x)

        emb_out = self.emb_layers(emb).type(h.dtype)
        if emb_out.dim() == 3: # btc for video data
            emb_out = rearrange(emb_out, 'b t c -> b c t')
        while len(emb_out.shape) < h.dim():
            emb_out = emb_out[..., None] # bct -> bct11 or bc -> bc111
        
        if self.use_scale_shift_norm:
            out_norm, out_rest = self.out_layers[0], self.out_layers[1:]
            scale, shift = th.chunk(emb_out, 2, dim=1)
            h = out_norm(h) * (1 + scale) + shift
            h = out_rest(h)
        else:
            h = h + emb_out
            h = self.out_layers(h)

        out = self.skip_connection(x) + h
        
        return out

# ---------------------------------------------------------------------------------------------------
def make_spatialtemporal_transformer(module_name='attention_temporal', class_name='SpatialTemporalTransformer'):
    module = __import__(f"lvdm.models.modules.{module_name}", fromlist=[class_name])
    global STTransformerClass
    STTransformerClass = getattr(module, class_name)
    return STTransformerClass

# ---------------------------------------------------------------------------------------------------
class UNetModel(nn.Module):
    """
    The full UNet model with attention and timestep embedding.
    :param in_channels: channels in the input Tensor.
    :param model_channels: base channel count for the model.
    :param out_channels: channels in the output Tensor.
    :param num_res_blocks: number of residual blocks per downsample.
    :param attention_resolutions: a collection of downsample rates at which
        attention will take place. May be a set, list, or tuple.
        For example, if this contains 4, then at 4x downsampling, attention
        will be used.
    :param dropout: the dropout probability.
    :param channel_mult: channel multiplier for each level of the UNet.
    :param conv_resample: if True, use learned convolutions for upsampling and
        downsampling.
    :param dims: determines if the signal is 1D, 2D, or 3D.
    :param num_classes: if specified (as an int), then this model will be
        class-conditional with `num_classes` classes.
    :param use_checkpoint: use gradient checkpointing to reduce memory usage.
    :param num_heads: the number of attention heads in each attention layer.
    :param num_heads_channels: if specified, ignore num_heads and instead use
                               a fixed channel width per attention head.
    :param num_heads_upsample: works with num_heads to set a different number
                               of heads for upsampling. Deprecated.
    :param use_scale_shift_norm: use a FiLM-like conditioning mechanism.
    :param resblock_updown: use residual blocks for up/downsampling.
    :param use_new_attention_order: use a different attention pattern for potentially
                                    increased efficiency.
    """

    def __init__(
        self,
        image_size, # not used in UNetModel
        in_channels,
        model_channels,
        out_channels,
        num_res_blocks,
        attention_resolutions,
        dropout=0,
        channel_mult=(1, 2, 4, 8),
        conv_resample=True,
        dims=3,
        num_classes=None,
        use_checkpoint=False,
        use_fp16=False,
        num_heads=-1,
        num_head_channels=-1,
        num_heads_upsample=-1,
        use_scale_shift_norm=False,
        resblock_updown=False,
        transformer_depth=1,              # custom transformer support
        context_dim=None,                 # custom transformer support
        legacy=True,
        # temporal related
        kernel_size_t=1,
        padding_t=1,
        use_temporal_transformer=True,
        temporal_length=None,
        use_relative_position=False,
        cross_attn_on_tempoal=False,
        temporal_crossattn_type="crossattn",
        order="stst",
        nonlinearity_type='silu',
        temporalcrossfirst=False,
        split_stcontext=False,
        temporal_context_dim=None,
        use_tempoal_causal_attn=False,
        ST_transformer_module='attention_temporal',
        ST_transformer_class='SpatialTemporalTransformer',
        **kwargs,
    ):
        super().__init__()
        assert(use_temporal_transformer)
        if context_dim is not None:
            if type(context_dim) == ListConfig:
                context_dim = list(context_dim)

        if num_heads_upsample == -1:
            num_heads_upsample = num_heads

        if num_heads == -1:
            assert num_head_channels != -1, 'Either num_heads or num_head_channels has to be set'

        if num_head_channels == -1:
            assert num_heads != -1, 'Either num_heads or num_head_channels has to be set'

        self.image_size = image_size
        self.in_channels = in_channels
        self.model_channels = model_channels
        self.out_channels = out_channels
        self.num_res_blocks = num_res_blocks
        self.attention_resolutions = attention_resolutions
        self.dropout = dropout
        self.channel_mult = channel_mult
        self.conv_resample = conv_resample
        self.num_classes = num_classes
        self.use_checkpoint = use_checkpoint
        self.dtype = th.float16 if use_fp16 else th.float32
        self.num_heads = num_heads
        self.num_head_channels = num_head_channels
        self.num_heads_upsample = num_heads_upsample

        self.use_relative_position = use_relative_position
        self.temporal_length = temporal_length
        self.cross_attn_on_tempoal = cross_attn_on_tempoal
        self.temporal_crossattn_type = temporal_crossattn_type
        self.order = order
        self.temporalcrossfirst = temporalcrossfirst
        self.split_stcontext = split_stcontext
        self.temporal_context_dim = temporal_context_dim
        self.nonlinearity_type = nonlinearity_type
        self.use_tempoal_causal_attn = use_tempoal_causal_attn
        

        time_embed_dim = model_channels * 4
        self.time_embed_dim = time_embed_dim
        self.time_embed = nn.Sequential(
            linear(model_channels, time_embed_dim),
            nonlinearity(nonlinearity_type),
            linear(time_embed_dim, time_embed_dim),
        )

        if self.num_classes is not None:
            self.label_emb = nn.Embedding(num_classes, time_embed_dim)
        
        STTransformerClass = make_spatialtemporal_transformer(module_name=ST_transformer_module, 
            class_name=ST_transformer_class)

        self.input_blocks = nn.ModuleList(
            [
                TimestepEmbedSequential(
                    conv_nd(dims, in_channels, model_channels, (kernel_size_t, 3,3), padding=(padding_t, 1,1))
                )
            ]
        )
        self._feature_size = model_channels
        input_block_chans = [model_channels]
        ch = model_channels
        ds = 1
        for level, mult in enumerate(channel_mult):
            for _ in range(num_res_blocks):
                layers = [
                    ResBlock(
                        ch,
                        time_embed_dim,
                        dropout,
                        out_channels=mult * model_channels,
                        dims=dims,
                        use_checkpoint=use_checkpoint,
                        use_scale_shift_norm=use_scale_shift_norm,
                        kernel_size_t=kernel_size_t,
                        padding_t=padding_t,
                        nonlinearity_type=nonlinearity_type,
                        **kwargs
                    )
                ]
                ch = mult * model_channels
                if ds in attention_resolutions:
                    if num_head_channels == -1:
                        dim_head = ch // num_heads
                    else:
                        num_heads = ch // num_head_channels
                        dim_head = num_head_channels
                    if legacy:
                        dim_head = ch // num_heads if use_temporal_transformer else num_head_channels
                    layers.append(STTransformerClass(
                            ch, num_heads, dim_head, depth=transformer_depth, context_dim=context_dim,
                            # temporal related
                            temporal_length=temporal_length,
                            use_relative_position=use_relative_position,
                            cross_attn_on_tempoal=cross_attn_on_tempoal,
                            temporal_crossattn_type=temporal_crossattn_type,
                            order=order,
                            temporalcrossfirst=temporalcrossfirst,
                            split_stcontext=split_stcontext,
                            temporal_context_dim=temporal_context_dim,
                            use_tempoal_causal_attn=use_tempoal_causal_attn,
                            **kwargs,
                            ))
                self.input_blocks.append(TimestepEmbedSequential(*layers))
                self._feature_size += ch
                input_block_chans.append(ch)
            if level != len(channel_mult) - 1:
                out_ch = ch
                self.input_blocks.append(
                    TimestepEmbedSequential(
                        ResBlock(
                            ch,
                            time_embed_dim,
                            dropout,
                            out_channels=out_ch,
                            dims=dims,
                            use_checkpoint=use_checkpoint,
                            use_scale_shift_norm=use_scale_shift_norm,
                            down=True,
                            kernel_size_t=kernel_size_t,
                            padding_t=padding_t,
                            nonlinearity_type=nonlinearity_type,
                            **kwargs
                        )
                        if resblock_updown
                        else Downsample(
                            ch, conv_resample, dims=dims, out_channels=out_ch, kernel_size_t=kernel_size_t, padding_t=padding_t
                        )
                    )
                )
                ch = out_ch
                input_block_chans.append(ch)
                ds *= 2
                self._feature_size += ch

        if num_head_channels == -1:
            dim_head = ch // num_heads
        else:
            num_heads = ch // num_head_channels
            dim_head = num_head_channels
        if legacy:
            dim_head = ch // num_heads if use_temporal_transformer else num_head_channels
        self.middle_block = TimestepEmbedSequential(
            ResBlock(
                ch,
                time_embed_dim,
                dropout,
                dims=dims,
                use_checkpoint=use_checkpoint,
                use_scale_shift_norm=use_scale_shift_norm,
                kernel_size_t=kernel_size_t,
                padding_t=padding_t,
                nonlinearity_type=nonlinearity_type,
                **kwargs
            ),
            STTransformerClass(
                            ch, num_heads, dim_head, depth=transformer_depth, context_dim=context_dim,
                            # temporal related
                            temporal_length=temporal_length,
                            use_relative_position=use_relative_position,
                            cross_attn_on_tempoal=cross_attn_on_tempoal,
                            temporal_crossattn_type=temporal_crossattn_type,
                            order=order,
                            temporalcrossfirst=temporalcrossfirst,
                            split_stcontext=split_stcontext,
                            temporal_context_dim=temporal_context_dim,
                            use_tempoal_causal_attn=use_tempoal_causal_attn,
                            **kwargs,
                        ),
            ResBlock(
                ch,
                time_embed_dim,
                dropout,
                dims=dims,
                use_checkpoint=use_checkpoint,
                use_scale_shift_norm=use_scale_shift_norm,
                kernel_size_t=kernel_size_t,
                padding_t=padding_t,
                nonlinearity_type=nonlinearity_type,
                **kwargs
            ),
        )
        self._feature_size += ch

        self.output_blocks = nn.ModuleList([])
        for level, mult in list(enumerate(channel_mult))[::-1]:
            for i in range(num_res_blocks + 1):
                ich = input_block_chans.pop()
                layers = [
                    ResBlock(
                        ch + ich,
                        time_embed_dim,
                        dropout,
                        out_channels=model_channels * mult,
                        dims=dims,
                        use_checkpoint=use_checkpoint,
                        use_scale_shift_norm=use_scale_shift_norm,
                        kernel_size_t=kernel_size_t,
                        padding_t=padding_t,
                        nonlinearity_type=nonlinearity_type,
                        **kwargs
                    )
                ]
                ch = model_channels * mult
                if ds in attention_resolutions:
                    if num_head_channels == -1:
                        dim_head = ch // num_heads
                    else:
                        num_heads = ch // num_head_channels
                        dim_head = num_head_channels
                    if legacy:
                        dim_head = ch // num_heads if use_temporal_transformer else num_head_channels
                    layers.append(
                        STTransformerClass(
                            ch, num_heads, dim_head, depth=transformer_depth, context_dim=context_dim,
                            # temporal related
                            temporal_length=temporal_length,
                            use_relative_position=use_relative_position,
                            cross_attn_on_tempoal=cross_attn_on_tempoal,
                            temporal_crossattn_type=temporal_crossattn_type,
                            order=order,
                            temporalcrossfirst=temporalcrossfirst,
                            split_stcontext=split_stcontext,
                            temporal_context_dim=temporal_context_dim,
                            use_tempoal_causal_attn=use_tempoal_causal_attn,
                            **kwargs,
                        )
                    )
                if level and i == num_res_blocks:
                    out_ch = ch
                    layers.append(
                        ResBlock(
                            ch,
                            time_embed_dim,
                            dropout,
                            out_channels=out_ch,
                            dims=dims,
                            use_checkpoint=use_checkpoint,
                            use_scale_shift_norm=use_scale_shift_norm,
                            up=True,
                            kernel_size_t=kernel_size_t,
                            padding_t=padding_t,
                            nonlinearity_type=nonlinearity_type,
                            **kwargs
                        )
                        if resblock_updown
                        else Upsample(ch, conv_resample, dims=dims, out_channels=out_ch, kernel_size_t=kernel_size_t, padding_t=padding_t)
                    )
                    ds //= 2
                self.output_blocks.append(TimestepEmbedSequential(*layers))
                self._feature_size += ch

        self.out = nn.Sequential(
            normalization(ch),
            nonlinearity(nonlinearity_type),
            zero_module(conv_nd(dims, model_channels, out_channels, (kernel_size_t, 3,3), padding=(padding_t, 1,1))),
        )
        

    def convert_to_fp16(self):
        """
        Convert the torso of the model to float16.
        """
        self.input_blocks.apply(convert_module_to_f16)
        self.middle_block.apply(convert_module_to_f16)
        self.output_blocks.apply(convert_module_to_f16)

    def convert_to_fp32(self):
        """
        Convert the torso of the model to float32.
        """
        self.input_blocks.apply(convert_module_to_f32)
        self.middle_block.apply(convert_module_to_f32)
        self.output_blocks.apply(convert_module_to_f32)

    def forward(self, x, timesteps=None, time_emb_replace=None, context=None, y=None, **kwargs):
        """
        Apply the model to an input batch.
        :param x: an [N x C x ...] Tensor of inputs.
        :param timesteps: a 1-D batch of timesteps.
        :param context: conditioning plugged in via crossattn
        :param y: an [N] Tensor of labels, if class-conditional.
        :return: an [N x C x ...] Tensor of outputs.
        """
        
        hs = []
        if time_emb_replace is None:
            t_emb = timestep_embedding(timesteps, self.model_channels, repeat_only=False)
            emb = self.time_embed(t_emb)
        else:
            emb = time_emb_replace
        
        if y is not None: # if class-conditional model, inject class labels
            assert y.shape == (x.shape[0],)
            emb = emb + self.label_emb(y)

        h = x.type(self.dtype)
        for module in self.input_blocks:
            h = module(h, emb, context, **kwargs)
            hs.append(h)
        h = self.middle_block(h, emb, context, **kwargs)
        for module in self.output_blocks:
            h = th.cat([h, hs.pop()], dim=1)
            h = module(h, emb, context, **kwargs)
        h = h.type(x.dtype)
        return self.out(h)
