# Copyright 2023 The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from dataclasses import dataclass
from typing import Optional, Tuple, Union

import torch
import torch.nn as nn

from ..configuration_utils import ConfigMixin, register_to_config
from ..utils import BaseOutput
from .embeddings import GaussianFourierProjection, TimestepEmbedding, Timesteps
from .modeling_utils import ModelMixin
from .unet_1d_blocks import get_down_block, get_mid_block, get_out_block, get_up_block


@dataclass
class UNet1DOutput(BaseOutput):
    """
    The output of [`UNet1DModel`].

    Args:
        sample (`torch.FloatTensor` of shape `(batch_size, num_channels, sample_size)`):
            The hidden states output from the last layer of the model.
    """

    sample: torch.FloatTensor


class UNet1DModel(ModelMixin, ConfigMixin):
    r"""
    A 1D UNet model that takes a noisy sample and a timestep and returns a sample shaped output.

    This model inherits from [`ModelMixin`]. Check the superclass documentation for it's generic methods implemented
    for all models (such as downloading or saving).

    Parameters:
        sample_size (`int`, *optional*): Default length of sample. Should be adaptable at runtime.
        in_channels (`int`, *optional*, defaults to 2): Number of channels in the input sample.
        out_channels (`int`, *optional*, defaults to 2): Number of channels in the output.
        extra_in_channels (`int`, *optional*, defaults to 0):
            Number of additional channels to be added to the input of the first down block. Useful for cases where the
            input data has more channels than what the model was initially designed for.
        time_embedding_type (`str`, *optional*, defaults to `"fourier"`): Type of time embedding to use.
        freq_shift (`float`, *optional*, defaults to 0.0): Frequency shift for Fourier time embedding.
        flip_sin_to_cos (`bool`, *optional*, defaults to `False`):
            Whether to flip sin to cos for Fourier time embedding.
        down_block_types (`Tuple[str]`, *optional*, defaults to `("DownBlock1DNoSkip", "DownBlock1D", "AttnDownBlock1D")`):
            Tuple of downsample block types.
        up_block_types (`Tuple[str]`, *optional*, defaults to `("AttnUpBlock1D", "UpBlock1D", "UpBlock1DNoSkip")`):
            Tuple of upsample block types.
        mid_block_type (`str`, *optional*, defaults to `"UNetMidBlock1D"`): Block type for middle of UNet.
        out_block_type (`str`, *optional*, defaults to `None`): Optional output processing block of UNet.
        block_out_channels (`Tuple[int]`, *optional*, defaults to `(32, 32, 64)`):
            Tuple of block output channels.
        layers_per_block (`int`, *optional*, defaults to 1): The number of layers per block.
        downsample_padding (`int`, *optional*, defaults to `1`): The padding for the downsample convolution.
        downsample_type (`str`, *optional*, defaults to `conv`):
            The downsample type for downsampling layers. Choose between "conv" and "resnet"
        dropout (`float`, *optional*, defaults to 0.0): The dropout probability to use.
        act_fn (`str`, defaults to `gelu`): activation function in UNet blocks.
        norm_num_groups (`int`, *optional*, defaults to 8): The number of groups for normalization.
        attn_norm_num_groups (`int`, *optional*, defaults to `None`):
            If set to an integer, a group norm layer will be created in the mid block's [`Attention`] layer with the
            given number of groups. If left as `None`, the group norm layer will only be created if
            `resnet_time_scale_shift` is set to `default`, and if created will have `norm_num_groups` groups.
        norm_eps (`float`, *optional*, defaults to `1e-5`): The epsilon for normalization.
        resnet_time_scale_shift (`str`, *optional*, defaults to `"default"`): Time scale shift config
            for ResNet blocks (see [`~models.resnet.ResnetBlock2D`]). Choose from `default` or `scale_shift`.
        downsample_each_block (`int`, *optional*, defaults to `False`):
            Experimental feature for using a UNet without upsampling.
    """

    @register_to_config
    def __init__(
        self,
        sample_size: int = 65536,
        sample_rate: Optional[int] = None,
        in_channels: int = 2,
        out_channels: int = 2,
        extra_in_channels: int = 0,
        time_embedding_type: str = "fourier",
        freq_shift: float = 0.0,
        flip_sin_to_cos: bool = True,
        use_timestep_embedding: bool = False,
        down_block_types: Tuple[str] = ("DownBlock1DNoSkip", "DownBlock1D", "AttnDownBlock1D"),
        up_block_types: Tuple[str] = ("AttnUpBlock1D", "UpBlock1D", "UpBlock1DNoSkip"),
        mid_block_type: Tuple[str] = "UNetMidBlock1D",
        out_block_type: str = None,
        block_out_channels: Tuple[int] = (32, 32, 64),
        layers_per_block: int = 1,
        mid_block_scale_factor: float = 1,
        downsample_each_block: bool = False,
        downsample_padding: int = 1,
        downsample_type: str = "conv",
        upsample_type: str = "conv",
        dropout: float = 0.0,
        act_fn: str = "gelu",
        attention_head_dim: Optional[int] = 8,
        norm_num_groups: int = 1,
        attn_norm_num_groups: Optional[int] = None,
        norm_eps: float = 1e-5,
        resnet_time_scale_shift: str = "default",
        add_attention: bool = True,
    ):
        super().__init__()
        self.sample_size = sample_size
        time_embed_dim = block_out_channels[0] * 4

        # Check inputs
        if len(down_block_types) != len(up_block_types):
            raise ValueError(
                f"Must provide the same number of `down_block_types` as `up_block_types`. `down_block_types`: {down_block_types}. `up_block_types`: {up_block_types}."
            )

        if len(block_out_channels) != len(down_block_types):
            raise ValueError(
                f"Must provide the same number of `block_out_channels` as `down_block_types`. `block_out_channels`: {block_out_channels}. `down_block_types`: {down_block_types}."
            )

        # input
        self.conv_in = nn.Conv1d(in_channels, block_out_channels[0], kernel_size=3, padding=1)

        # time
        if time_embedding_type == "fourier":
            self.time_proj = GaussianFourierProjection(
                embedding_size=8, set_W_to_weight=False, log=False, flip_sin_to_cos=flip_sin_to_cos
            )
            timestep_input_dim = 2 * block_out_channels[0]
        elif time_embedding_type == "positional":
            self.time_proj = Timesteps(
                block_out_channels[0], flip_sin_to_cos=flip_sin_to_cos, downscale_freq_shift=freq_shift
            )
            timestep_input_dim = block_out_channels[0]

        if use_timestep_embedding:
            self.time_mlp = TimestepEmbedding(
                in_channels=timestep_input_dim,
                time_embed_dim=time_embed_dim,
                act_fn=act_fn,
            )

        self.down_blocks = nn.ModuleList([])
        self.mid_block = None
        self.up_blocks = nn.ModuleList([])
        self.out_block = None

        # down
        output_channel = block_out_channels[0]
        for i, down_block_type in enumerate(down_block_types):
            input_channel = output_channel
            output_channel = block_out_channels[i]
            is_final_block = i == len(block_out_channels) - 1

            if i == 0:
                input_channel += extra_in_channels

            down_block = get_down_block(
                down_block_type,
                num_layers=layers_per_block,
                in_channels=input_channel,
                out_channels=output_channel,
                temb_channels=time_embed_dim,
                add_downsample=not is_final_block or downsample_each_block,
                resnet_eps=norm_eps,
                resnet_act_fn=act_fn,
                resnet_groups=norm_num_groups,
                attention_head_dim=attention_head_dim if attention_head_dim is not None else output_channel,
                downsample_padding=downsample_padding,
                resnet_time_scale_shift=resnet_time_scale_shift,
                downsample_type=downsample_type,
                dropout=dropout,
            )
            self.down_blocks.append(down_block)

        # mid
        self.mid_block = get_mid_block(
            mid_block_type,
            num_layers=layers_per_block,
            in_channels=block_out_channels[-1],
            out_channels=block_out_channels[-1],
            embed_dim=time_embed_dim,
            add_downsample=downsample_each_block,
            dropout=dropout,
            norm_eps=norm_eps,
            act_fn=act_fn,
            mid_block_scale_factor=mid_block_scale_factor,
            resnet_time_scale_shift=resnet_time_scale_shift,
            attention_head_dim=attention_head_dim if attention_head_dim is not None else block_out_channels[-1],
            norm_num_groups=norm_num_groups,
            attn_norm_num_groups=attn_norm_num_groups,
            add_attention=add_attention,
        )

        # up
        reversed_block_out_channels = list(reversed(block_out_channels))
        output_channel = reversed_block_out_channels[0]

        for i, up_block_type in enumerate(up_block_types):
            prev_output_channel = output_channel
            output_channel = reversed_block_out_channels[i]
            input_channel = reversed_block_out_channels[min(i + 1, len(block_out_channels) - 1)]

            is_final_block = i == len(block_out_channels) - 1

            up_block = get_up_block(
                up_block_type,
                num_layers=layers_per_block + 1,
                in_channels=input_channel,
                out_channels=output_channel,
                prev_output_channel=prev_output_channel,
                temb_channels=time_embed_dim,
                add_upsample=not is_final_block,
                resnet_eps=norm_eps,
                resnet_act_fn=act_fn,
                resnet_groups=norm_num_groups,
                attention_head_dim=attention_head_dim if attention_head_dim is not None else output_channel,
                resnet_time_scale_shift=resnet_time_scale_shift,
                upsample_type=upsample_type,
                dropout=dropout,
            )
            self.up_blocks.append(up_block)
            prev_output_channel = output_channel

        # out
        num_groups_out = norm_num_groups if norm_num_groups is not None else min(block_out_channels[0] // 4, 32)
        self.out_block = get_out_block(
            out_block_type=out_block_type,
            num_groups_out=num_groups_out,
            embed_dim=block_out_channels[0],
            out_channels=out_channels,
            act_fn=act_fn,
            fc_dim=block_out_channels[-1] // 4,
        )

    def forward(
        self,
        sample: torch.FloatTensor,
        timestep: Union[torch.Tensor, float, int],
        return_dict: bool = True,
    ) -> Union[UNet1DOutput, Tuple]:
        r"""
        The [`UNet1DModel`] forward method.

        Args:
            sample (`torch.FloatTensor`):
                The noisy input tensor with the following shape `(batch_size, num_channels, sample_size)`.
            timestep (`torch.FloatTensor` or `float` or `int`): The number of timesteps to denoise an input.
            return_dict (`bool`, *optional*, defaults to `True`):
                Whether or not to return a [`~models.unet_1d.UNet1DOutput`] instead of a plain tuple.

        Returns:
            [`~models.unet_1d.UNet1DOutput`] or `tuple`:
                If `return_dict` is True, an [`~models.unet_1d.UNet1DOutput`] is returned, otherwise a `tuple` is
                returned where the first element is the sample tensor.
        """

        # 1. time
        timesteps = timestep
        if not torch.is_tensor(timesteps):
            timesteps = torch.tensor([timesteps], dtype=torch.long, device=sample.device)
        elif torch.is_tensor(timesteps) and len(timesteps.shape) == 0:
            timesteps = timesteps[None].to(sample.device)

        timestep_embed = self.time_proj(timesteps)
        if self.config.use_timestep_embedding:
            timestep_embed = self.time_mlp(timestep_embed)
        else:
            timestep_embed = timestep_embed[..., None]
            timestep_embed = timestep_embed.repeat([1, 1, sample.shape[2]]).to(sample.dtype)
            timestep_embed = timestep_embed.broadcast_to((sample.shape[:1] + timestep_embed.shape[1:]))

        # 2. pre-process
        sample = self.conv_in(sample)

        # 3. down
        down_block_res_samples = (sample,)
        for downsample_block in self.down_blocks:
            sample, res_samples = downsample_block(hidden_states=sample, temb=timestep_embed)
            down_block_res_samples += res_samples

        # 4. mid
        if self.mid_block is not None:
            sample = self.mid_block(sample, timestep_embed)

        # 5. up
        for upsample_block in self.up_blocks:
            res_samples = down_block_res_samples[-len(upsample_block.resnets) :]
            down_block_res_samples = down_block_res_samples[: -len(upsample_block.resnets)]

            sample = upsample_block(sample, res_hidden_states_tuple=res_samples, temb=timestep_embed)

        # 6. post-process
        if self.out_block is not None:
            sample = self.out_block(sample, timestep_embed)

        if not return_dict:
            return (sample,)

        return UNet1DOutput(sample=sample)
