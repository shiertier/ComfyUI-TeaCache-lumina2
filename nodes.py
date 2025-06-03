"""
ComfyUI TeaCache for Lumina
Professional acceleration nodes for Lumina diffusion models using TeaCache technology.

This module provides ComfyUI nodes that implement TeaCache (Timestep Embedding Aware Cache)
specifically optimized for Lumina model architectures including Lumina2 and LuminaNext.
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Any, Dict, Optional, Tuple, Union, List


from diffusers import (
    Lumina2Transformer2DModel,
    Lumina2Pipeline,
    LuminaText2ImgPipeline,
)
from diffusers.models import LuminaNextDiT2DModel
from diffusers.models.modeling_outputs import Transformer2DModelOutput
from diffusers.utils import (
    USE_PEFT_BACKEND,
    logging,
    scale_lora_layers,
    unscale_lora_layers,
)

logger = logging.get_logger(__name__)

# TeaCache coefficients optimized for Lumina models
LUMINA_COEFFICIENTS = [
    393.76566581,
    -603.50993606,
    209.10239044,
    -23.00726601,
    0.86377344,
]


def poly1d(coefficients: List[float], x: torch.Tensor) -> torch.Tensor:
    """Compute polynomial evaluation for TeaCache rescaling function."""
    result = torch.zeros_like(x)
    for i, coeff in enumerate(coefficients):
        result += coeff * (x ** (len(coefficients) - 1 - i))
    return result


def teacache_lumina2_forward(
    self,
    hidden_states: torch.Tensor,
    timestep: torch.Tensor,
    encoder_hidden_states: torch.Tensor,
    encoder_attention_mask: torch.Tensor,
    attention_kwargs: Optional[Dict[str, Any]] = None,
    return_dict: bool = True,
) -> Union[torch.Tensor, Transformer2DModelOutput]:
    """
    TeaCache-enhanced forward pass for Lumina2 transformer models.

    This function replaces the original forward method to enable intelligent caching
    based on timestep embedding analysis.
    """

    # Handle LoRA scaling
    if attention_kwargs is not None:
        attention_kwargs = attention_kwargs.copy()
        lora_scale = attention_kwargs.pop("scale", 1.0)
    else:
        lora_scale = 1.0
    if USE_PEFT_BACKEND:
        scale_lora_layers(self, lora_scale)

    # Process embeddings
    batch_size, _, height, width = hidden_states.shape
    temb, encoder_hidden_states_processed = self.time_caption_embed(
        hidden_states, timestep, encoder_hidden_states
    )
    (
        image_patch_embeddings,
        context_rotary_emb,
        noise_rotary_emb,
        joint_rotary_emb,
        encoder_seq_lengths,
        seq_lengths,
    ) = self.rope_embedder(hidden_states, encoder_attention_mask)

    # Apply embedders and refiners
    image_patch_embeddings = self.x_embedder(image_patch_embeddings)
    for layer in self.context_refiner:
        encoder_hidden_states_processed = layer(
            encoder_hidden_states_processed, encoder_attention_mask, context_rotary_emb
        )
    for layer in self.noise_refiner:
        image_patch_embeddings = layer(image_patch_embeddings, None, noise_rotary_emb, temb)

    # Prepare main loop input
    max_seq_len = max(seq_lengths)
    input_to_main_loop = image_patch_embeddings.new_zeros(
        batch_size, max_seq_len, self.config.hidden_size
    )
    for i, (enc_len, seq_len_val) in enumerate(zip(encoder_seq_lengths, seq_lengths)):
        input_to_main_loop[i, :enc_len] = encoder_hidden_states_processed[i, :enc_len]
        input_to_main_loop[i, enc_len:seq_len_val] = image_patch_embeddings[i]

    # Handle variable sequence lengths
    use_mask = len(set(seq_lengths)) > 1
    attention_mask_for_main_loop_arg = None
    if use_mask:
        mask = input_to_main_loop.new_zeros(batch_size, max_seq_len, dtype=torch.bool)
        for i, (enc_len, seq_len_val) in enumerate(zip(encoder_seq_lengths, seq_lengths)):
            mask[i, :seq_len_val] = True
        attention_mask_for_main_loop_arg = mask

    # TeaCache decision logic
    should_calc = True
    if self.enable_teacache:
        cache_key = max_seq_len
        if cache_key not in self.cache:
            self.cache[cache_key] = {
                "accumulated_rel_l1_distance": 0.0,
                "previous_modulated_input": None,
                "previous_residual": None,
            }

        current_cache = self.cache[cache_key]
        modulated_inp, _, _, _ = self.layers[0].norm1(input_to_main_loop.clone(), temb.clone())

        if self.cnt == 0 or self.cnt == self.num_steps - 1:
            should_calc = True
            current_cache["accumulated_rel_l1_distance"] = 0.0
        else:
            if current_cache["previous_modulated_input"] is not None:
                rescale_func = np.poly1d(LUMINA_COEFFICIENTS)
                prev_mod_input = current_cache["previous_modulated_input"]
                prev_mean = prev_mod_input.abs().mean()

                if prev_mean.item() > 1e-9:
                    rel_l1_change = (
                        ((modulated_inp - prev_mod_input).abs().mean() / prev_mean).cpu().item()
                    )
                else:
                    rel_l1_change = (
                        0.0 if modulated_inp.abs().mean().item() < 1e-9 else float("inf")
                    )

                current_cache["accumulated_rel_l1_distance"] += rescale_func(rel_l1_change)

                if current_cache["accumulated_rel_l1_distance"] < self.rel_l1_thresh:
                    should_calc = False
                else:
                    should_calc = True
                    current_cache["accumulated_rel_l1_distance"] = 0.0
            else:
                should_calc = True
                current_cache["accumulated_rel_l1_distance"] = 0.0

        current_cache["previous_modulated_input"] = modulated_inp.clone()

        # Update step counter
        if not hasattr(self, "uncond_seq_len"):
            self.uncond_seq_len = cache_key
        if cache_key != self.uncond_seq_len:
            self.cnt += 1
            if self.cnt >= self.num_steps:
                self.cnt = 0

    # Execute main computation or use cache
    if self.enable_teacache and not should_calc:
        processed_hidden_states = input_to_main_loop + self.cache[max_seq_len]["previous_residual"]
    else:
        ori_input = input_to_main_loop.clone()
        current_processing_states = input_to_main_loop
        for layer in self.layers:
            current_processing_states = layer(
                current_processing_states,
                attention_mask_for_main_loop_arg,
                joint_rotary_emb,
                temb,
            )

        if self.enable_teacache:
            self.cache[max_seq_len]["previous_residual"] = current_processing_states - ori_input
        processed_hidden_states = current_processing_states

    # Final processing and output
    output_after_norm = self.norm_out(processed_hidden_states, temb)
    p = self.config.patch_size
    final_output_list = []
    for i, (enc_len, seq_len_val) in enumerate(zip(encoder_seq_lengths, seq_lengths)):
        image_part = output_after_norm[i][enc_len:seq_len_val]
        h_p, w_p = height // p, width // p
        reconstructed_image = (
            image_part.view(h_p, w_p, p, p, self.out_channels)
            .permute(4, 0, 2, 1, 3)
            .flatten(3, 4)
            .flatten(1, 2)
        )
        final_output_list.append(reconstructed_image)

    final_output_tensor = torch.stack(final_output_list, dim=0)

    if USE_PEFT_BACKEND:
        unscale_lora_layers(self, lora_scale)

    return Transformer2DModelOutput(sample=final_output_tensor)


def teacache_lumina_next_forward(
    self,
    hidden_states: torch.Tensor,
    timestep: torch.Tensor,
    encoder_hidden_states: torch.Tensor,
    encoder_mask: torch.Tensor,
    image_rotary_emb: torch.Tensor,
    cross_attention_kwargs: Dict[str, Any] = None,
    return_dict=True,
) -> torch.Tensor:
    """
    TeaCache-enhanced forward pass for LuminaNext DiT models.

    This function replaces the original forward method to enable intelligent caching
    based on timestep embedding analysis.
    """

    # Process patches and embeddings
    hidden_states, mask, img_size, image_rotary_emb = self.patch_embedder(
        hidden_states, image_rotary_emb
    )
    image_rotary_emb = image_rotary_emb.to(hidden_states.device)
    temb = self.time_caption_embed(timestep, encoder_hidden_states, encoder_mask)
    encoder_mask = encoder_mask.bool()

    # TeaCache decision logic
    if self.enable_teacache:
        inp = hidden_states.clone()
        temb_ = temb.clone()
        modulated_inp, gate_msa, scale_mlp, gate_mlp = self.layers[0].norm1(inp, temb_)

        if self.cnt == 0 or self.cnt == self.num_steps - 1:
            should_calc = True
            self.accumulated_rel_l1_distance = 0
        else:
            rescale_func = np.poly1d(LUMINA_COEFFICIENTS)
            self.accumulated_rel_l1_distance += rescale_func(
                (
                    (modulated_inp - self.previous_modulated_input).abs().mean()
                    / self.previous_modulated_input.abs().mean()
                )
                .cpu()
                .item()
            )
            if self.accumulated_rel_l1_distance < self.rel_l1_thresh:
                should_calc = False
            else:
                should_calc = True
                self.accumulated_rel_l1_distance = 0

        self.previous_modulated_input = modulated_inp
        self.cnt += 1
        if self.cnt == self.num_steps:
            self.cnt = 0

    # Execute main computation or use cache
    if self.enable_teacache:
        if not should_calc:
            hidden_states += self.previous_residual
        else:
            ori_hidden_states = hidden_states.clone()
            for layer in self.layers:
                hidden_states = layer(
                    hidden_states,
                    mask,
                    image_rotary_emb,
                    encoder_hidden_states,
                    encoder_mask,
                    temb=temb,
                    cross_attention_kwargs=cross_attention_kwargs,
                )
            self.previous_residual = hidden_states - ori_hidden_states
    else:
        for layer in self.layers:
            hidden_states = layer(
                hidden_states,
                mask,
                image_rotary_emb,
                encoder_hidden_states,
                encoder_mask,
                temb=temb,
                cross_attention_kwargs=cross_attention_kwargs,
            )

    # Final processing and unpatchify
    hidden_states = self.norm_out(hidden_states, temb)
    height_tokens = width_tokens = self.patch_size
    height, width = img_size[0]
    batch_size = hidden_states.size(0)
    sequence_length = (height // height_tokens) * (width // width_tokens)
    hidden_states = hidden_states[:, :sequence_length].view(
        batch_size,
        height // height_tokens,
        width // width_tokens,
        height_tokens,
        width_tokens,
        self.out_channels,
    )
    output = hidden_states.permute(0, 5, 1, 3, 2, 4).flatten(4, 5).flatten(2, 3)

    if not return_dict:
        return (output,)

    return Transformer2DModelOutput(sample=output)


class TeaCacheForLumina2:
    """ComfyUI node for applying TeaCache acceleration to Lumina2 models."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("MODEL", {"tooltip": "Lumina2 model to accelerate with TeaCache"}),
                "enable_teacache": (
                    "BOOLEAN",
                    {"default": True, "tooltip": "Enable TeaCache acceleration"},
                ),
                "rel_l1_thresh": (
                    "FLOAT",
                    {
                        "default": 0.3,
                        "min": 0.0,
                        "max": 10.0,
                        "step": 0.01,
                        "tooltip": "Cache threshold (higher = more aggressive caching)",
                    },
                ),
                "num_inference_steps": (
                    "INT",
                    {
                        "default": 30,
                        "min": 1,
                        "max": 1000,
                        "step": 1,
                        "tooltip": "Number of inference steps",
                    },
                ),
            }
        }

    RETURN_TYPES = ("MODEL",)
    RETURN_NAMES = ("model",)
    FUNCTION = "apply_teacache"
    CATEGORY = "TeaCache/Lumina"
    TITLE = "TeaCache for Lumina2"
    DESCRIPTION = "Apply TeaCache acceleration to Lumina2 transformer models"

    def apply_teacache(
        self,
        model,
        enable_teacache: bool,
        rel_l1_thresh: float,
        num_inference_steps: int,
    ):
        """Apply TeaCache optimization to Lumina2 model."""

        model_copy = model.clone()
        transformer = getattr(model_copy.model, "diffusion_model", None)

        if transformer is None:
            raise ValueError("Could not find transformer component in model")

        if not isinstance(transformer, Lumina2Transformer2DModel):
            print(
                f"Warning: Model type {type(transformer).__name__} "
                "may not be compatible with Lumina2 TeaCache"
            )

        # Apply TeaCache modifications
        transformer.__class__.forward = teacache_lumina2_forward
        transformer.__class__.enable_teacache = enable_teacache
        transformer.__class__.rel_l1_thresh = rel_l1_thresh
        transformer.__class__.num_steps = num_inference_steps
        transformer.__class__.cnt = 0
        transformer.__class__.cache = {}
        transformer.__class__.uncond_seq_len = None

        print(
            f"Applied Lumina2 TeaCache: enabled={enable_teacache}, "
            f"threshold={rel_l1_thresh}, steps={num_inference_steps}"
        )
        return (model_copy,)


class TeaCacheForLuminaNext:
    """ComfyUI node for applying TeaCache acceleration to LuminaNext models."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("MODEL", {"tooltip": "LuminaNext model to accelerate with TeaCache"}),
                "enable_teacache": (
                    "BOOLEAN",
                    {"default": True, "tooltip": "Enable TeaCache acceleration"},
                ),
                "rel_l1_thresh": (
                    "FLOAT",
                    {
                        "default": 0.3,
                        "min": 0.0,
                        "max": 10.0,
                        "step": 0.01,
                        "tooltip": "Cache threshold (higher = more aggressive caching)",
                    },
                ),
                "num_inference_steps": (
                    "INT",
                    {
                        "default": 30,
                        "min": 1,
                        "max": 1000,
                        "step": 1,
                        "tooltip": "Number of inference steps",
                    },
                ),
            }
        }

    RETURN_TYPES = ("MODEL",)
    RETURN_NAMES = ("model",)
    FUNCTION = "apply_teacache"
    CATEGORY = "TeaCache/Lumina"
    TITLE = "TeaCache for LuminaNext"
    DESCRIPTION = "Apply TeaCache acceleration to LuminaNext DiT models"

    def apply_teacache(
        self,
        model,
        enable_teacache: bool,
        rel_l1_thresh: float,
        num_inference_steps: int,
    ):
        """Apply TeaCache optimization to LuminaNext model."""

        model_copy = model.clone()
        transformer = getattr(model_copy.model, "diffusion_model", None)

        if transformer is None:
            raise ValueError("Could not find transformer component in model")

        if not isinstance(transformer, LuminaNextDiT2DModel):
            print(
                f"Warning: Model type {type(transformer).__name__} "
                "may not be compatible with LuminaNext TeaCache"
            )

        # Apply TeaCache modifications
        transformer.__class__.forward = teacache_lumina_next_forward
        transformer.__class__.enable_teacache = enable_teacache
        transformer.__class__.rel_l1_thresh = rel_l1_thresh
        transformer.__class__.num_steps = num_inference_steps
        transformer.__class__.cnt = 0
        transformer.__class__.accumulated_rel_l1_distance = 0
        transformer.__class__.previous_modulated_input = None
        transformer.__class__.previous_residual = None

        print(
            f"Applied LuminaNext TeaCache: enabled={enable_teacache}, "
            f"threshold={rel_l1_thresh}, steps={num_inference_steps}"
        )
        return (model_copy,)


class TeaCacheForLuminaAuto:
    """ComfyUI node for automatically detecting Lumina model type and applying appropriate TeaCache."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("MODEL", {"tooltip": "Lumina model (auto-detect type)"}),
                "enable_teacache": (
                    "BOOLEAN",
                    {"default": True, "tooltip": "Enable TeaCache acceleration"},
                ),
                "rel_l1_thresh": (
                    "FLOAT",
                    {
                        "default": 0.3,
                        "min": 0.0,
                        "max": 10.0,
                        "step": 0.01,
                        "tooltip": "Cache threshold (higher = more aggressive caching)",
                    },
                ),
                "num_inference_steps": (
                    "INT",
                    {
                        "default": 30,
                        "min": 1,
                        "max": 1000,
                        "step": 1,
                        "tooltip": "Number of inference steps",
                    },
                ),
            }
        }

    RETURN_TYPES = ("MODEL",)
    RETURN_NAMES = ("model",)
    FUNCTION = "apply_teacache"
    CATEGORY = "TeaCache/Lumina"
    TITLE = "TeaCache for Lumina (Auto)"
    DESCRIPTION = (
        "Automatically detect Lumina model type and apply appropriate TeaCache acceleration"
    )

    def apply_teacache(
        self,
        model,
        enable_teacache: bool,
        rel_l1_thresh: float,
        num_inference_steps: int,
    ):
        """Automatically detect Lumina model type and apply appropriate TeaCache."""

        model_copy = model.clone()
        transformer = getattr(model_copy.model, "diffusion_model", None)

        if transformer is None:
            raise ValueError("Could not find transformer component in model")

        # Auto-detect model type and apply appropriate TeaCache
        if isinstance(transformer, Lumina2Transformer2DModel):
            print("Detected Lumina2 model, applying Lumina2 TeaCache")
            return self._apply_lumina2_teacache(
                model_copy, transformer, enable_teacache, rel_l1_thresh, num_inference_steps
            )
        elif isinstance(transformer, LuminaNextDiT2DModel):
            print("Detected LuminaNext model, applying LuminaNext TeaCache")
            return self._apply_lumina_next_teacache(
                model_copy, transformer, enable_teacache, rel_l1_thresh, num_inference_steps
            )
        else:
            # Fallback: try to detect by class name
            class_name = transformer.__class__.__name__
            if "Lumina2" in class_name:
                print(f"Detected Lumina2-compatible model: {class_name}")
                return self._apply_lumina2_teacache(
                    model_copy, transformer, enable_teacache, rel_l1_thresh, num_inference_steps
                )
            elif "LuminaNext" in class_name or "Lumina" in class_name:
                print(f"Detected LuminaNext-compatible model: {class_name}")
                return self._apply_lumina_next_teacache(
                    model_copy, transformer, enable_teacache, rel_l1_thresh, num_inference_steps
                )
            else:
                raise ValueError(
                    f"Unsupported model type: {type(transformer).__name__}. "
                    "Supported types: Lumina2Transformer2DModel, LuminaNextDiT2DModel"
                )

    def _apply_lumina2_teacache(
        self, model, transformer, enable_teacache, rel_l1_thresh, num_inference_steps
    ):
        """Apply Lumina2 TeaCache configuration."""
        transformer.__class__.forward = teacache_lumina2_forward
        transformer.__class__.enable_teacache = enable_teacache
        transformer.__class__.rel_l1_thresh = rel_l1_thresh
        transformer.__class__.num_steps = num_inference_steps
        transformer.__class__.cnt = 0
        transformer.__class__.cache = {}
        transformer.__class__.uncond_seq_len = None

        print(
            f"Applied Lumina2 TeaCache: enabled={enable_teacache}, "
            f"threshold={rel_l1_thresh}, steps={num_inference_steps}"
        )
        return (model,)

    def _apply_lumina_next_teacache(
        self, model, transformer, enable_teacache, rel_l1_thresh, num_inference_steps
    ):
        """Apply LuminaNext TeaCache configuration."""
        transformer.__class__.forward = teacache_lumina_next_forward
        transformer.__class__.enable_teacache = enable_teacache
        transformer.__class__.rel_l1_thresh = rel_l1_thresh
        transformer.__class__.num_steps = num_inference_steps
        transformer.__class__.cnt = 0
        transformer.__class__.accumulated_rel_l1_distance = 0
        transformer.__class__.previous_modulated_input = None
        transformer.__class__.previous_residual = None

        print(
            f"Applied LuminaNext TeaCache: enabled={enable_teacache}, "
            f"threshold={rel_l1_thresh}, steps={num_inference_steps}"
        )
        return (model,)


# Node registration
NODE_CLASS_MAPPINGS = {
    "TeaCacheForLumina2": TeaCacheForLumina2,
    "TeaCacheForLuminaNext": TeaCacheForLuminaNext,
    "TeaCacheForLuminaAuto": TeaCacheForLuminaAuto,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "TeaCacheForLumina2": "TeaCache for Lumina2",
    "TeaCacheForLuminaNext": "TeaCache for LuminaNext",
    "TeaCacheForLuminaAuto": "TeaCache for Lumina (Auto)",
}
