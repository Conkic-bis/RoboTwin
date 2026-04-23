# Qwen3-VL Vision-Language Model Interface
#
# Wraps Qwen3VLForConditionalGeneration as a vision-language backbone
# for robotic manipulation. Extracts hidden states from images + text
# instructions to condition the flow matching action head.
#

import torch
import torch.nn as nn
from typing import List, Optional, Dict, Any


class Qwen3VLInterface(nn.Module):
    """
    Wrapper around Qwen3-VL for extracting vision-language features.

    Processes multi-camera images and text instructions through Qwen3-VL,
    returning the last hidden states as conditioning for the action model.

    Args:
        model_name: HuggingFace model name or local path
        freeze: Whether to freeze VLM weights (default: True)
        use_lora: Whether to apply LoRA for fine-tuning (default: False)
        lora_r: LoRA rank (default: 16)
        lora_alpha: LoRA alpha (default: 32)
        lora_target_modules: LoRA target modules
        max_pixels: Maximum pixels per image for Qwen3-VL processor
        min_pixels: Minimum pixels per image for Qwen3-VL processor
    """

    def __init__(
        self,
        model_name: str = "Qwen/Qwen3-VL-4B-Instruct",
        freeze: bool = True,
        use_lora: bool = False,
        lora_r: int = 16,
        lora_alpha: int = 32,
        lora_target_modules: Optional[List[str]] = None,
        max_pixels: int = 401408,
        min_pixels: int = 100352,
    ):
        super().__init__()

        self.model_name = model_name
        self.freeze = freeze
        self.use_lora = use_lora

        # Import here to avoid top-level dependency
        from transformers import Qwen3VLForConditionalGeneration, AutoProcessor

        # Load model in bfloat16 for efficiency
        self.model = Qwen3VLForConditionalGeneration.from_pretrained(
            model_name,
            torch_dtype=torch.bfloat16,
            attn_implementation="flash_attention_2",
        )

        # Load processor for tokenization and image processing
        self.processor = AutoProcessor.from_pretrained(
            model_name,
            max_pixels=max_pixels,
            min_pixels=min_pixels,
        )

        # Get text hidden size from model config.
        # Qwen3-VL uses a composite config where the language-model hidden
        # size lives under ``config.text_config`` (not at the top level like
        # earlier Qwen2-VL checkpoints). Fall back to the top-level attribute
        # if the composite field is unavailable, for forward/backward
        # compatibility.
        cfg = self.model.config
        if hasattr(cfg, "text_config") and hasattr(cfg.text_config, "hidden_size"):
            self.hidden_size = cfg.text_config.hidden_size
        elif hasattr(cfg, "hidden_size"):
            self.hidden_size = cfg.hidden_size
        else:
            raise AttributeError(
                f"Could not locate hidden_size on {type(cfg).__name__}. "
                "Expected either config.text_config.hidden_size or "
                "config.hidden_size."
            )

        # Freeze VLM weights if specified
        if freeze and not use_lora:
            for param in self.model.parameters():
                param.requires_grad = False

        # Apply LoRA if specified
        if use_lora:
            self._apply_lora(lora_r, lora_alpha, lora_target_modules)

    def _apply_lora(self, lora_r, lora_alpha, target_modules):
        """Apply LoRA adapters to the VLM."""
        from peft import LoraConfig, get_peft_model

        if target_modules is None:
            target_modules = ["q_proj", "v_proj", "k_proj", "o_proj"]

        lora_config = LoraConfig(
            r=lora_r,
            lora_alpha=lora_alpha,
            target_modules=target_modules,
            lora_dropout=0.05,
            bias="none",
            task_type="CAUSAL_LM",
        )
        self.model = get_peft_model(self.model, lora_config)

    def build_inputs(
        self,
        images: List[Any],
        instruction: str = "Predict the next robot actions.",
        robot_state: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Build Qwen3-VL chat-format inputs from images and instruction.

        Uses the Qwen3-VL processor's ``apply_chat_template`` with
        ``tokenize=True, return_dict=True`` which handles image loading
        and tokenization in a single call (no ``qwen_vl_utils`` dependency).

        Args:
            images: List of PIL Images or numpy arrays (one per camera view)
            instruction: Text instruction for the task
            robot_state: Optional robot state tensor for discretized state input

        Returns:
            Dictionary of tokenized inputs ready for model forward pass
        """
        from PIL import Image
        import numpy as np

        # Build content list with images (convert numpy -> PIL so the
        # processor accepts them directly via the chat template).
        content = []
        for img in images:
            if isinstance(img, np.ndarray):
                img = Image.fromarray(img.astype(np.uint8))
            content.append({"type": "image", "image": img})

        # Add robot state as text if provided
        if robot_state is not None:
            # Discretize state to 256 bins (OpenPI convention)
            state_np = robot_state.detach().float().cpu().numpy().flatten()
            state_tokens = " ".join([str(int(v * 127.5 + 127.5)) for v in np.clip(state_np, -1, 1)])
            content.append({"type": "text", "text": f"Robot state: [{state_tokens}]\n{instruction}"})
        else:
            content.append({"type": "text", "text": instruction})

        messages = [{"role": "user", "content": content}]

        # Qwen3-VL processor does text + vision preprocessing in one shot
        # when tokenize=True, return_dict=True. This matches the official
        # Qwen3-VL-4B-Instruct usage and removes the qwen_vl_utils dependency.
        inputs = self.processor.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_dict=True,
            return_tensors="pt",
        )

        return inputs

    @torch.amp.autocast("cuda", dtype=torch.bfloat16)
    def forward(
        self,
        images: Optional[List[Any]] = None,
        instruction: str = "Predict the next robot actions.",
        robot_state: Optional[torch.Tensor] = None,
        preprocessed_inputs: Optional[Dict[str, torch.Tensor]] = None,
    ) -> torch.Tensor:
        """
        Extract VLM hidden states from images + instruction.

        Args:
            images: List of PIL Images or numpy arrays (one per camera)
            instruction: Text instruction
            robot_state: Optional robot state for state conditioning
            preprocessed_inputs: Pre-tokenized inputs (skips build_inputs)

        Returns:
            hidden_states: (batch_size, seq_len, hidden_size) last hidden states
        """
        if preprocessed_inputs is not None:
            inputs = preprocessed_inputs
        else:
            inputs = self.build_inputs(images, instruction, robot_state)

        # Move inputs to model device
        device = next(self.model.parameters()).device
        inputs = {k: v.to(device) if isinstance(v, torch.Tensor) else v
                  for k, v in inputs.items()}

        # Forward pass to get hidden states
        outputs = self.model(
            **inputs,
            output_hidden_states=True,
            return_dict=True,
        )

        # Use last hidden state
        hidden_states = outputs.hidden_states[-1]  # (B, L, H)

        return hidden_states

    def get_hidden_size(self) -> int:
        """Return the VLM hidden dimension."""
        return self.hidden_size

    # --------------------------------------------------------------------- #
    # Batch path: single processor call + single model call for a full batch.
    # Removes the per-sample Python loop that VLM-branch's train_step used.
    # --------------------------------------------------------------------- #

    def build_batch_inputs(
        self,
        images_batch: List[List[Any]],
        instructions: List[str],
        robot_states: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Build batched Qwen3-VL inputs from a list of samples.

        Args:
            images_batch: length-B list; each entry is a list of per-camera
                images (numpy uint8 arrays or PIL Images) for one sample.
            instructions: length-B list of task instruction strings.
            robot_states: optional (B, state_dim) tensor of robot states.

        Returns:
            Dict of tensors as returned by the processor with padding=True.
            Notably contains:
                - input_ids        : (B, L_max)
                - attention_mask   : (B, L_max)  ← ← used to mask cross-attn
                - pixel_values     : concatenated image tensor
                - image_grid_thw   : (sum_over_B(num_images), 3)
        """
        from PIL import Image
        import numpy as np

        B = len(images_batch)
        assert len(instructions) == B, (
            f"VLM batch mismatch: images={B}, instructions={len(instructions)}"
        )
        if robot_states is not None:
            assert robot_states.shape[0] == B, (
                f"VLM batch mismatch: images={B}, robot_states={robot_states.shape[0]}"
            )

        messages_list = []
        for i in range(B):
            content = []
            for img in images_batch[i]:
                if isinstance(img, np.ndarray):
                    img = Image.fromarray(img.astype(np.uint8))
                content.append({"type": "image", "image": img})

            if robot_states is not None:
                state_np = robot_states[i].detach().float().cpu().numpy().flatten()
                state_tokens = " ".join(
                    str(int(v * 127.5 + 127.5))
                    for v in np.clip(state_np, -1, 1)
                )
                content.append({
                    "type": "text",
                    "text": f"Robot state: [{state_tokens}]\n{instructions[i]}",
                })
            else:
                content.append({"type": "text", "text": instructions[i]})

            messages_list.append([{"role": "user", "content": content}])

        # HF processors accept a list-of-conversations here. With padding=True
        # the returned input_ids is (B, L_max) right-padded, together with a
        # matching attention_mask. pixel_values is concatenated along batch.
        inputs = self.processor.apply_chat_template(
            messages_list,
            tokenize=True,
            add_generation_prompt=True,
            return_dict=True,
            return_tensors="pt",
            padding=True,
        )
        return inputs

    @torch.amp.autocast("cuda", dtype=torch.bfloat16)
    def forward_batch(
        self,
        images_batch: List[List[Any]],
        instructions: List[str],
        robot_states: Optional[torch.Tensor] = None,
    ) -> (torch.Tensor, torch.Tensor):
        """
        Batched VLM forward. Single model call for the whole batch.

        Returns:
            hidden_states  : (B, L_max, H_vlm) last hidden states
            attention_mask : (B, L_max) 1=real token, 0=pad
        """
        inputs = self.build_batch_inputs(images_batch, instructions, robot_states)

        device = next(self.model.parameters()).device
        inputs = {
            k: (v.to(device) if isinstance(v, torch.Tensor) else v)
            for k, v in inputs.items()
        }

        outputs = self.model(
            **inputs,
            output_hidden_states=True,
            return_dict=True,
        )

        hidden_states = outputs.hidden_states[-1]   # (B, L_max, H_vlm)
        attention_mask = inputs["attention_mask"]   # (B, L_max), int
        return hidden_states, attention_mask


def get_vlm_model(model_name: str = "Qwen/Qwen3-VL-4B-Instruct", **kwargs) -> Qwen3VLInterface:
    """Factory function to create a Qwen3-VL interface."""
    return Qwen3VLInterface(model_name=model_name, **kwargs)
