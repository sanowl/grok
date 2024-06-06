# Copyright 2024 X.AI Corp.
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

import logging
from typing import Any

from model import LanguageModelConfig, TransformerConfig, QuantizedWeight8bit as QW8Bit
from runners import InferenceRunner, ModelRunner, sample_from_model

CKPT_PATH = "./checkpoints/"
TOKENIZER_PATH = "./tokenizer.model"


def configure_model() -> LanguageModelConfig:
    """Configure and return the language model configuration."""
    return LanguageModelConfig(
        vocab_size=128 * 1024,
        pad_token=0,
        eos_token=2,
        sequence_len=8192,
        embedding_init_scale=1.0,
        output_multiplier_scale=0.5773502691896257,
        embedding_multiplier_scale=78.38367176906169,
        model=TransformerConfig(
            emb_size=48 * 128,
            widening_factor=8,
            key_size=128,
            num_q_heads=48,
            num_kv_heads=8,
            num_layers=64,
            attn_output_multiplier=0.08838834764831845,
            shard_activations=True,
            # MoE.
            num_experts=8,
            num_selected_experts=2,
            # Activation sharding.
            data_axis="data",
            model_axis="model",
        ),
    )


def setup_inference_runner(model_config: LanguageModelConfig) -> InferenceRunner:
    """Setup and return the inference runner."""
    return InferenceRunner(
        pad_sizes=(1024,),
        runner=ModelRunner(
            model=model_config,
            bs_per_device=0.125,
            checkpoint_path=CKPT_PATH,
        ),
        name="local",
        load=CKPT_PATH,
        tokenizer_path=TOKENIZER_PATH,
        local_mesh_config=(1, 8),
        between_hosts_config=(1, 1),
    )


def generate_text(inference_runner: InferenceRunner, prompt: str, max_len: int = 100, temperature: float = 0.01) -> str:
    """Generate text using the model."""
    return sample_from_model(inference_runner.run(), prompt, max_len=max_len, temperature=temperature)


def main() -> None:
    """Main function to initialize the model and generate text."""
    logging.basicConfig(level=logging.INFO)
    try:
        model_config = configure_model()
        inference_runner = setup_inference_runner(model_config)
        inference_runner.initialize()
        
        prompt = "The answer to life the universe and everything is of course"
        output = generate_text(inference_runner, prompt)
        print(f"Output for prompt: {prompt}\n{output}")
    except Exception as e:
        logging.error("An error occurred during model initialization or inference.", exc_info=True)


if __name__ == "__main__":
    main()
