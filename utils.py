import os, random, gc, json, textwrap
import numpy as np
import tensorflow as tf

import pprint, itertools, math, time
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
from scipy import stats
from rouge_score import rouge_scorer
# import torch

import keras
import keras_hub
import keras_nlp
from keras.callbacks import EarlyStopping


def set_global_seed(seed: int = 42):
    """Seed Python, NumPy, TF & Keras RNGs (see TF docs for details)."""
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)
    tf.keras.utils.set_random_seed(seed)
    print(f"✅ Seed set to {seed}")


def estimate_tokens(text: str) -> int:
    """Rough token estimate (1 token ≈ 4 characters for most models)."""
    return len(text) // 4


def clear_memory(models=["gemma_lm", "base_model", "fresh_lm"]):
    """
    Free GPU memory before loading base model (prevents OOM errors)
    """
    for m in models:
        if m in globals():
            del globals()[m]
    tf.keras.backend.clear_session()
    gc.collect()


def filter_data_by_length(
    prompts,
    responses,
    max_input_tokens: int = 1024,
    min_prompt_len: int = 40,
    min_response_len: int = 20,
    end_token: str = "<end_of_turn>",
):
    """
    Clean + length-filter prompt/response pairs.
    """
    assert len(prompts) == len(responses), "Prompts and responses must align"

    filtered_prompts, filtered_responses = [], []

    # Drop pairs with None or empty strings.
    for prompt, response in zip(prompts, responses):
        if not prompt or not response:              # None or empty string
            continue

        prompt, response = prompt.strip(), response.strip()
        # Keep samples whose character lengths meet `min_*_len` and `max_input_tokens`
        if (
            len(prompt) < min_prompt_len
            or len(response) < min_response_len
            or estimate_tokens(prompt) > max_input_tokens
            or estimate_tokens(response) > max_input_tokens
        ):
            continue
        # Append `end_token`
        if not response.endswith(end_token):
            response += f"\n{end_token}"

        filtered_prompts.append(prompt)
        filtered_responses.append(response)

    kept = len(filtered_prompts)
    total = len(prompts)
    print(f"📊 Kept {kept:,} / {total:,} pairs ({kept/total*100:.1f}%)")

    return filtered_prompts, filtered_responses


def make_ds(inp, out):
    ds = tf.data.Dataset.from_tensor_slices({
        "prompts":   inp,
        "responses": out,
    })
    ds = ds.shuffle(buffer_size=1000, seed=SEED)
    # Batch-1 keeps variable-length strings intact; adjust if you pad/pack
    return ds.batch(1).prefetch(tf.data.AUTOTUNE)


def compile_with_sampler(model, k: int = 7, temperature: float = 0.7, seed: int = 42):
    sampler = keras_nlp.samplers.TopKSampler(k=k, temperature=temperature, seed=SEED)
    model.compile(sampler=sampler)
    return model


def compile_for_training(model):
    """
    Compile model for LoRA fine-tuning following Google's official tutorial.
    """
    optimizer = keras.optimizers.AdamW(
        learning_rate=5e-4, #3e-5,
        weight_decay=0.01,
    )
    # Exclude layernorm and bias terms from decay (critical for LoRA)
    optimizer.exclude_from_weight_decay(var_names=["bias", "scale"])

    model.compile(
        loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        optimizer=optimizer,
        weighted_metrics=[keras.metrics.SparseCategoricalAccuracy()],
    )
    print("✅ Model compiled for training with AdamW optimizer (bias/scale excluded from weight decay)")
    return model


def safe_generate(
        model,
        prompt: str,
        max_new_tokens: int = 1024,
        strip_prompt: bool = True,
        stop_tokens: list | None = None,
        **kw,
):
    """Length-safe wrapper around Gemma3CausalLM.generate()."""
    # 1 ) count prompt tokens
    prompt_len = len(model.preprocessor.generate_preprocess([prompt])["token_ids"][0])

    ceiling = getattr(getattr(model, "config", None), "max_position_embeddings", 8192)
    if prompt_len >= ceiling:
        raise ValueError(
            f"Prompt is {prompt_len} tokens but Gemma can accept at most {ceiling}."
        )

    # 2 ) new-token budget that still fits into the context window
    allowed_new = min(max_new_tokens, ceiling - prompt_len)

    # 3 ) Use only supported parameters for Gemma3CausalLM
    result = model.generate(
        prompt,
        max_length=prompt_len + allowed_new,
    )

    # 4 ) Manually strip prompt if needed
    if strip_prompt:
        result = result[len(prompt):].strip()

    return result


# Evaluate using ROUGE metrics (standard for summarization tasks)
# ROUGE-1/2: n-gram overlap, ROUGE-L: longest common subsequence
def evaluate_model(model, prompts, references, model_name="model"):
    scorer  = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
    scores  = {k: [] for k in ['rouge1', 'rouge2', 'rougeL']}
    outputs = []

    for prompt in tqdm(prompts, desc=f"Generating with {model_name}"):
        outputs.append(
            safe_generate(model, prompt, max_new_tokens=3000)
        )

    for pred, ref in zip(outputs, references):
      rouge = scorer.score(ref, pred)
      for k in scores:
          scores[k].append(rouge[k].fmeasure)


    return outputs, scores, {k: np.mean(v) for k, v in scores.items()}
