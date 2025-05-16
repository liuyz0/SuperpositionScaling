# do statistics for different tokenizers and datasets
from datasets import load_dataset
from transformers import AutoTokenizer
from collections import Counter
from tqdm import tqdm
import json
import os

# SETTINGS

# Create output directory
os.makedirs("token_freqs", exist_ok=True)

datasets_to_use = {
    "wikitext": {"name": "wikitext", "config": "wikitext-103-v1", "split": "train"},
    "c4": {"name": "c4", "config": "en", "split": "train"},
    "pile": {"name": "monology/pile-uncopyrighted", "split": "train"},
    "bookcorpus": {"name": "bookcorpus", "split": "train"},
}
tokenizers = { 
    "pythia": {"name": "EleutherAI/pythia-1b"},
    "opt": {"name": "facebook/opt-1.3B"},
    "Qwen": {"name": "Qwen/Qwen2.5-0.5B"},
    "gpt2": {"name": "openai-community/gpt2"}
}

TARGET_NUM_TOKENS = 1_000_000  # Stop once this many tokens have been processed

for tokenizer_key, tokenizer_val in tokenizers.items():
    print(f"Processing {tokenizer_key}...")
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_val['name'], cache_dir = "./cache")
    tokenizer.pad_token = tokenizer.eos_token if tokenizer.pad_token is None else tokenizer.pad_token

    token_freqs = {}

    for dataset_key, config in datasets_to_use.items():
        print(f"\nProcessing {dataset_key}...")

        # Load streaming dataset
        ds = load_dataset(config["name"], config.get("config", None), split=config["split"], 
                          streaming=True, trust_remote_code=True, cache_dir = "./cache")
        it = iter(ds)
        freq_counter = Counter()
        total_tokens = 0
        pbar = tqdm(total=TARGET_NUM_TOKENS, desc=f"{dataset_key} tokens")

        while total_tokens < TARGET_NUM_TOKENS:
            sample = next(it, None)
            if sample is None:
                break

            text = sample.get("text") or sample.get("content")
            if not text or not isinstance(text, str) or text.strip() == "":
                continue

            tokens = tokenizer.encode(text, add_special_tokens=False)
            if not tokens:
                continue

            freq_counter.update(tokens)
            total_tokens += len(tokens)
            pbar.update(len(tokens))

        pbar.close()
        token_freqs[dataset_key] = freq_counter

        # Save raw token frequency counts
        file_path = f"token_freqs/{tokenizer_key}_token_counts.json"
        with open(file_path, "w") as f:
            json.dump(token_freqs, f)
        print(f"Saved token frequencies for {tokenizer_key} to {file_path}")