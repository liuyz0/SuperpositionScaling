# overlap analysis for all the models so far

import torch
import torch.nn.functional as F
from huggingface_hub import hf_hub_download
from safetensors.torch import load_file
import time
import math

batch_size = 8192

model_names = ["facebook/opt-125m",
               "facebook/opt-350m",
               "facebook/opt-1.3B",
               "facebook/opt-2.7B",
               "facebook/opt-6.7B",
               "facebook/opt-13B",
               "facebook/opt-30B",
               "facebook/opt-66b",
               "Qwen/Qwen2.5-0.5B",
               "Qwen/Qwen2.5-1.5B",
               "Qwen/Qwen2.5-3B",
               "Qwen/Qwen2.5-7B",
               "Qwen/Qwen2.5-14B",
               "Qwen/Qwen2.5-32B",
               "Qwen/Qwen2.5-72B",
               "openai-community/gpt2",
               "openai-community/gpt2-medium",
               "openai-community/gpt2-large",
               "openai-community/gpt2-xl",
               "EleutherAI/pythia-70m",
               "EleutherAI/pythia-160m",
               "EleutherAI/pythia-410m",
               "EleutherAI/pythia-1b",
               "EleutherAI/pythia-1.4b",
               "EleutherAI/pythia-2.8b",
               "EleutherAI/pythia-6.9b",
               "EleutherAI/pythia-12b"]

file_names = ["pytorch_model.bin",
              "pytorch_model.bin",
              "pytorch_model.bin",
              "pytorch_model.bin",
              "pytorch_model-00001-of-00002.bin",
              "pytorch_model-00001-of-00003.bin",
              "pytorch_model-00001-of-00007.bin",
              "pytorch_model-00014-of-00014.bin",
              "model.safetensors",
              "model.safetensors",
              "model-00001-of-00002.safetensors",
              "model-00004-of-00004.safetensors",
              "model-00008-of-00008.safetensors",
              "model-00017-of-00017.safetensors",
              "model-00037-of-00037.safetensors",
              "model.safetensors",
              "model.safetensors",
              "model.safetensors",
              "model.safetensors",
              "model.safetensors",
              "model.safetensors",
              "model.safetensors",
              "model.safetensors",
              "model.safetensors",
              "model.safetensors",
              "model-00002-of-00002.safetensors",
              "model-00003-of-00003.safetensors"]

weight_names = ["model.decoder.embed_tokens.weight",
                "model.decoder.embed_tokens.weight",
                "model.decoder.embed_tokens.weight",
                "model.decoder.embed_tokens.weight",
                "model.decoder.embed_tokens.weight",
                "model.decoder.embed_tokens.weight",
                "model.decoder.embed_tokens.weight",
                "lm_head.weight",
                "model.embed_tokens.weight",
                "model.embed_tokens.weight",
                "model.embed_tokens.weight",
                "lm_head.weight",
                "lm_head.weight",
                "lm_head.weight",
                "lm_head.weight",
                "wte.weight",
                "wte.weight",
                "wte.weight",
                "wte.weight",
                "embed_out.weight",
                "embed_out.weight",
                "embed_out.weight",
                "embed_out.weight",
                "embed_out.weight",
                "embed_out.weight",
                "embed_out.weight",
                "embed_out.weight"]

assert len(model_names) == len(file_names)
assert len(model_names) == len(weight_names)

for model_name, file_name, weight_name in zip(model_names, file_names, weight_names):
    print(f"Loading {model_name}...")
    start_time = time.time()
    result = {}
    file_path = hf_hub_download(model_name, file_name, cache_dir="./cache")
    weights = load_file(file_path, device="cuda") if 'safetensors' in file_name else torch.load(file_path, map_location="cuda")
    lm_head_weight = weights[weight_name] if weight_name in weights else weights["decoder.embed_tokens.weight"]
    weights = None

    result["n"] = lm_head_weight.size(0)
    result["m"] = lm_head_weight.size(1)

    overlap_sum = 0.0
    overlap_sq_sum = 0.0

    for i in range(0, result["n"], batch_size):
        end_i = min(i + batch_size, result["n"])
        W_i = lm_head_weight[i:end_i] / torch.linalg.norm(lm_head_weight[i:end_i], dim=-1, keepdim=True)

        for j in range(i, result["n"], batch_size):
            end_j = min(j + batch_size, result["n"])
            W_j = lm_head_weight[j:end_j] / torch.linalg.norm(lm_head_weight[j:end_j], dim=-1, keepdim=True)

            # Compute batched dot products
            dot_products = torch.matmul(W_i, W_j.T).abs()

            # For the diagonal block, exclude lower triangle and diagonal
            if i == j:
                dot_products = dot_products.triu(diagonal=1)

            overlap_sum += dot_products.to(torch.float64).sum().item() / (result["n"] * (result["n"] - 1) / 2)
            overlap_sq_sum += (dot_products ** 2).to(torch.float64).sum().item() / (result["n"] * (result["n"] - 1) / 2)

    result["mean_over"] = overlap_sum #math.sqrt
    result["std_over"] = (overlap_sq_sum - result["mean_over"] ** 2) ** 0.5

    print(f"Model {model_name} is analyzed. Time elapsed: {time.time() - start_time}.")
    torch.save(result, f"overlap-0-{model_name.replace('/', '_')}.pt")
