# model parallelism

import os
import json
import torch
from tqdm import tqdm
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
import argparse
from torch.utils.data import Dataset, DataLoader

class TextDataset(Dataset):
    def __init__(self, texts, tokenizer, max_length):
        self.tokenizer = tokenizer
        self.texts = texts
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]
        encodings = self.tokenizer(text, truncation=True, max_length=self.max_length, 
                                  return_tensors="pt")
        # Remove batch dimension
        for k, v in encodings.items():
            encodings[k] = v.squeeze()
        return encodings
    
def collate_fn(batch):
    input_ids = [item["input_ids"] for item in batch]
    attention_mask = [item["attention_mask"] for item in batch]
    
    # Pad to max length in batch
    max_length = max([len(ids) for ids in input_ids])
    
    # Manual padding
    padded_input_ids = []
    padded_attention_mask = []
    labeles = []
    
    for ids, mask in zip(input_ids, attention_mask):
        padding_length = max_length - len(ids)
        padded_input_ids.append(torch.cat([ids, torch.full((padding_length,), 0, dtype=torch.long)]))
        padded_attention_mask.append(torch.cat([mask, torch.zeros(padding_length, dtype=torch.long)]))
        labeles.append(torch.cat([ids, torch.full((padding_length,), -100, dtype=torch.long)]))
    
    # Stack to create tensors
    input_ids_tensor = torch.stack(padded_input_ids)
    attention_mask_tensor = torch.stack(padded_attention_mask)
    labeles_tensor = torch.stack(labeles)
    
    # For causal LM in HuggingFace, if you pass the same tensor for input_ids and labels,
    # the model will internally handle the shift for next-token prediction loss calculation
    return {
        "input_ids": input_ids_tensor,
        "attention_mask": attention_mask_tensor,
        "labels": labeles_tensor  # HuggingFace handles the shift internally
    }

def sample_dataset(dataset_name, num_samples=1000, max_seq_length=2048, seed=42):
    """Sample examples from a HF dataset efficiently with deterministic seeding"""
    # Set random seeds for reproducibility
    '''
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    '''
    
    if dataset_name == "wikitext":
        # Use fixed seed for dataset loading
        dataset = load_dataset("wikitext", "wikitext-103-v1", split="train", streaming=True,
                              trust_remote_code=True)
        texts = []
        current_text = ""
        
        for i, example in enumerate(dataset):
            if example["text"].strip():  # Skip empty lines
                current_text += example["text"] + " "
            
            # When we have enough text for a sample, save it
            if len(current_text.split()) >= max_seq_length // 2:  # Rough words to tokens conversion
                texts.append(current_text)
                current_text = ""
                
            if len(texts) >= num_samples:
                break
                
        return texts
    
    elif dataset_name == "pile":
        dataset = load_dataset("NeelNanda/pile-10k", split='train', streaming=True, trust_remote_code=True)
        
        texts = []
        
        for i, example in enumerate(dataset):
            texts.append(example["text"])
            if len(texts) >= num_samples:
                break
                
        return texts
    
    elif dataset_name == "c4":
        dataset = load_dataset("allenai/c4", "en", split="train", streaming=True,
                              trust_remote_code=True)
        texts = []
        
        for i, example in enumerate(dataset):
            texts.append(example["text"])
            if len(texts) >= num_samples:
                break
                
        return texts
    
    elif dataset_name == "bookcorpus":
        dataset = load_dataset("bookcorpus", split="train", streaming=True,
                              trust_remote_code=True)
        texts = []
        current_text = ""
        
        for i, example in enumerate(dataset):
            if example["text"].strip():  # Skip empty lines
                current_text += example["text"] + " "
            
            # When we have enough text for a sample, save it
            if len(current_text.split()) >= max_seq_length // 2:  # Rough words to tokens conversion
                texts.append(current_text)
                current_text = ""
                
            if len(texts) >= num_samples:
                break
                
        return texts
    
    else:
        raise ValueError(f"Dataset {dataset_name} not supported")

def evaluate_model(model_id, datasets, batch_size=1, max_seq_length=2048, quantization="8bit", 
                  ds_config=None, log_dir="scaling_logs"):
    """Evaluate a model on multiple datasets"""
    
    # Create log directory
    os.makedirs(log_dir, exist_ok=True)
    
    # Load tokenizer 
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    tokenizer.pad_token = tokenizer.eos_token
    
    # Load model with appropriate settings
    load_in_8bit = (quantization == "8bit")
    load_in_4bit = (quantization == "4bit")

    if load_in_8bit:
        quantization_config = BitsAndBytesConfig(load_in_8bit=True)
    elif load_in_4bit:
        quantization_config = BitsAndBytesConfig(load_in_4bit=True)
    else:
        quantization_config = None
    
    print(f"Loading model {model_id} with {quantization} quantization...")
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        device_map="auto",
        quantization_config=quantization_config,
        torch_dtype=torch.bfloat16 if not (load_in_8bit or load_in_4bit) else None,
        cache_dir='./cache'
    )
    
    results = {}
    
    # Evaluate on each dataset
    for dataset_name, texts in datasets.items():
        print(f"Evaluating {model_id} on {dataset_name} dataset...")
        
        # Create dataset and dataloader
        dataset = TextDataset(texts, tokenizer, max_seq_length)
        dataloader = DataLoader(
            dataset, 
            batch_size=batch_size, 
            collate_fn=collate_fn
        )
        
        # Evaluation loop
        model.eval()
        total_loss = 0
        total_tokens = 0
        
        with torch.no_grad():
            for batch in tqdm(dataloader, desc=f"Evaluating on {dataset_name}"):
                
                outputs = model(
                    input_ids=batch["input_ids"].to(model.device),
                    attention_mask=batch["attention_mask"].to(model.device),
                    labels=batch["labels"].to(model.device)
                )
                
                # Calculate loss
                loss = outputs.loss
                num_tokens = (batch["labels"] != -100).sum().item()
                
                # Accumulate
                total_loss += loss.item() * num_tokens
                total_tokens += num_tokens
        
        # Calculate perplexity
        avg_loss = total_loss / total_tokens
        perplexity = torch.exp(torch.tensor(avg_loss)).item()
        
        # Store results
        results[dataset_name] = {
            "loss": avg_loss,
            "perplexity": perplexity
        }
        
        print(f"Results for {model_id} on {dataset_name}:")
        print(f"  Loss: {avg_loss:.4f}")
        print(f"  Perplexity: {perplexity:.4f}")
    
    # Save results
    result_file = os.path.join(log_dir, f"{model_id.replace('/', '_')}_results.json")
    with open(result_file, 'w') as f:
        json.dump({
            "model_id": model_id,
            "results": results
        }, f, indent=2)
    
    return results

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate language models for scaling law analysis")
    
    # Add seed argument for reproducibility
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for reproducible sampling")
    
    parser.add_argument("--models", nargs="+", required=True, 
                        help="List of model IDs to evaluate")
    parser.add_argument("--datasets", nargs="+", default=["wikitext", "c4", "pile"],
                        help="Datasets to evaluate on")
    parser.add_argument("--samples", type=int, default=1000, 
                        help="Number of samples per dataset")
    parser.add_argument("--batch_size", type=int, default=1, 
                        help="Batch size for evaluation")
    parser.add_argument("--max_seq_length", type=int, default=2048,
                        help="Maximum sequence length for evaluation")
    parser.add_argument("--quantization", choices=["none", "8bit", "4bit"], default="8bit",
                        help="Quantization level for model loading")
    parser.add_argument("--log_dir", default="scaling_logs",
                        help="Directory to save evaluation results")
    parser.add_argument("--ds_config", default=None,
                        help="DeepSpeed config file path (optional)")
                        
    args = parser.parse_args()
    
    # Sample from datasets
    print("Sampling from datasets...")
    datasets = {}
    for dataset_name in args.datasets:
        print(f"  Sampling from {dataset_name}...")
        datasets[dataset_name] = sample_dataset(
            dataset_name, 
            num_samples=args.samples,
            max_seq_length=args.max_seq_length,
            seed=args.seed
        )
        print(f"  Sampled {len(datasets[dataset_name])} examples from {dataset_name}")
    
    # Load DeepSpeed config if provided
    ds_config = None
    if args.ds_config and os.path.exists(args.ds_config):
        with open(args.ds_config, 'r') as f:
            ds_config = json.load(f)
    
    # Evaluate each model
    for model_id in args.models:
        print(f"\nEvaluating model: {model_id}")
        try:
            evaluate_model(
                model_id,
                datasets,
                batch_size=args.batch_size,
                max_seq_length=args.max_seq_length,
                quantization=args.quantization,
                ds_config=ds_config,
                log_dir=args.log_dir
            )
        except Exception as e:
            print(f"Error evaluating {model_id}: {e}")