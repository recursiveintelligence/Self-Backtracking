import os
import json
import argparse
import random

import torch
from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    Trainer,
    TrainingArguments,
    DataCollatorForSeq2Seq
)
from datasets import load_dataset, DatasetDict
from decoder import DECODER_DICT
from eval_search import eval_ll
import warnings
try:
    import wandb
    _WANDB_AVAILABLE = True
except Exception:
    wandb = None  # type: ignore
    _WANDB_AVAILABLE = False
os.environ["TOKENIZERS_PARALLELISM"] = "false"

def parse_args():
    parser = argparse.ArgumentParser()
    # Evaluation related arguments
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--num", type=int, default=5000)
    parser.add_argument("--offset", type=int, default=0)
    parser.add_argument("--data", type=str, default="val")
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--ctx", type=int, default=128)
    parser.add_argument("--decoder", type=str, default='self_backtrack')
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--b", type=int, default=1)
    parser.add_argument("--n", type=int, default=16)
    parser.add_argument("--past_model", type=str, default="yangxw/Llama-3.2-1B-countdown-backtrack")
    # Training related arguments
    parser.add_argument("--output_dir", type=str, default="results_self_improvement")
    parser.add_argument("--num_train_epochs", type=int, default=3)
    parser.add_argument("--learning_rate", type=float, default=1e-5)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1)
    parser.add_argument("--eval_steps", type=int, default=500)
    parser.add_argument("--save_steps", type=int, default=500)
    parser.add_argument("--warmup_steps", type=int, default=0)
    parser.add_argument("--wandb", action="store_true")
    parser.add_argument("--times", type=int, default=3)
    
    return parser.parse_args()

def get_training_examples(index,model, tokenizer, data, args):
    model.eval()
    model=model.cuda()
    """Evaluate model and collect successful examples for training"""
    test_prompts = [
        "###Question: numbers "
        + str(sample['nums'])
        + "->target "
        + str(sample['target'])
        + "\n###Response:\n"             
        for sample in data
    ]
    outputs, accuracy = eval_ll(
        args,
        model, 
        tokenizer, 
        test_prompts, 
        batch_size=1, 
        context_len=args.ctx
    )
    
    print(f"Current accuracy: {accuracy:.4f}")
    
    training_data = {
        "nums": [],
        "target": [],
        "search_path": []
    }
    
    for i, output in enumerate(outputs):
        if output["is_correct"]:
            training_data["nums"].append(data[i]["nums"])
            training_data["target"].append(data[i]["target"])
            training_data["search_path"].append(output["model_output"].split('<|begin_of_text|>')[1])
    print(len(training_data))
    # Save evaluation results
    results_file = os.path.join(args.output_dir, f"eval_results_{index}_{args.decoder}_{args.n}_{args.b}.json")
    os.makedirs(args.output_dir, exist_ok=True)
    with open(results_file, "w") as f:
        json.dump({
            'outputs': outputs,
            'accuracy': accuracy,
            'num_successful_examples': len(training_data["nums"])
        }, f, indent=4)
            
    return training_data, accuracy

def train_on_successful_examples(model, tokenizer, training_data, args):
    """Train model on successful examples"""
    model.train()
    # Convert to HuggingFace dataset
    dataset = Dataset.from_dict(training_data)
    
    def tokenize(element):
        text = [
            e
            + tokenizer.eos_token
            for e in element["search_path"]
        ]
        outputs = tokenizer(
            text,
            truncation=True,
            max_length=args.ctx,
            return_tensors="pt",
            padding=True,
        )
        
        labels = outputs["input_ids"].clone()
        
        # Mask the input part in labels
        for i, txt in enumerate(text):
            response_start = txt.find("###Response:\n") + len("###Response:\n")
            pre_response_tokens = tokenizer(txt[:response_start])["input_ids"]
            labels[i, :len(pre_response_tokens)] = -100
            
        return {"input_ids": outputs["input_ids"], "labels": labels}
    
    tokenized_dataset = dataset.map(
        tokenize, 
        batched=True, 
        remove_columns=dataset.column_names
    )
    
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        num_train_epochs=args.num_train_epochs,
        warmup_steps=args.warmup_steps,
        save_strategy="no",
        logging_steps=5,
        report_to="wandb" if args.wandb else "none",
        bf16=True,
    )
    
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset,
        data_collator=DataCollatorForSeq2Seq(
            tokenizer, 
            pad_to_multiple_of=8, 
            return_tensors="pt", 
            padding=True
        ),
    )
    
    trainer.train()
    

def main():
    args = parse_args()
    args.output_dir=args.output_dir+'_'+args.data
    # Set random seed
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    
    # Initialize wandb if requested
    if args.wandb and not _WANDB_AVAILABLE:
        warnings.warn("wandb is not installed. Disable --wandb or `pip install wandb`. Disabling wandb logging.")
    if args.wandb and _WANDB_AVAILABLE:
        wandb.init(project="self-improvement-training")
    
    # Load evaluation data
    # Robust dataset load (reuse eval's approach)
    def _load_ds():
        try:
            return load_dataset("yangxw/countdown-backtracking")
        except Exception as e:
            print("Primary dataset load failed, trying fallback via Hub raw files. Error:", e)
            try:
                from huggingface_hub import list_repo_files, hf_hub_download
                repo_id = "yangxw/countdown-backtracking"
                files = list_repo_files(repo_id, repo_type="dataset")
                def find(keys, exts):
                    ks = tuple(k.lower() for k in keys)
                    es = tuple(ext.lower() for ext in exts)
                    return [f for f in files if any(k in f.lower() for k in ks) and f.lower().endswith(es)]
                train_cands = find(["train"], [".jsonl", ".json", ".parquet", ".arrow"]) 
                valid_cands = find(["validation", "valid", "val"], [".jsonl", ".json", ".parquet", ".arrow"]) 
                if not train_cands:
                    raise RuntimeError("No train file found in dataset repo.")
                def pick(cands):
                    for ext in [".jsonl", ".json", ".parquet", ".arrow"]:
                        for c in cands:
                            if c.lower().endswith(ext):
                                return c, ext
                    return cands[0], ".json"
                train_file, train_ext = pick(train_cands)
                local_train = hf_hub_download(repo_id=repo_id, filename=train_file, repo_type="dataset")
                data_files = {"train": local_train}
                if valid_cands:
                    valid_file, _ = pick(valid_cands)
                    local_valid = hf_hub_download(repo_id=repo_id, filename=valid_file, repo_type="dataset")
                    data_files["validation"] = local_valid
                builder = "json" if train_ext in (".jsonl", ".json") else "parquet"
                return load_dataset(builder, data_files=data_files)
            except Exception as e2:
                raise RuntimeError("Failed to load dataset via both builder and raw file fallback.") from e2

    dataset = _load_ds()

    if 'validation' in dataset:
        val_split = dataset['validation'].train_test_split(test_size=0.5, shuffle=False)
        hf_datasets = DatasetDict({
            'train': dataset['train'],
            'val': val_split['train'],
            'val_new': val_split['test'],
        })
    else:
        split = dataset['train'].train_test_split(test_size=0.01, shuffle=False)
        val_split = split['test'].train_test_split(test_size=0.5, shuffle=False)
        hf_datasets = DatasetDict({
            'train': split['train'],
            'val': val_split['train'],
            'val_new': val_split['test'],
        })
    if args.data == 'val':
        data = hf_datasets['val']
    elif args.data == 'val_new':
        data = hf_datasets['val_new']
    else:
        raise ValueError(f"Unknown data: {args.data}")

    eval_data = data.select(range(args.offset, args.num))
    
    for t in range(args.times):
        # Load model and tokenizer
        if t==0:
            model_save_path_past = args.past_model
        else:
            model_save_path_past = os.path.join(args.output_dir, f"model_iteration_{t}")
        model = AutoModelForCausalLM.from_pretrained(model_save_path_past, torch_dtype=torch.bfloat16)
        tokenizer = AutoTokenizer.from_pretrained(model_save_path_past)
        tokenizer.pad_token = tokenizer.eos_token
        
        
        print(f"Evaluating model to collect successful examples for iteration {t}...")
        training_data, accuracy = get_training_examples(t, model, tokenizer, eval_data, args)
        print(f"Collected {len(training_data['nums'])} successful examples for training")
        # exit(0)
        if args.wandb and _WANDB_AVAILABLE:
            wandb.log({
                f"initial_accuracy_{t}": accuracy,
                f"num_training_examples_{t}": len(training_data["nums"])
            })
        
        # Train on successful examples
        print(f"Starting training on successful examples for iteration {t}...")
        train_on_successful_examples(model, tokenizer, training_data, args)
        
        # Save the model with the iteration index
        model_save_path = os.path.join(args.output_dir, f"model_iteration_{t+1}")
        model.save_pretrained(model_save_path)
        tokenizer.save_pretrained(model_save_path)
    
    print(f"Evaluating model to collect successful examples for iteration {args.times}...")
    training_data, accuracy = get_training_examples(args.times, model, tokenizer, eval_data, args)
    if args.wandb and _WANDB_AVAILABLE:
        wandb.finish()

if __name__ == "__main__":
    main()
