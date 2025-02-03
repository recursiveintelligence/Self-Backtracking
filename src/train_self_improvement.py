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
from decoder import DECODER_DICT
from eval_search import eval_ll
import wandb
os.environ["TOKENIZERS_PARALLELISM"] = "false"

def parse_args():
    parser = argparse.ArgumentParser()
    # Evaluation related arguments
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--num", type=int, default=5000)
    parser.add_argument("--offset", type=int, default=0)
    parser.add_argument("--data_dir", type=str, default="../data/")
    parser.add_argument("--data", type=str, default="val_backtrack.json")
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--ctx", type=int, default=128)
    parser.add_argument("--decoder", type=str, default='self_backtrack')
    parser.add_argument("--backtrack_times", type=int, default=1)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--k", type=int, default=16)
    
    # Training related arguments
    parser.add_argument("--output_dir", type=str)
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
        for sample in data[args.offset:args.num]
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
        # if not output["is_backtrack"]:
            training_data["nums"].append(data[i]["nums"])
            training_data["target"].append(data[i]["target"])
            training_data["search_path"].append(output["model_output"].split('<|begin_of_text|>')[1])
    print(len(training_data))
    # Save evaluation results
    results_file = os.path.join(args.output_dir, f"eval_results_{index}_{args.decoder}_{args.k}.json")
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
    # Set random seed
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    
    # Initialize wandb if requested
    if args.wandb:
        wandb.init(project="self-improvement-training")
    
    # Load evaluation data
    with open(os.path.join(args.data_dir, args.data), "r") as f:
        eval_data = json.load(f)
    
    for t in range(args.times):
        # Load model and tokenizer
        if t==0:
            model_save_path_past = "your path"
        else:
            model_save_path_past = os.path.join(args.output_dir, f"model_iteration_{t}")
        model = AutoModelForCausalLM.from_pretrained(model_save_path_past, torch_dtype=torch.bfloat16)
        tokenizer = AutoTokenizer.from_pretrained(model_save_path_past)
        tokenizer.pad_token = tokenizer.eos_token
        
        
        if t == 0:
            training_data = {
                "nums": [],
                "target": [],
                "search_path": [],
            }
            
            path = "your_path.json".format(args.data.split(".")[0], args.k, args.backtrack_times)
            
            if os.path.exists(path):
                with open(path, "r") as f:
                    model_output_data = json.load(f)
                for i in range(len(model_output_data['output'])):
                    if model_output_data['output'][i]["is_correct"]:
                        training_data["nums"].append(eval_data[i]["nums"])
                        training_data["target"].append(eval_data[i]["target"])
                        training_data["search_path"].append(model_output_data['output'][i]["model_output"].split('<|begin_of_text|>')[1])
            else:
                # Get successful examples
                print(f"Evaluating model to collect successful examples for iteration {t}...")
                training_data, accuracy = get_training_examples(t, model, tokenizer, eval_data, args)
        else:
            # Get successful examples
            print(f"Evaluating model to collect successful examples for iteration {t}...")
            training_data, accuracy = get_training_examples(t, model, tokenizer, eval_data, args)
        print(f"Collected {len(training_data['nums'])} successful examples for training")
        # exit(0)
        if args.wandb:
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
    if args.wandb:
        wandb.finish()

if __name__ == "__main__":
    main()
