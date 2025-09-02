import argparse
import json
import os
import random

import torch
from accelerate import Accelerator
from datasets import load_dataset,DatasetDict

from transformers import AutoTokenizer
from transformers import DataCollatorForSeq2Seq
from transformers import AutoModelForCausalLM
from transformers import Trainer
from transformers import TrainingArguments
from transformers import AutoConfig

import warnings
import os

try:
    import wandb  # optional
    _WANDB_AVAILABLE = True
except Exception:
    wandb = None  # type: ignore
    _WANDB_AVAILABLE = False

os.environ["TOKENIZERS_PARALLELISM"] = "false"

def main(args):
    # read config from a json config file
    with open(args.config, "r") as f:
        config = json.load(f)

    # set seeds
    random.seed(config["seed"])
    torch.manual_seed(config["seed"])

    # set up accelerator
    accelerator = Accelerator()

    use_wandb = bool(args.wandb) and _WANDB_AVAILABLE
    if args.wandb and not _WANDB_AVAILABLE and accelerator.is_main_process:
        warnings.warn(
            "wandb is not installed. Disable --wandb or install it via `pip install wandb`. Disabling wandb logging for this run.")
    if use_wandb and accelerator.is_main_process:
        wandb_kwargs = config.get("wandb", {"project": "", "entity": "", "dir": ""})
        wandb.init(
            project=wandb_kwargs["project"],
            # entity=wandb_kwargs["entity"],
            name=config["name"],
            config=config,
            dir=wandb_kwargs["dir"],
        )

    if not args.reset:

        model = AutoModelForCausalLM.from_pretrained("unsloth/Llama-3.2-1B")
        tokenizer = AutoTokenizer.from_pretrained("unsloth/Llama-3.2-1B")
        
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id =  tokenizer.eos_token_id
        tokenizer.padding_side = 'right'
        special_tokens_dict = {'additional_special_tokens': ["<backtrack>"]}
        tokenizer.add_special_tokens(special_tokens_dict)
        model.resize_token_embeddings(len(tokenizer))
    else:
        model = AutoModelForCausalLM.from_pretrained(
            args.ckpt,
            torch_dtype=torch.float16
        )
        tokenizer = AutoTokenizer.from_pretrained(args.ckpt)

    print(f"Number of parameters: {model.num_parameters()}")

    # load dataset
    dataset = load_dataset("yangxw/countdown-backtracking")

    val_split = dataset['validation'].train_test_split(test_size=0.5, shuffle=False)
    hf_datasets = DatasetDict({
        'train': dataset['train'],
        'val': val_split['train'],
        # 'val_new': val_split['test'],
    })

    
    random_indices = random.sample(range(len(hf_datasets["train"])), int(config["num_train"]))
    hf_datasets["train"] = hf_datasets["train"].select(random_indices)
    print('Dataset Length: ', len(hf_datasets["train"] ))
    context_length = config["context_length"]
    tokenizer.model_max_length = context_length

    def tokenize(element):
        text = [
            "###Question: numbers "
            + str(element["nums"][e])
            + "->target "
            + str(element["target"][e])
            + "\n###Response:\n" 
            + element["search_path"][e].strip()
            + tokenizer.eos_token
            for e in range(len(element["search_path"]))
        ]
        outputs = tokenizer(
            text,
            truncation=True,
            max_length=context_length,
            return_tensors="pt",
            padding=True,
        ) 
        
        # Copy input_ids to labels
        labels = outputs["input_ids"].clone()

        # Optionally, mask non-target parts in labels with -100
        for i, txt in enumerate(text):
            # Find the starting position of the "Response" part
            response_start = txt.find("###Response:\n") + len("###Response:\n")
            
            pre_response_tokens = tokenizer(txt[:response_start])["input_ids"]
                # Mask the non-response tokens in labels
            labels[i, :len(pre_response_tokens)] = -100
            if '<backtrack>' in txt:
                
                # find the second last \n
                last_index = txt.rfind('\n')
                second_last_index = txt.rfind('\n', 0, last_index)
                
                # text between second last \n and <backtrack>
                before_second_last_tokens = tokenizer(txt[:second_last_index+1])["input_ids"]
                between_backtrack_tokens = tokenizer(txt[:last_index+1])["input_ids"]
                
                labels[i, len(before_second_last_tokens):len(between_backtrack_tokens)] = -100

                
        res = {"input_ids": outputs["input_ids"], "labels": labels}
        return res

    # tokenize dataset for causal LM
    tokenized_datasets = hf_datasets.map(
        tokenize, batched=True, remove_columns=hf_datasets["train"].column_names
    )

    data_collator=DataCollatorForSeq2Seq(
            tokenizer, pad_to_multiple_of=8, return_tensors="pt", padding=True
        )

    # prepare training
    training_args = TrainingArguments(
        output_dir=config["output_dir"],
        per_device_train_batch_size=config["batch_size"],
        evaluation_strategy="steps",
        eval_steps=config["eval_steps"],
        logging_steps=config["log_steps"],
        gradient_accumulation_steps=config["gradient_accumulation_steps"],
        gradient_checkpointing=True,
        num_train_epochs=config["num_train_epochs"],
        weight_decay=config["weight_decay"],
        warmup_steps=config["warmup_steps"],
        lr_scheduler_type=config["lr_scheduler_type"],
        learning_rate=config["lr"],
        save_strategy="steps",
        save_total_limit=config["save_total_limit"],
        save_steps=config["save_steps"],
        seed=config["seed"],
        bf16=True,
        push_to_hub=False,
        report_to=("wandb" if use_wandb else "none"),
        run_name=config["name"],
        ddp_find_unused_parameters=False
    )

    trainer = Trainer(
        model=model,
        tokenizer=tokenizer,
        args=training_args,
        data_collator=data_collator,
        train_dataset=tokenized_datasets["train"],
        eval_dataset={
            "valid": tokenized_datasets["val"]
        },
    )
    # train
    if args.resume:
        trainer.train(resume_from_checkpoint=args.ckpt)
    else:
        trainer.train()
        output_dir = os.path.join(config["output_dir"],"final_model")
        trainer.save_model(output_dir)
        tokenizer.save_pretrained(output_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="../configs/sft.conf")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--ckpt", type=str, default=None)
    parser.add_argument("--reset", action="store_true")
    parser.add_argument("--wandb", action=argparse.BooleanOptionalAction, default=False)

    args = parser.parse_args()
    main(args)
