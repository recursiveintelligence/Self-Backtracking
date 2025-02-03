import os
import json
import argparse

import tqdm
import re
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from decoder import *
from peft import PeftModel

def parse_input(s):
    numbers_match = re.search(r'numbers \[(.*?)\]', s)
    if numbers_match:
        numbers = eval(numbers_match.group(1))
    else:
        raise ValueError("Not found numbers")
    target_match = re.search(r'target (\d+)', s)
    if target_match:
        target = int(target_match.group(1))
    else:
        raise ValueError("Not found target")
    response_match = re.search(r'###Response:\n(.*?)(?:<backtrack>|Goal Reached!)', s, re.DOTALL)
    if response_match:
        response_steps = response_match.group(1).strip().split('\n')
    else:
        raise ValueError("Not found response")
    return numbers, target, response_steps

def validate_steps(numbers, target, response_steps):
    current_numbers = list(numbers) 
    current_result = None
    steps_valid = True
    reason=''
    for step in response_steps:
        match = re.match(r'(\d+)\s*([+\-*/])\s*(\d+)\s*=\s*(\d+)', step)
        if match:
            num1 = int(match.group(1))
            operator = match.group(2)
            num2 = int(match.group(3))
            result = int(match.group(4))
            
            if num1 not in current_numbers or num2 not in current_numbers:
                steps_valid = False
                reason='Unknown number in step: '+step
                break
            if operator == '+':
                current_result = num1 + num2
            elif operator == '-':
                current_result = num1 - num2
            elif operator == '*':
                current_result = num1 * num2
            elif operator == '/':
                if num2==0:
                    steps_valid = False
                    reason='Division by zero in step: '+step
                    break
                else:
                    current_result = num1 // num2    
            else:
                steps_valid = False
                reason='Unknown operator in step: '+step
                break
            if current_result != result:
                steps_valid = False
                reason='Incorrect result in step: '+step
                break

            current_numbers = [num for num in current_numbers if num != num1]
            current_numbers = [num for num in current_numbers if num != num2]
            current_numbers.append(current_result)
        else:
            steps_valid = False
            reason='Invalid step format: '+step
            break

    if steps_valid and current_result == target:
        return steps_valid,reason
    elif reason=='' and current_result != target:
        steps_valid = False
        reason=f'Not reached target: {result} not equal to {target}'
    return steps_valid,reason

def eval_ll(args,model, tokenizer, data, batch_size=128, context_len=1024, temperature=0.0, n=1):
    output_dict_concat = []
    correct=0
    for b in tqdm.trange(0, len(data), batch_size):      
        batch = data[b:min(b+batch_size, len(data))]
        output_texts = ["" for _ in range(len(batch))]
        tokenizer.padding_side = "left"
        inputs = tokenizer(batch, return_tensors="pt", padding=True).to("cuda")
        inputs = inputs['input_ids']

        if DECODER_DICT.get(args.decoder,None) is not None:
            decoder = DECODER_DICT[args.decoder](model, tokenizer)
        else:
            raise ValueError(f"Unknown decoder name: {args.decoder}")
        
        outputs = decoder.decode(input_ids=inputs, max_length=50,backtrack_times=args.backtrack_times,k=args.k)
        if outputs is None:
            output_texts=['None']
        else:
            output_texts = tokenizer.batch_decode(outputs, skip_special_tokens=False)
        
        for txt in output_texts:
            txt_dict = {}
            output_simple=''
            find_backtrack=False
            if not re.search(r'###Response:\n(.*?)(?:<backtrack>|Goal Reached!)', txt, re.DOTALL):
                output_simple=txt
                txt_dict = {'model_output': output_simple, 'is_backtrack': False,'failed_step':'Invalid output format','is_correct':False}
                output_dict_concat.append(txt_dict)
                continue
            back_index=txt.find("<backtrack>")
            reach_index=txt.find("Goal Reached!")
            if back_index == -1:
                if reach_index != -1:
                    find_backtrack=False
                    output_simple=txt[:reach_index + len("Goal Reached!")]
                
            elif reach_index == -1:
                find_backtrack=True
                output_simple=txt[:back_index + len("<backtrack>")]
            else:
                if back_index < reach_index:
                    find_backtrack=True
                    output_simple=txt[:back_index + len("<backtrack>")]
                else:
                    find_backtrack=False
                    output_simple=txt[:reach_index + len("Goal Reached!")]
            
            numbers, target, response_steps = parse_input(txt)
            is_valid,reasons = validate_steps(numbers, target, response_steps)
            txt_dict = {'model_output': output_simple, 'is_backtrack': find_backtrack,'failed_step':reasons,'is_correct':is_valid}
            if is_valid:
                correct+=1  
            output_dict_concat.append(txt_dict)
    return output_dict_concat,correct/len(data)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=4)
    parser.add_argument("--ckpt", type=str, help="path to checkpoint")
    parser.add_argument("-n", "--num",type=int, default=5000)
    parser.add_argument("-o", "--offset",type=int, default=0)
    parser.add_argument("--data_dir", type=str, default="../data_new/")
    parser.add_argument("-d", "--data",type=str, default="val_backtrack.json")
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--ctx", type=int, default=1024)
    parser.add_argument("--decoder", type=str, required=True, choices=DECODER_DICT.keys(), help='Name of the decoder to use')
    parser.add_argument("--backtrack_times", type=int, default=3)
    parser.add_argument("--k", type=int, default=8)
    parser.add_argument("--use_lora", action='store_true', help="Use LoRA for model loading")
    
    args = parser.parse_args()
    torch.manual_seed(args.seed)
    tokenizer = AutoTokenizer.from_pretrained(args.ckpt, padding_side='left')
    tokenizer.pad_token = tokenizer.eos_token
    if args.use_lora:
        from peft import PeftModel
        basemodel_name = "unsloth/Meta-Llama-3.1-1B"
        base_model = AutoModelForCausalLM.from_pretrained(basemodel_name, torch_dtype=torch.float16).cuda()
        base_model.resize_token_embeddings(len(tokenizer))
        # Load the LoRA adapter
        # adapter_path = os.path.join(args.ckpt, "adapter_model.safetensors")
        model = PeftModel.from_pretrained(base_model, args.ckpt)
    else:
        # model = AutoModelForCausalLM.from_pretrained(args.ckpt, torch_dtype=torch.bfloat16)
        model = AutoModelForCausalLM.from_pretrained(args.ckpt, torch_dtype=torch.bfloat16)

    model.eval()
    model.cuda()

    data_file = os.path.join(args.data_dir, args.data)

    with open(data_file, "r") as json_file:
        data = json.load(json_file)

    predictions = []
    tokenizer.padding_side = "left"
    test_prompts = ["###Question: numbers "
                    + str(sample['nums'])
                    + "->target "
                    + str(sample['target'])
                    + "\n###Response:\n"             
                    for sample in data[args.offset:args.num]
                ]

    len_nums = [len(sample['nums']) for sample in data[args.offset:args.num]]
    data_4 = [d for d, l in zip(test_prompts, len_nums) if l == 4]

    output,accuracy = eval_ll(args,model, tokenizer, test_prompts, batch_size=args.batch_size, context_len=args.ctx)
    len_pred_nums = [4 for _ in predictions]

    if args.decoder == 'topk' or args.decoder == 'oracle':
        results_file = os.path.join(args.ckpt, f"results_{(args.data[:-5]).replace('/','_')}_{args.offset}_{args.num}_{args.decoder}_{args.k}.json")
    elif args.decoder == 'recbon':
        results_file = os.path.join(args.ckpt, f"results_{(args.data[:-5]).replace('/','_')}_{args.offset}_{args.num}_{args.decoder}_{args.k}_{args.backtrack_times}.json")
    else:
        results_file = os.path.join(args.ckpt, f"results_{(args.data[:-5]).replace('/','_')}_{args.offset}_{args.num}_{args.decoder}.json")
    with open(results_file, "w") as f:
        json.dump({'output':output,'accuracy':accuracy}, f, indent=4)
if __name__ == "__main__":
    main()