from abc import ABC, abstractmethod
import torch
import time
import copy
import numpy as np
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch.nn.functional as F
class BaseDecoder(ABC):
    """
    Abstract base class for decoders, defining a common interface for implementing custom decoding algorithms.
    """
    def __init__(self, model, tokenizer,*args,**kwargs):
        """
        Initialize the decoder
        :param model: Language model
        :param tokenizer: Tokenizer
        """
        self.model = model
        self.tokenizer = tokenizer

    @abstractmethod
    def decode(self, input_ids, max_length,*args,**kwargs):
        """
        Abstract method to perform decoding
        :param input_ids: Input token IDs
        :param max_length: Maximum decoding length
        :return: Decoded token IDs
        """
        pass
    
class GreedyDecoder(BaseDecoder):
    def decode(self, input_ids, max_length,*args,**kwargs):
        generated_ids = input_ids.cuda()
        self.model.eval()
        with torch.no_grad():
            for _ in range(max_length):
                outputs = self.model(input_ids=generated_ids)
                logits = outputs.logits[:, -1, :]
                next_token = torch.argmax(logits, dim=-1)
                generated_ids = torch.cat([generated_ids, next_token.unsqueeze(-1)], dim=-1)
                if next_token.item() == self.tokenizer.eos_token_id:
                    break
        return generated_ids
    

    
    
class SelfBackTrackingDecoder(BaseDecoder):
    def __init__(self, model, tokenizer, *args, **kwargs):
        super().__init__(model, tokenizer, *args, **kwargs)
    def agg(self):
        if not self.candidate_outputs:
            return []
        else:
            # print("origin length: ", len(self.candidate_outputs))
            output_dict = {}  # Dictionary to store {output_text: (count, total_score, candidate)}
            
            for candidate in self.candidate_outputs:
                only_output_text = candidate[1].split("###Response:\n")[1].split("Goal Reached!")[0]
                if only_output_text not in output_dict:
                    output_dict[only_output_text] = [1, candidate[2], candidate]
                else:
                    # Accumulate count and score
                    output_dict[only_output_text][0] += 1
                    output_dict[only_output_text][1] += candidate[2]
            
            new_candidate_outputs=[]
            for only_output_text in output_dict:
                # avg_score = output_dict[only_output_text][1] / output_dict[only_output_text][0]
                sum_score = output_dict[only_output_text][1]
                new_candidate_outputs.append((output_dict[only_output_text][2][0],output_dict[only_output_text][2][1],sum_score))
            
            # print("after deduplication: ", len(new_candidate_outputs))
            return new_candidate_outputs
    def backtrack(self, cur_input_ids):
        
        cur_text=self.tokenizer.decode(cur_input_ids, skip_special_tokens=True)
        last_index = cur_text.rfind('\n')
        second_last_index = cur_text.rfind('\n', 0, last_index)
        before_second_last_input_ids = self.tokenizer(cur_text[:second_last_index+1],return_tensors="pt").input_ids
        if before_second_last_input_ids.shape[1]<=self.init_input_ids.shape[1]:
            return self.init_input_ids
        else:
            return before_second_last_input_ids
        
    def decode(self, input_ids, max_length,k=8,backtrack_times=2,temperature=0.7,*args,**kwargs):
        self.k=k
        self.n=int(np.sqrt(k))
        self.temperature=temperature
        self.init_input_ids=input_ids
        self.model.eval()
        self.candidate_outputs=[]
        self.visited_state=[self.tokenizer.decode(input_ids[0], skip_special_tokens=True)]
        next_input_ids_list=[input_ids]
        for t in range(backtrack_times):
            next_input_ids_list_new=[]
            for generated_ids in next_input_ids_list:
                # print(generated_ids)
                with torch.no_grad():
                    # print(self.tokenizer.decode(generated_ids[0]))
                    generated_ids=generated_ids.cuda()
                    try:
                        outputs = self.model.generate(
                            tokenizer=self.tokenizer,
                            attention_mask=torch.ones_like(generated_ids).cuda(),
                            pad_token_id=self.tokenizer.eos_token_id,
                            input_ids=generated_ids,
                            max_length=max_length,            
                            num_return_sequences=self.k,   
                            do_sample=True,                          
                            temperature=self.temperature,             
                            num_beams=self.k,
                            # early_stopping=True,
                            output_scores=True,
                            return_dict_in_generate=True,
                            # stop_strings=['<backtrack>','Goal Reached!']
                        )
                    except Exception as e:
                        print(e)
                        continue
                        
                # scores = outputs.sequences_scores
                with torch.no_grad():
                    logits_batch = self.model(outputs.sequences).logits
                # print(logits_batch[0][-1])
                # log_logits_batch = F.log_softmax(logits_batch, dim=-1)
                    log_probs = F.log_softmax(logits_batch, dim=-1)
                    # Shift input_ids and logits for next-token prediction
                    shift_logits = log_probs[:, :-1, :].contiguous()
                    shift_labels = outputs.sequences[:, 1:].contiguous()
                    
                    # Create mask to exclude init input tokens
                    input_length = input_ids.shape[1]
                    mask = torch.ones_like(shift_labels, dtype=torch.bool)
                    mask[:, :input_length-1] = False  # Mask out the input tokens (-1 due to shift)
                    
                    # Gather the log probabilities of the actual next tokens
                    gathered_log_probs = torch.gather(
                        shift_logits, 
                        dim=2, 
                        index=shift_labels.unsqueeze(-1)
                    ).squeeze(-1)  # [batch_size, seq_len-1]
                    
                    # Apply mask to gathered_log_probs
                    gathered_log_probs = gathered_log_probs * mask
                    
                    # Calculate sequence lengths (excluding input and padding)
                    seq_lengths = mask.sum(dim=1)
                    
                    # Calculate NLL for each sequence in batch (only for generated part)
                scores = gathered_log_probs.sum(dim=1) / seq_lengths     
                      
                cur_text=self.tokenizer.batch_decode(outputs.sequences, skip_special_tokens=False)
                # print(self.k)
                # print(cur_text)
                for i in range(self.k):
                    if 'Goal Reached!' in cur_text[i]:
                        
                        self.candidate_outputs.append((outputs.sequences[i].unsqueeze(0),cur_text[i],scores[i]))
                    if len(next_input_ids_list_new)<self.n and '<backtrack>' in cur_text[i]:
                        next_state=self.backtrack(outputs.sequences[i])
                        next_state_text=self.tokenizer.decode(next_state[0], skip_special_tokens=True)
                        if not next_state_text in self.visited_state:
                            next_input_ids_list_new.append(next_state)
                            self.visited_state.append(next_state_text)
                    
                
            next_input_ids_list=next_input_ids_list_new
            # print(len(self.candidate_outputs))
        
        self.candidate_outputs=self.agg()
        if self.candidate_outputs:
            # print(self.candidate_outputs)
            best_sequence = max(self.candidate_outputs, key=lambda x: x[2])[0]
            return best_sequence
        else:
            try:
                return outputs.sequences[0].unsqueeze(0)
            except:
                print("no output")
                return None
    
    
    