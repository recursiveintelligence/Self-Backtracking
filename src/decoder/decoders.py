from abc import ABC, abstractmethod
import torch
import time
import copy
import numpy as np
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch.nn.functional as F
from .stopping import make_stop_criteria
from collections import OrderedDict
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
        self._past_cache = OrderedDict()
        self._past_cache_cap = 8
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
        
    def decode(self, input_ids, max_length, b=1, n=32, temperature=0.7, *args, **kwargs):
        self.b=b
        self.n2=int(np.sqrt(b))
        self.n=n
        self.temperature=temperature
        self.init_input_ids=input_ids
        self.model.eval()
        self.candidate_outputs=[]
        self.visited_state=[self.tokenizer.decode(input_ids[0], skip_special_tokens=True)]
        next_input_ids_list=[input_ids]
        stop_criteria = make_stop_criteria(self.tokenizer, ["<backtrack>", "Goal Reached!"])

        def get_or_build_past(prefix_ids: torch.Tensor):
            key = tuple(prefix_ids[0].tolist())
            cached = self._past_cache.get(key, None)
            if cached is not None:
                # refresh LRU
                self._past_cache.move_to_end(key)
                return cached
            with torch.no_grad():
                attn = torch.ones_like(prefix_ids, device="cuda")
                out = self.model(
                    input_ids=prefix_ids.cuda(),
                    attention_mask=attn,
                    use_cache=True,
                    return_dict=True,
                )
                last_tok = prefix_ids[:, -1:].cuda()
                past = out.past_key_values
            self._past_cache[key] = (past, attn, last_tok)
            # Evict if over capacity
            if len(self._past_cache) > self._past_cache_cap:
                self._past_cache.popitem(last=False)
            return self._past_cache[key]
        for t in range(b+1):
            next_input_ids_list_new=[]
            for generated_ids in next_input_ids_list:
                # print(generated_ids)
                with torch.no_grad():
                    # print(self.tokenizer.decode(generated_ids[0]))
                    generated_ids=generated_ids.cuda()
                    try:
                        # Reuse KV cache for this prefix to avoid recomputing context
                        past, attn, last_tok = get_or_build_past(generated_ids)
                        outputs = self.model.generate(
                            attention_mask=attn,
                            pad_token_id=self.tokenizer.eos_token_id,
                            input_ids=last_tok,
                            past_key_values=past,
                            max_new_tokens=max_length,
                            num_return_sequences=self.n,
                            do_sample=True,
                            temperature=self.temperature,
                            num_beams=self.n,
                            output_scores=True,
                            return_dict_in_generate=True,
                            stopping_criteria=stop_criteria,
                        )
                    except Exception as e:
                        print(e)
                        continue
                # Use sequences_scores directly from generate (avoids extra forward)
                seq_scores = getattr(outputs, "sequences_scores", None)
                if seq_scores is None:
                    # Fallback to zeros if scores are not returned (should not happen with beams)
                    scores = torch.zeros(outputs.sequences.size(0), device=generated_ids.device)
                else:
                    scores = seq_scores

                cur_text=self.tokenizer.batch_decode(outputs.sequences, skip_special_tokens=False)
                for i in range(self.n):
                    if 'Goal Reached!' in cur_text[i]:
                        self.candidate_outputs.append((outputs.sequences[i].unsqueeze(0),cur_text[i],scores[i]))
                    if len(next_input_ids_list_new)<self.n2 and '<backtrack>' in cur_text[i]:
                        next_state=self.backtrack(outputs.sequences[i])
                        next_state_text=self.tokenizer.decode(next_state[0], skip_special_tokens=True)
                        if not next_state_text in self.visited_state:
                            next_input_ids_list_new.append(next_state)
                            self.visited_state.append(next_state_text)
                
                
            next_input_ids_list=next_input_ids_list_new

        
        self.candidate_outputs=self.agg()
        if self.candidate_outputs:
            best_sequence = max(self.candidate_outputs, key=lambda x: x[2])[0]
            return best_sequence
        else:
            try:
                return outputs.sequences[0].unsqueeze(0)
            except:
                print("no output")
                return None
    
    
    
