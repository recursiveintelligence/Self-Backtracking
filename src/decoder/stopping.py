from typing import List
import torch
from transformers import StoppingCriteria, StoppingCriteriaList


class SuffixStopOnSubsequence(StoppingCriteria):
    """
    Stops generation when any of the target token sequences appears as a suffix of
    the current sequence. Works per-example in batch.
    """

    def __init__(self, stop_token_seqs: List[torch.Tensor]):
        self.stop_token_seqs = [s.to(torch.long).view(-1) for s in stop_token_seqs if s.numel() > 0]

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs) -> bool:
        if not self.stop_token_seqs:
            return False
        seqlen = input_ids.size(1)
        for s in self.stop_token_seqs:
            t = s.numel()
            if t == 0 or t > seqlen:
                continue
            # Check per batch row if suffix matches; early exit if any match
            if torch.all(input_ids[:, -t:] == s, dim=-1).any():
                return True
        return False


def make_stop_criteria(tokenizer, phrases: List[str]) -> StoppingCriteriaList:
    toks = []
    for p in phrases:
        ids = tokenizer.encode(p, add_special_tokens=False)
        toks.append(torch.tensor(ids, dtype=torch.long))
    return StoppingCriteriaList([SuffixStopOnSubsequence(toks)])

