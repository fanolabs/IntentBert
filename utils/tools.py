import torch
from torch.utils.data import TensorDataset
import numpy as np
from random import sample
import random
random.seed(0)

entail_label_map = {'entail':1, 'nonentail':0}

def getDomainName(name):
    if name=="auto_commute":
        return "auto  commute"
    elif name=="credit_cards":
        return "credit cards"
    elif name=="kitchen_dining":
        return "kitchen  dining"
    elif name=="small_talk":
        return "small talk"
    elif ' ' not in name:
        return name
    else:
        raise NotImplementedError("Not supported domain name %s"%(name))

def splitName(dom):
    domList = []
    for name in dom.split(','):
        domList.append(getDomainName(name))
    return domList

def makeTrainExamples(data:list, tokenizer, label=None, mode='unlabel'):
    """
    unlabel: simply pad data and then convert into tensor
    multi-class: pad data and compose tensor dataset with labels
    """
    if mode != "unlabel":
        assert label is not None, f"Label is provided for the required setting {mode}"
        if mode == "multi-class":
            examples = tokenizer.pad(data, padding='longest', return_tensors='pt')
            if not isinstance(label, torch.Tensor):
                label = torch.tensor(label)
            examples = TensorDataset(label,
                                     examples['input_ids'],
                                     examples['token_type_ids'],
                                     examples['attention_mask'])
        else:
            raise ValueError(f"Undefined setting {mode}")
    else:
        examples = tokenizer.pad(data, padding='longest', return_tensors='pt')
        examples = TensorDataset(examples['input_ids'],
                                 examples['token_type_ids'],
                                 examples['attention_mask'])
    return examples

def makeEvalExamples(supportX, supportY, queryX, queryY, tokenizer, mode='multi-class'):
    """
    multi-class: simply pad data
    """
    if mode == "multi-class":
        supportX = tokenizer.pad(supportX, padding='longest', return_tensors='pt')
        queryX = tokenizer.pad(queryX, padding='longest', return_tensors='pt')
    else:
        raise ValueError("Invalid mode %d."%(mode))
    return supportX, supportY, queryX, queryY

#https://github.com/huggingface/transformers/blob/master/src/transformers/data/data_collator.py#L70
def mask_tokens(inputs, tokenizer,\
    special_tokens_mask=None, mlm_probability=0.15):
        """
        Prepare masked tokens inputs/labels for masked language modeling: 80% MASK, 10% random, 10% original.
        """
        labels = inputs.clone()
        # We sample a few tokens in each sequence for MLM training (with probability `self.mlm_probability`)
        probability_matrix = torch.full(labels.shape, mlm_probability)
        if special_tokens_mask is None:
            special_tokens_mask = [
                tokenizer.get_special_tokens_mask(val, already_has_special_tokens=True) for val in labels.tolist()
            ]
            special_tokens_mask = torch.tensor(special_tokens_mask, dtype=torch.bool)
        else:
            special_tokens_mask = special_tokens_mask.bool()

        probability_matrix.masked_fill_(special_tokens_mask, value=0.0)
        probability_matrix[torch.where(inputs==0)] = 0.0
        masked_indices = torch.bernoulli(probability_matrix).bool()
        labels[~masked_indices] = -100  # We only compute loss on masked tokens

        # 80% of the time, we replace masked input tokens with tokenizer.mask_token ([MASK])
        indices_replaced = torch.bernoulli(torch.full(labels.shape, 0.8)).bool() & masked_indices
        inputs[indices_replaced] = tokenizer.convert_tokens_to_ids(tokenizer.mask_token)

        # 10% of the time, we replace masked input tokens with random word
        indices_random = torch.bernoulli(torch.full(labels.shape, 0.5)).bool() & masked_indices & ~indices_replaced
        random_words = torch.randint(len(tokenizer), labels.shape, dtype=torch.long)
        inputs[indices_random] = random_words[indices_random]

        # The rest of the time (10% of the time) we keep the masked input tokens unchanged
        return inputs, labels
