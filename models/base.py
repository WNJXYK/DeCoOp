import os
import random
import argparse
import yaml
from collections import OrderedDict
from tqdm import tqdm

import torch
import torch.nn.functional as F
import torch.nn as nn
import torchvision.transforms as transforms

from datasets import build_dataset
import clip
from clip.simple_tokenizer import SimpleTokenizer as _Tokenizer
from torch.autograd import Variable

__all__ = ["TextEncoder", "Clip_Prompt"]

_tokenizer = _Tokenizer()

class Clip_Prompt(nn.Module):
    def __init__(self, cfg, log, classnames, clip_model):
        super().__init__()
        self.token_embedding = clip_model.token_embedding
        self.dtype = clip_model.dtype
        self.n_cls = len(classnames)
        
        ctx_dim = clip_model.ln_final.weight.shape[0]
        ctx_init = 'a photo of a' # caltech101

        # use given words to initialize context vectors
        prompt_prefix = ctx_init.replace("_", " ")
        self.n_ctx = len(ctx_init.split(" "))
        classnames = [name.replace("_", " ") for name in classnames]
        name_lens = [len(_tokenizer.encode(name)) for name in classnames]
        prompts = [prompt_prefix + " " + name + "." for name in classnames]
        self.tokenized_prompts = torch.cat([clip.tokenize(p) for p in prompts]).cuda()  # (n_cls, n_tkn)

        log.info(f'Initial context: "{prompt_prefix}"')
        log.info(f"Number of context words (tokens): {self.n_ctx}")

    def forward(self, text=None):
        if text :
            # x : [batch_size, n_ctx, d_model]
            x = self.token_embedding(text).type(self.dtype)  
            return x
        
        else:
            return self.token_embedding(self.tokenized_prompts).type(self.dtype) 
    
class Task_Indicator(nn.Module):
    
    def __init__(self):
        super().__init__()

    def forward(self):
        pass

class TextEncoder(nn.Module):
    def __init__(self, clip_model):
        super().__init__()
        self.transformer = clip_model.transformer
        self.positional_embedding = clip_model.positional_embedding
        self.ln_final = clip_model.ln_final
        self.text_projection = clip_model.text_projection
        self.dtype = clip_model.dtype

    def forward(self, prompts, tokenized_prompts):
        # NOTE: token_embedding is customed in Clip_Prompt and Prompt_Learner
        x = prompts + self.positional_embedding.type(self.dtype)
        x = x.permute(1, 0, 2)  # NLD -> LND
        x = self.transformer(x)
        x = x.permute(1, 0, 2)  # LND -> NLD
        x = self.ln_final(x).float()

        x = x[torch.arange(x.shape[0]), tokenized_prompts.argmax(dim=-1)].type(self.dtype) @ self.text_projection

        return x







    
