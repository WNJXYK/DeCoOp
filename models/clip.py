import os
import time
import yaml
import shutil
import pickle
import random
import argparse
from collections import OrderedDict
from tqdm import tqdm
import numpy as np

import torch
from torch.autograd import Variable
import torch.nn.functional as F
import torch.nn as nn
import torchvision.transforms as transforms

import clip
from clip.simple_tokenizer import SimpleTokenizer as _Tokenizer
from models import CoOpPrompter, CoOpLearner, TextEncoder

__all__ = ["ClipPrompter", "ClipLearner"]

_tokenizer = _Tokenizer()

class ClipPrompter(nn.Module):
    def __init__(self, cfg, log, device, clip_model):
        super().__init__()
        self.clip_model = clip_model
        self.dtype = clip_model.dtype
        self.device = device
        self.classtokens = None
        self.n_classes = 0
        self.clip_model.eval()
        if cfg["method"]["ctx_init"] == "auto":
            ctx_init = [cfg["dataset"]["ctx_init"]]
        else:
            ctx_init = cfg["method"]["ctx_init"].split(" ")
        self.prefixs  = []
        for prefix in ctx_init:
            prefix = prefix.replace("_", " ")
            self.prefixs += [prefix]
        for prefix in self.prefixs:
            log.info(f'Initial context: "{prefix}"')
        
    def get_prefix_suffix_token(self, classnames):
        clip_model = self.clip_model
        prefix_classnames = []
        for prefix in self.prefixs:
            prefix_classnames += [prefix.format(name.replace("_", " ")) for name in classnames]
        classtokens = clip.tokenize(prefix_classnames).to(self.device)
        self.classtokens = classtokens.detach()
        self.n_classes = len(classnames) 

    def forward(self, images):
        images = images.type(self.dtype).to(self.device)
        logits_per_image, logits_per_text = self.clip_model(images, self.classtokens)
        logits_per_image = logits_per_image.view(images.shape[0], -1, self.n_classes)
        logits_per_image = logits_per_image.mean(1)
        return logits_per_image

class ClipLearner(CoOpLearner):
    def __init__(self, args, cfg, logger, device, clip_model):
        # Logger
        self.model_path = cfg["log"]["model"]
        self.predict_path = cfg["log"]["prediction"]
        self.args = args
        
        # Computing Device
        self.device = device 

        # CLIP Model
        self.clip_model = clip_model 
        self.text_encoder = TextEncoder(self.clip_model)
        self.image_encoder = self.clip_model.visual
        self.logit_scale = self.clip_model.logit_scale
        self.dtype = self.clip_model.dtype

        # Prompter Learner
        self.prompter = ClipPrompter(cfg, logger, device, clip_model).to(device)
    
    def train_epoch(self, cfg, logger, train_loader):
        return 0

    def evaluate(self, cfg, logger, loader):
        """This function computes predictions on test data.
        :param data: Dataset object - test dataset
        """
        self.prompter.get_prefix_suffix_token(loader.dataset.classnames)

        predicts, targets, classnames = [], [], []
        with torch.no_grad():
            for ids, (images, labels, cnames) in enumerate(loader):
                images = images.type(self.dtype).to(self.device)
                logits = self.prompter(images)
                predicts.append(logits.detach().cpu())
                targets.append(labels.cpu())
                classnames += list(cnames)
            predicts = torch.cat(predicts, 0).numpy()
            targets  = torch.cat(targets,  0).numpy()
        return predicts, targets, classnames