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

from models import TextEncoder
from utils.util_algo import metrics_old, metrics_new

__all__ = ["CoOpPrompter", "CoOpLearner"]

_tokenizer = _Tokenizer()

class CoOpPrompter(nn.Module):
    def __init__(self, cfg, log, device, clip_model):
        super().__init__()
        
        self.clip_model = clip_model
        self.dtype = clip_model.dtype
        ctx_dim = clip_model.ln_final.weight.shape[0]
        
        self.n_cls = 0
        ctx_init = None
        # ctx_init = 'a photo of a' # caltech101
        n_ctx = int(cfg["method"]["n_ctx"])
        self.n_ctx = n_ctx
        self.device = device

        if ctx_init:
            # use given words to initialize context vectors
            ctx_init = ctx_init.replace("_", " ")
            n_ctx = len(ctx_init.split(" "))
            prompt = clip.tokenize(ctx_init).to(device)
            with torch.no_grad():
                embedding = clip_model.token_embedding(prompt).type(self.dtype)
            ctx_vectors = embedding[0, 1 : 1 + n_ctx, :]
            prompt_prefix = ctx_init       
            self.n_ctx = n_ctx
        else:
            # random initialization
            ctx_vectors = torch.empty(n_ctx, ctx_dim, dtype=self.dtype).to(device)
            nn.init.normal_(ctx_vectors, std=0.02)
            prompt_prefix = " ".join(["X"] * n_ctx)

        log.info(f'Initial context: "{prompt_prefix}"')
        log.info(f"Number of context words (tokens): {n_ctx}")

        self.ctx = nn.Parameter(ctx_vectors)
        self.prompt_prefix = prompt_prefix
        
    def get_prefix_suffix_token(self, classnames):
        clip_model = self.clip_model
        self.n_cls = len(classnames)
        prompt_prefix = self.prompt_prefix
        classnames = [name.replace("_", " ") for name in classnames]
        name_lens = [len(_tokenizer.encode(name)) for name in classnames]
        prompts = [prompt_prefix + " " + name + "." for name in classnames]
        tokenized_prompts = torch.cat([clip.tokenize(p) for p in prompts]).to(self.device)  # (n_cls, n_tkn)
        with torch.no_grad():
            embedding = clip_model.token_embedding(tokenized_prompts).type(self.dtype)

        self.register_buffer("token_prefix", embedding[:, :1, :])  # SOS
        self.register_buffer("token_suffix", embedding[:, 1 + self.n_ctx :, :])  # CLS, EOS
        
        self.tokenized_prompts = tokenized_prompts  # torch.Tensor
        self.name_lens = name_lens

    def forward(self):
        ctx = self.ctx
        if ctx.dim() == 2:
            ctx = ctx.unsqueeze(0).expand(self.n_cls, -1, -1)

        prefix = self.token_prefix
        suffix = self.token_suffix

        prompts = torch.cat(
            [
                prefix,  # (n_cls, 1, dim)
                ctx,     # (n_cls, n_ctx, dim)
                suffix,  # (n_cls, *, dim)
            ],
            dim=1,
        )

        return prompts

class CoOpLearner(object):
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
        self.prompter = CoOpPrompter(cfg, logger, device, clip_model).to(device)

        # Optimizer
        self.optimizer = torch.optim.SGD( self.prompter.parameters(), lr=cfg["method"]['lr'] )
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR( self.optimizer,  cfg["method"]['train_epoch'])
        self.record = []

    def freeze_encoder(self, logger):
        logger.info("Turning off gradients in both the image and the text encoder")

        for param in self.image_encoder.parameters():
            param.requires_grad = False

        for param in self.text_encoder.parameters():
            param.requires_grad = False

    def train(self, cfg, logger, base_loader, valid_loader):
        train_loader = base_loader
        
        # Freeze clip parameter
        self.freeze_encoder(logger)
        # Get config for method
        method_cfg = cfg["method"]
        
        last_time = time.time()
        for epoch in range(method_cfg['train_epoch']):
            self.prompter.get_prefix_suffix_token(train_loader.dataset.classnames)
            loss = self.train_epoch(cfg, logger, train_loader)
            
            if (epoch + 1) % method_cfg["print_epoch"] == 0 or (epoch + 1) == method_cfg['train_epoch']:
                self.prompter.get_prefix_suffix_token(valid_loader.dataset.classnames)
                predicts, targets, classnames = self.evaluate(cfg, logger, valid_loader)
                self.save_prediction((
                        epoch, predicts, targets, 
                        train_loader.dataset.classnames, 
                        valid_loader.dataset.classnames, 
                        classnames
                    ),
                    f"{epoch+1}"
                )
                self.save_model(f"{epoch+1}")

                base_correct, base_total, new_correct, new_total, correct, total = metrics_old(predicts, targets, classnames, train_loader.dataset.classnames, valid_loader.dataset.classnames)
                base_correct_new, base_total_new, new_correct_new, new_total_new, _, _ = metrics_new(predicts, targets, classnames, train_loader.dataset.classnames, valid_loader.dataset.classnames)
                logger.info("Epoch:[{:3d}/{:3d}]({:.2f}s) Loss:{:.2f} Base:[{:4d}/{:4d}]={:.2f}% New:[{:4d}/{:4d}]={:.2f}% ComBase:[{:4d}/{:4d}]={:.2f}% ComNew:[{:4d}/{:4d}]={:.2f}% All:[{:4d}/{:4d}]={:.2f}%".format( 
                        epoch, method_cfg['train_epoch'], time.time() - last_time, loss,
                        base_correct, base_total, base_correct * 100.0 / base_total, 
                        new_correct, new_total, new_correct * 100.0 / new_total, 
                        base_correct_new, base_total_new, base_correct_new * 100.0 / base_total_new, 
                        new_correct_new, new_total_new, new_correct_new * 100.0 / new_total_new,
                        correct, total, correct * 100.0 / total
                    )
                )
                
                last_time = time.time()
    
    def train_epoch(self, cfg, logger, train_loader):
        '''image_encoder -> open set detector -> close set & open set
            fine-tuned prompt with close set and the other with open set
        '''
        tokenized_prompts = self.prompter.tokenized_prompts
        avg_loss = []
        for idx, (images, target, _) in enumerate(train_loader):
            images, target = images.type(self.dtype).to(self.device), target.to(self.device)
            image_features = self.image_encoder(images)
            image_features /= image_features.norm(dim=-1, keepdim=True)
            prompts = self.prompter()
            text_features = self.text_encoder(prompts, tokenized_prompts)
            text_features = text_features / text_features.norm(dim=-1, keepdim=True)
            logit_scale = self.logit_scale.exp()
            logits = logit_scale * image_features.float() @ text_features.T.float()
            loss = F.cross_entropy(logits, target.long())

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            self.scheduler.step()
            avg_loss.append(loss.item())

        return np.mean(avg_loss)

    def evaluate(self, cfg, logger, loader):
        """This function computes predictions on test data.
        :param data: Dataset object - test dataset
        """
        last_time = time.time()
        prompts = self.prompter()
        tokenized_prompts = self.prompter.tokenized_prompts
        text_features = self.text_encoder(prompts, tokenized_prompts)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)

        predicts, targets, classnames = [], [], []
        with torch.no_grad():
            for ids, (images, labels, cnames) in enumerate(loader):
                images = images.type(self.dtype).to(self.device)
                image_features = self.image_encoder(images)
                image_features = image_features / image_features.norm(dim=-1, keepdim=True)
                logit_scale = self.logit_scale.exp()
                logits = logit_scale * image_features @ text_features.t()
                
                predicts.append(logits.detach().cpu())
                targets.append(labels.cpu())
                classnames += list(cnames)
            predicts = torch.cat(predicts, 0).numpy()
            targets  = torch.cat(targets,  0).numpy()
        logger.info("Evaluate testing set {:.2f}S".format(time.time() - last_time))
        return predicts, targets, classnames
    
    def _remove_clip_model(self, state_dict):
        '''
        Remove clip model to save disk space
        '''
        new_state_dict = {}
        for key, value in state_dict.items():
            if not key.startswith("clip_model."):
                new_state_dict[key] = value
        return new_state_dict
    
    def save_model(self, name="last"):
        """
        Storage models to log folder
        """
        model_path = os.path.join(self.model_path, f"{name}.pth")
        last_model_path = os.path.join(self.model_path, "last.pth")
        state_dict = self._remove_clip_model(self.prompter.state_dict())
        torch.save(state_dict, model_path)
        shutil.copyfile(model_path, last_model_path)
        # shutil.move(model_path, last_model_path)

    def save_prediction(self, predicts, name="last"):
        """
        Storage predictions to log folder
        """
        pred_path = os.path.join(self.predict_path, f"{name}.pkl")
        last_pred_path = os.path.join(self.predict_path, "last.pkl")
        pickle.dump(predicts, file=open(pred_path, 'wb+'))
        shutil.copyfile(pred_path, last_pred_path)