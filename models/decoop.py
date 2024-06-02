import os
import time
import yaml
import clip
import shutil
import pickle
import random
import argparse
from collections import OrderedDict
from tqdm import tqdm
import numpy as np
from sklearn import metrics

import torch
from torch.autograd import Variable
import torch.nn.functional as F
import torch.nn as nn
import torchvision.transforms as transforms
from skimage.filters import threshold_otsu
from clip.simple_tokenizer import SimpleTokenizer as _Tokenizer
# from models import TextEncoder, ClipPrompter, CoOpPrompter, DeCoOpLearnerV1, OODPrompter
from utils.util_algo import metrics_old, metrics_new
from datasets import WrapperDataset
from torch.utils.data import DataLoader

__all__ = ["DeCoOpLearner"]

_tokenizer = _Tokenizer()

#### Text Encoder ####
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

#### Out-of-Distribution Prompter ####
class OODPrompter(nn.Module):
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

#### CoOp Prompter ####
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

#### Clip Prompter #### 
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

#### DeCoOp Model ####
class DeCoOpLearner(object):
    def __init__(self, args, cfg, logger, device, clip_model, dataset):
        # Logger
        self.model_path = cfg["log"]["model"]
        self.predict_path = cfg["log"]["prediction"]
        self.K = cfg["method"]["K"]
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
        self.clip_prompter = ClipPrompter(cfg, logger, device, clip_model).to(device)
        self.coop_prompter = nn.ModuleList([
            CoOpPrompter(cfg, logger, device, clip_model)
            for i in range(self.K)
        ]).to(device)
        self.ood_prompter = nn.ModuleList([
            OODPrompter(cfg, logger, device, clip_model)
            for i in range(self.K)
        ]).to(device)

        # Optimizer
        self.ood_optimizer = torch.optim.SGD( self.ood_prompter.parameters(), lr=cfg["method"]['lr'] )
        # self.ood_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR( self.ood_optimizer,  cfg["method"]['train_ood_epoch'])
        self.iid_optimizer = torch.optim.SGD( self.coop_prompter.parameters(), lr=cfg["method"]['lr'] )
        self.iid_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR( self.iid_optimizer,  cfg["method"]['train_epoch'])
        self.record = []

        # Init OOD Detector
        self.n_classes = len(dataset.classnames)
        self.classnames = dataset.classnames
        self.n_divs = (self.n_classes * (self.K - 1) + self.K - 1) // self.K
        self.tokenized_prompts, self.ood_masks, self.ood_labels = [], [], []
        self.ood_classes = []
        self.ood_recoveries = []
        self.thresholds = []
        self.ood_classnames = []

        # Process mask, label and classname for OOD detector 
        logger.info("OOD Detector: {} classes in {} classes".format(self.n_divs, self.n_classes))
        for i in range(self.K):
            classname, ood_mask, ood_label= [], [], []
            ood_recovery = []
            for j in range(self.n_classes):
                if j % self.K != i:
                    ood_label.append(len(classname))
                    ood_mask.append(1)
                    ood_recovery.append(j)
                    classname.append(dataset.classnames[j])
                else:
                    ood_label.append(-1)
                    ood_mask.append(0)
            if len(classname) < self.n_divs:
                ood_label.append(len(classname))
                ood_mask.append(1)
                ood_recovery.append(i)
                classname.append(dataset.classnames[i])
            logger.info("OOD Detector #{}: {} classes in {} classes".format(i, len(classname), self.n_classes))
            self.ood_classes.append(set(classname))
            self.ood_classnames.append(classname)
            self.ood_masks.append(torch.tensor(ood_mask).long().to(self.device))
            self.ood_labels.append(torch.tensor(ood_label).long().to(self.device))
            self.ood_recoveries.append(torch.tensor(ood_recovery).long().to(self.device))
            self.ood_prompter[i].get_prefix_suffix_token(classname)
            self.tokenized_prompts.append(self.ood_prompter[i].tokenized_prompts)
        self.thresholds = []
        self.threshold  = 0
        
        # Setup OOD margin
        if cfg["method"]["ood_margin"] == "auto":
            energy_func = lambda v: -np.sum(np.array(v) * np.log(np.array(v)))
            self.ood_margin = energy_func([1 / self.n_divs] * self.n_divs) - energy_func([0.4 / (self.n_divs - 1)] * (self.n_divs - 1) + [0.6])
        else:
            self.ood_margin = float(cfg["method"]["ood_margin"])
        logger.info("OOD Margin: {} => {}".format(cfg["method"]["ood_margin"], self.ood_margin))
        
        # Evaluate Information
        self.evaluate_OOD_cache = False
        self.OOD_scores = None
        self.CLIP_logits = None
    
    def setup_ood_prompter(self, loader):
        classnames = loader.dataset.classnames
        self.tokenized_prompts = []
        for i in range(self.K):
            classname = [c for c in self.ood_classnames[i]]
            for c in classnames:
                if c not in self.ood_classnames[i]:
                    classname.append(c)
            self.ood_prompter[i].get_prefix_suffix_token(classname)
            self.tokenized_prompts.append(self.ood_prompter[i].tokenized_prompts)
        
    def recover_ood_prompter(self):
        self.tokenized_prompts = []
        for i in range(self.K):
            self.ood_prompter[i].get_prefix_suffix_token(self.ood_classnames[i])
            self.tokenized_prompts.append(self.ood_prompter[i].tokenized_prompts)
            
    def setup_coop_prompter(self, loader):
        n_classes = len(loader.dataset.classnames)
        classnames = loader.dataset.classnames
        coop_tokenized_prompts = []
        for i in range(self.K):
            self.coop_prompter[i].get_prefix_suffix_token(loader.dataset.classnames)
            coop_tokenized_prompts.append(self.coop_prompter[i].tokenized_prompts)
        mapto = {classnames[i]: i for i in range(n_classes)}

        coop_masks, coop_labels = [], []
        coop_indices = []
        for i in range(self.K):
            mask, label = [], []
            index = []
            for j in range(n_classes):
                if classnames[j] in self.ood_classes[i]:
                    label.append(j)
                    mask.append(True)
                    index.append(j)
                else:
                    label.append(-1)
                    mask.append(False)
            coop_masks.append(torch.tensor(mask).long().to(self.device))
            coop_labels.append(torch.tensor(label).long().to(self.device))
            coop_indices.append(torch.tensor(index).long().to(self.device))
        
        return coop_tokenized_prompts, coop_masks, coop_labels, coop_indices

    def freeze_encoder(self, logger):
        logger.info("Turning off gradients in both the image and the text encoder")

        for param in self.image_encoder.parameters():
            param.requires_grad = False

        for param in self.text_encoder.parameters():
            param.requires_grad = False
    
    def train(self, cfg, logger, base_loader, test_loader):
        self.freeze_encoder(logger)
        method_cfg = cfg["method"]
        train_loader = base_loader
        last_time = time.time()
        self.evaluate_OOD_cache = False
        
        # Stage 1: Training OOD Detector
        for epoch in range(method_cfg["train_ood_epoch"]):
            loss_id, loss_ood, engry_id, engry_ood = self.train_ood_epoch(cfg, logger, train_loader)

            if (epoch + 1) == method_cfg['train_ood_epoch']:
                _, thresholds, threshold = self.estimate_ood_epoch(cfg, logger, train_loader)
                classnames, scores, auroc = self.evaluate_ood(cfg, logger, test_loader, train_loader.dataset.classnames)
                message =  "Epoch:[{:3d}/{:3d}]({:.2f}s) IDLoss:{:.2f} OODLoss:{:.2f} Loss:{:.2f} "
                message += "IDEngry:{:.2f} OODEngry:{:.2f} "
                message += "AUROC:{:.2f}%"
                logger.info(message.format( 
                        epoch, method_cfg['train_ood_epoch'], time.time() - last_time, 
                        loss_id, loss_ood, loss_id + loss_ood,
                        engry_id, engry_ood,
                        auroc*100
                    )
                )
                
                last_time = time.time()

        # Stage 2: Training Classifier
        new_train_loader = self.rebuild_train_loader(cfg, logger, train_loader)
        for epoch in range(method_cfg['train_epoch']):
            loss = self.train_iid_epoch(cfg, logger, new_train_loader)

            # Compute metrics
            if (epoch + 1) == method_cfg['train_epoch']:
                predicts, targets, classnames, scores, auroc = self.evaluate(cfg, logger, test_loader, train_loader.dataset.classnames)
                base_correct, base_total, new_correct, new_total, correct, total = metrics_old(predicts, targets, classnames, train_loader.dataset.classnames, test_loader.dataset.classnames)
                base_acc = base_correct * 100.0 / base_total
                new_acc  = new_correct * 100.0 / new_total
                H_acc    = 2 * base_acc * new_acc / (base_acc + new_acc)
                message =  "Epoch:[{:3d}/{:3d}]({:.2f}s) ClsLoss:{:.2f} - "
                message += "H:{:.2f}% Acc:{:.2f}%"
                logger.info(message.format( 
                        epoch, method_cfg['train_epoch'], time.time() - last_time, loss,
                        H_acc, correct * 100.0 / total
                    )
                )
                last_time = time.time()
        
    def rebuild_train_loader(self, cfg, logger, train_loader):
        self.ood_prompter.eval()
        self.coop_prompter.eval()
        self.clip_prompter.eval()

        last_time = time.time()
        train_loader_noshuffle = DataLoader(dataset=train_loader.dataset, batch_size=cfg['method']['eval_batch_size'],  shuffle=False, num_workers=16)
        
        # Run for clip and score
        self.setup_ood_prompter(train_loader)
        self.clip_prompter.get_prefix_suffix_token(train_loader.dataset.classnames)
        
        with torch.no_grad():
            # Compute OOD text features
            ood_prompts = [self.ood_prompter[i]() for i in range(self.K)]
            ood_text_features = [self.text_encoder(ood_prompts[i], self.tokenized_prompts[i]).to(self.device) for i in range(self.K)]
            ood_text_features = [ood_text_features[i] / ood_text_features[i].norm(dim=-1, keepdim=True) for i in range(self.K)]
        

            clip, score = [], []
            for idx, (images, target, _) in enumerate(train_loader_noshuffle):
                # Get Image Features
                images, target = images.type(self.dtype).to(self.device), target.to(self.device)
                image_features = self.image_encoder(images)
                image_features /= image_features.norm(dim=-1, keepdim=True)

                logit_scale = self.logit_scale.exp()
                ood_logits  = [ logit_scale * image_features.float() @ ood_text_features[i].T.float() for i in range(self.K) ]
                clip_logits = self.clip_prompter(images).detach()
                ood_probas = torch.cat([
                    F.softmax(ood_logits[i], dim=1)[:, :self.n_divs].unsqueeze(-1)
                    for i in range(self.K)
                ], -1).detach()
                
                clip.append(clip_logits.cpu())
                score.append(ood_probas.cpu())
                
            clip = torch.cat(clip, 0)
            score = torch.cat(score, 0)
        wdataset = WrapperDataset(train_loader.dataset, clip, score)
        new_train_loader = DataLoader(dataset=wdataset,  batch_size=cfg['method']['train_batch_size'], shuffle=True,  num_workers=16)

        logger.info("Rebuild training set {:.2f}S".format(time.time() - last_time))
        return new_train_loader
        
    def estimate_ood_epoch(self, cfg, logger, train_loader):
        self.setup_ood_prompter(train_loader)
        with torch.no_grad():
            # Compute OOD text features
            ood_prompts = [self.ood_prompter[i]() for i in range(self.K)]
            ood_text_features = [self.text_encoder(ood_prompts[i], self.tokenized_prompts[i]).to(self.device) for i in range(self.K)]
            ood_text_features = [ood_text_features[i] / ood_text_features[i].norm(dim=-1, keepdim=True) for i in range(self.K)]
            
        scores = []
        with torch.no_grad():
             for ids, (images, labels, cnames) in enumerate(train_loader):
                images = images.type(self.dtype).to(self.device)
                image_features = self.image_encoder(images)
                image_features = image_features / image_features.norm(dim=-1, keepdim=True)
                logit_scale = self.logit_scale.exp()
                ood_logits  = [ logit_scale * image_features.float() @ ood_text_features[i].T.float() for i in range(self.K) ]
                ood_probas = torch.cat([
                    F.softmax(ood_logits[i], dim=1)[:, :self.n_divs].unsqueeze(-1)
                    for i in range(self.K)
                ], -1)
                ood_scores  = torch.max(ood_probas, 1)[0]
                scores.append(ood_probas.cpu())
        scores = torch.cat(scores, 0)
        scores = torch.max(scores, 1)[0]
        scores = torch.nan_to_num(scores, nan=0.0)
        scores = torch.clamp(scores, min=0, max=1)
        
        self.thresholds = [threshold_otsu(scores.numpy()[:, [i]]) for i in range(self.K)]
        # self.threshold = threshold_otsu(torch.max(scores, 1)[0].numpy())
        self.threshold = np.min(self.thresholds)
        return scores, self.thresholds, self.threshold
                
    
    def train_ood_epoch(self, cfg, logger, train_loader):
        '''image_encoder -> open set detector -> close set & open set
            fine-tuned prompt with close set and the other with open set
        '''
        self.ood_prompter.train()
        # Train OOD Detector
        self.recover_ood_prompter()
        avg_id_loss, avg_ood_loss = [], []
        avg_id_engry, avg_ood_engry = [], []

        for idx, (images, target, _) in enumerate(train_loader):
            with torch.no_grad():
                # Get Image Features
                images, target = images.type(self.dtype).to(self.device), target.to(self.device)
                image_features = self.image_encoder(images)
                image_features /= image_features.norm(dim=-1, keepdim=True)

            # Get OOD Features
            prompts = [self.ood_prompter[i]() for i in range(self.K)]
            text_features = [self.text_encoder(prompts[i], self.tokenized_prompts[i]) for i in range(self.K)]
            text_features = [text_features[i] / text_features[i].norm(dim=-1, keepdim=True) for i in range(self.K)]
            logit_scale = self.logit_scale.exp().detach()
            logits = [logit_scale * image_features.float() @ text_features[i].T.float() for i in range(self.K)]
            
            # Compute OOD training metrics
            loss_id, loss_ood = 0, 0
            engry_id, engry_ood = 0, 0

            for i in range(self.K):
                batch_label = torch.index_select(self.ood_labels[i], 0, target.long()).long()
                output_id  = logits[i][batch_label >= 0, :]
                output_ood = logits[i][batch_label < 0, :]
                E_id  = -torch.mean(torch.sum(F.log_softmax(output_id, dim=1) * F.softmax(output_id, dim=1), dim=1))
                E_ood = -torch.mean(torch.sum(F.log_softmax(output_ood, dim=1) * F.softmax(output_ood, dim=1), dim=1))
                cur_loss_id  = F.cross_entropy(logits[i], batch_label, ignore_index=-1)
                cur_loss_ood = torch.clamp(self.ood_margin + E_id - E_ood, min=0)
                loss_id += cur_loss_id
                loss_ood += cur_loss_ood
                engry_id += E_id.item()
                engry_ood += E_ood.item()
            loss_id, loss_ood = loss_id / self.K, loss_ood / self.K
            engry_id, engry_ood = engry_id / self.K, engry_ood / self.K
            # loss_id  = torch.clamp(loss_id,  min=0.01)
            # loss_ood = torch.clamp(loss_ood, min=0.01)
            loss = loss_id + loss_ood

            self.ood_optimizer.zero_grad()
            loss.backward()
            self.ood_optimizer.step()
            # self.ood_scheduler.step()
            avg_id_loss.append(loss_id.item())
            avg_ood_loss.append(loss_ood.item())
            avg_id_engry.append(engry_id)
            avg_ood_engry.append(engry_ood)

        return np.mean(avg_id_loss), np.mean(avg_ood_loss), np.mean(avg_id_engry), np.mean(avg_ood_engry)
    
    def train_iid_epoch(self, cfg, logger, train_loader):
        '''image_encoder -> open set detector -> close set & open set
            fine-tuned prompt with close set and the other with open set
        '''
        self.coop_prompter.train()
        # Init Coop Prompter
        coop_tokenized_prompts, coop_masks, coop_labels, coop_indices = self.setup_coop_prompter(train_loader)

        cos = torch.nn.CosineSimilarity(dim=1, eps=1e-07)
        avg_loss = []
        iid_rate = []
        for idx, (images, target, _, clip_logits, ood_probas) in enumerate(train_loader):
            with torch.no_grad():
                # Get Image Features
                images, target = images.type(self.dtype).to(self.device), target.to(self.device)
                image_features = self.image_encoder(images)
                image_features /= image_features.norm(dim=-1, keepdim=True)
                clip_logits = clip_logits.to(self.device)
                ood_probas = ood_probas.to(self.device)
                ood_scores = torch.max(ood_probas, 1)[0]
                ood_select = torch.max(ood_scores, 1)[0]

            # Get CoOp text features
            coop_prompts = [self.coop_prompter[i]() for i in range(self.K)]
            coop_text_features = [self.text_encoder(coop_prompts[i], coop_tokenized_prompts[i]) for i in range(self.K)]
            coop_text_features = [coop_text_features[i] / coop_text_features[i].norm(dim=-1, keepdim=True) for i in range(self.K)]
            logit_scale = self.logit_scale.exp().detach()
            coop_logits = [logit_scale * image_features.float() @ coop_text_features[i].T.float() for i in range(self.K)]
            
            # Get OOD detector
            loss = 0
            for i in range(self.K):
                iid_sample = ood_scores[:, i] >= self.threshold
                ood_sample = ood_scores[:, i] <  self.threshold
                cur_loss = 0
                iid_rate += iid_sample.cpu().numpy().tolist()
                if torch.sum(iid_sample) > 0:
                    cur_loss += F.cross_entropy(coop_logits[i][iid_sample, :], target.long()[iid_sample], ignore_index=-1)
                if torch.sum(ood_sample) > 0:
                    iid_class  = coop_labels[i] >= 0
                    ood_class  = coop_labels[i] <  0
                    P = F.softmax(coop_logits[i], dim=1)
                    Q = F.softmax(clip_logits, dim=1)
                    P = P[ood_sample, :]
                    Q = Q[ood_sample, :]
                    score = F.kl_div(P.log(), Q, reduction="batchmean")
                    cur_loss += score 
                loss += cur_loss
            loss /= self.K
            self.iid_optimizer.zero_grad()
            loss.backward()
            self.iid_optimizer.step()
            self.iid_scheduler.step()
            avg_loss.append(loss.item())

        return np.mean(avg_loss)

    def evaluate_ood(self, cfg, logger, loader, seen_classes):
        self.ood_prompter.eval()
        self.coop_prompter.eval()
        self.clip_prompter.eval()

        self.setup_ood_prompter(loader)
        with torch.no_grad():
            # Compute OOD text features
            ood_prompts = [self.ood_prompter[i]() for i in range(self.K)]
            ood_text_features = [self.text_encoder(ood_prompts[i], self.tokenized_prompts[i]).to(self.device) for i in range(self.K)]
            ood_text_features = [ood_text_features[i] / ood_text_features[i].norm(dim=-1, keepdim=True) for i in range(self.K)]
        # print(ood_prompts[0].shape)
        # Compute energy
        scores, classnames = [], []
        gts, probs = [], []
        with torch.no_grad():
            for ids, (images, labels, cnames) in enumerate(loader):
                images = images.type(self.dtype).to(self.device)
                image_features = self.image_encoder(images)
                image_features = image_features / image_features.norm(dim=-1, keepdim=True)
                logit_scale = self.logit_scale.exp()
                ood_logits  = [ logit_scale * image_features.float() @ ood_text_features[i].T.float() for i in range(self.K) ]
                ood_probas = torch.cat([
                    F.softmax(ood_logits[i], dim=1)[:, : self.n_divs].unsqueeze(-1)
                    for i in range(self.K)
                ], -1)
                # print(ood_probas.shape)
               
                scores.append(ood_probas.cpu())
                classnames += list(cnames)

            # Compute Cls metrics
            scores = torch.cat(scores, 0)
            scores = torch.nan_to_num(scores, nan=0.0)
            scores = torch.clamp(scores, min=0, max=1)
            
            # Compute OOD Auroc
            probs  = torch.max(scores.view(scores.shape[0], -1), 1)[0].cpu().numpy()
            gts    = np.array([c in seen_classes for c in classnames])
            auroc  = metrics.roc_auc_score(gts, probs)

        return classnames, scores.numpy(), auroc
    
    def evaluate_OOD_stage(self, cfg, logger, loader):
        last_time = time.time()

        # Compute text features
        with torch.no_grad():
            self.setup_ood_prompter(loader)
            ood_prompts = [self.ood_prompter[i]() for i in range(self.K)]
            ood_text_features = [self.text_encoder(ood_prompts[i], self.tokenized_prompts[i]).to(self.device) for i in range(self.K)]
            ood_text_features = [ood_text_features[i] / ood_text_features[i].norm(dim=-1, keepdim=True) for i in range(self.K)]
            self.OOD_scores = []
            self.clip_prompter.get_prefix_suffix_token(loader.dataset.classnames)
            self.clip_prompter.eval()
            self.CLIP_logits = []

        # Compute OOD values
        with torch.no_grad():
            for ids, (images, labels, cnames) in enumerate(loader):
                images = images.type(self.dtype).to(self.device)
                image_features = self.image_encoder(images)
                image_features = image_features / image_features.norm(dim=-1, keepdim=True)
                logit_scale = self.logit_scale.exp()

                ood_logits  = [ logit_scale * image_features.float() @ ood_text_features[i].T.float() for i in range(self.K) ]
                ood_probas = torch.cat([
                    F.softmax(ood_logits[i], dim=1)[:, :self.n_divs].unsqueeze(-1)
                    for i in range(self.K)
                ], -1)
                self.OOD_scores.append(ood_probas.detach().cpu())
                
                clip_logits = self.clip_prompter(images)
                self.CLIP_logits.append(clip_logits.detach().cpu())
        
        logger.info("Evaluate OOD testing set {:.2f}S".format(time.time() - last_time))
    
    
    def evaluate(self, cfg, logger, loader, seen_classes):
        last_time = time.time()
        self.ood_prompter.eval()
        self.coop_prompter.eval()
        self.clip_prompter.eval()
        self.evaluate_OOD_stage(cfg, logger, loader)

        # Compute text features
        with torch.no_grad():
            coop_tokenized_prompts, coop_masks, coop_labels, coop_indices = self.setup_coop_prompter(loader)
            coop_prompts = [self.coop_prompter[i]() for i in range(self.K)]
            coop_text_features = [self.text_encoder(coop_prompts[i], coop_tokenized_prompts[i]) for i in range(self.K)]
            coop_text_features = [coop_text_features[i] / coop_text_features[i].norm(dim=-1, keepdim=True) for i in range(self.K)]
        
        # Compute Prediction
        scores, classnames = [], []
        predicts, targets = [], []
        gts, probs = [], []
        with torch.no_grad():
            for ids, (images, labels, cnames) in enumerate(loader):
                images = images.type(self.dtype).to(self.device)
                image_features = self.image_encoder(images)
                image_features = image_features / image_features.norm(dim=-1, keepdim=True)
                logit_scale = self.logit_scale.exp()
                
                ood_probas = self.OOD_scores[ids].to(self.device)
                clip_logits = self.CLIP_logits[ids].to(self.device)
                coop_logits = torch.cat([ (logit_scale * image_features.float() @ coop_text_features[i].T.float()).unsqueeze(-1) for i in range(self.K) ], -1)

                ood_scores = torch.max(ood_probas, 1)[0]
                iid_select = ood_scores >= self.threshold
                iid_select = iid_select.unsqueeze(1).expand(*coop_logits.shape)
                coop_logits = torch.where(iid_select, coop_logits, 0).mean(-1)
                coop_probas = F.softmax(coop_logits, dim=1)
                clip_probas = F.softmax(clip_logits, dim=1)
                ood_select = torch.max(ood_scores, 1)[0] < self.threshold
                ood_select = ood_select.unsqueeze(1).expand(*clip_logits.shape)
                logits = torch.where(ood_select, clip_probas, coop_probas)

                predicts.append(logits.cpu())
                targets.append(labels.cpu())
                scores.append(ood_probas.cpu())
                classnames += list(cnames)

            # Compute Cls metrics
            scores   = torch.cat(scores, 0)
            predicts = torch.cat(predicts, 0)
            targets  = torch.cat(targets,  0)
            scores   = torch.nan_to_num(scores, nan=0.0)
            scores   = torch.clamp(scores, min=0, max=1)
            
            # Compute OOD Auroc
            probs    = torch.max(scores.view(scores.shape[0], -1), 1)[0].cpu().numpy()
            gts      = np.array([c in seen_classes for c in classnames])
            auroc    = metrics.roc_auc_score(gts, probs)
        
        logger.info("Evaluate testing set {:.2f}S".format(time.time() - last_time))
        return predicts.numpy(), targets.numpy(), classnames, scores.numpy(), auroc
    
    def save_prediction(self, predicts, name="last"):
        pred_path = os.path.join(self.predict_path, f"{name}.pkl")
        last_pred_path = os.path.join(self.predict_path, "last.pkl")
        pickle.dump(predicts, file=open(pred_path, 'wb+'))
        shutil.copyfile(pred_path, last_pred_path)