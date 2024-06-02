import os
import random
import argparse
import yaml
import logging
from tqdm import tqdm

import torch
import torch.nn.functional as F
import torch.nn as nn
from torch.autograd import Variable
from torch.utils.data import DataLoader
import torchvision.transforms as transforms

from clip import clip
from clip.simple_tokenizer import SimpleTokenizer as _Tokenizer

from utils.util_algo import *
from utils.util_data import *
from models import DeCoOpLearner
from datasets import build_dataset

_tokenizer = _Tokenizer()

def set_random_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

def get_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config',   dest='config',  help='settings of methods in yaml format')
    parser.add_argument('--dataset',  dest='dataset', help='settings of dataset in yaml format')
    parser.add_argument('--seed',     type=int, default=1, metavar='N', help='fix random seed')
    parser.add_argument('--logdir',   type=str, default="./results")
    args = parser.parse_args()
    return args    

def main():
    # Set Cuda
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Prepare configs and logs
    args = get_arguments()
    set_random_seed(args.seed)
    assert(os.path.exists(args.config))
    assert(os.path.exists(args.dataset))
    cfg = yaml.load(open(args.config, 'r'), Loader=yaml.Loader)
    cfg.update( yaml.load(open(args.dataset, 'r'), Loader=yaml.Loader) )
    logging.basicConfig(level=logging.INFO)
    log_file_path = args.logdir + f"/{cfg['method']['name']}/{cfg['dataset']['name']}/{args.seed}/log.txt"
    log_directory = os.path.dirname(log_file_path)
    cfg["log"] = {
        "root": log_directory, 
        "model": os.path.join(log_directory, "model"), 
        "prediction": os.path.join(log_directory, "prediction")
    }
    if not os.path.exists(log_directory): 
        os.makedirs(log_directory)
        os.makedirs(cfg["log"]["model"])
        os.makedirs(cfg["log"]["prediction"])
    file_handler = logging.FileHandler(log_file_path)
    file_handler.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(formatter)
    logger = logging.getLogger()
    logger.addHandler(file_handler)
    logger.info(args)
    logger.info(cfg)

    # Load clip
    clip_model, transform = clip.load(cfg['method']["backbone"])
    clip_model = clip_model.to(device)

    # Prepare dataset
    logger.info('Preparing dataset')
    base_dataset = build_dataset(dataset=cfg['dataset']['name'], root=cfg['dataset']['root'], subsample='base', num_shots=cfg['dataset']['shots'], transform=transform, type='train', seed=args.seed)
    new_dataset  = build_dataset(dataset=cfg['dataset']['name'], root=cfg['dataset']['root'], subsample='new',  num_shots=cfg['dataset']['shots'], transform=transform, type='train', seed=args.seed)
    test_dataset = build_dataset(dataset=cfg['dataset']['name'], root=cfg['dataset']['root'], subsample='all',  num_shots=-1, transform=transform, type='test', seed=args.seed)
    test_new_dataset = build_dataset(dataset=cfg['dataset']['name'], root=cfg['dataset']['root'], subsample='new',  num_shots=-1, transform=transform, type='test', seed=args.seed)
    val_dataset  = build_dataset(dataset=cfg['dataset']['name'], root=cfg['dataset']['root'], subsample='all',  num_shots=-1, transform=transform, type='val', seed=args.seed)
    val_new_dataset  = build_dataset(dataset=cfg['dataset']['name'], root=cfg['dataset']['root'], subsample='new',  num_shots=-1, transform=transform, type='val', seed=args.seed)
    base_loader = DataLoader(dataset=base_dataset, batch_size=cfg['method']['train_batch_size'], shuffle=True,  num_workers=16)
    new_loader  = DataLoader(dataset=new_dataset,  batch_size=cfg['method']['train_batch_size'], shuffle=True,  num_workers=16)
    test_loader = DataLoader(dataset=test_dataset, batch_size=cfg['method']['eval_batch_size'],  shuffle=False, num_workers=16)
    test_new_loader = DataLoader(dataset=test_new_dataset, batch_size=cfg['method']['eval_batch_size'],  shuffle=False, num_workers=16)
    val_loader =  DataLoader(dataset=val_dataset,  batch_size=cfg['method']['eval_batch_size'],  shuffle=False, num_workers=16)
    val_new_loader =  DataLoader(dataset=val_new_dataset,  batch_size=cfg['method']['eval_batch_size'],  shuffle=False, num_workers=16)
    logger.info("Training set: {}  Testing set: {}".format(len(base_dataset), len(test_dataset)))
    
    model = None
    if cfg["method"]["name"].startswith("decoop"): 
        model = DeCoOpLearner(args=args, cfg=cfg, logger=logger, device=device, clip_model=clip_model, dataset=base_dataset)
    model.train(cfg=cfg, logger=logger, base_loader=base_loader, test_loader=test_loader)


if __name__ == '__main__':
    main()