import os
import random
import os.path as osp
import tarfile
import zipfile
from collections import defaultdict
import gdown
import json
import torch
from torch.utils.data import Dataset
import torchvision.transforms as T
from PIL import Image
import clip
import math
import numpy as np

class Datum:
    """Data instance which defines the basic attributes.

    Args:
        impath (str): image path.
        label (int): class label.
        domain (int): domain label.
        classname (str): class name.
    """

    def __init__(self, impath='', label=0, domain=-1, classname=''):
        assert isinstance(impath, str)
        assert isinstance(label, int)
        assert isinstance(domain, int)
        assert isinstance(classname, str)

        self.impath = impath
        self.label = label
        self.domain = domain
        self.classname = classname

def read_json(fpath):
    """Read json file from a path."""
    with open(fpath, 'r') as f:
        obj = json.load(f)
    return obj

def write_json(obj, fpath):
    """Writes to a json file."""
    if not osp.exists(osp.dirname(fpath)):
        os.makedirs(osp.dirname(fpath))
    with open(fpath, 'w') as f:
        json.dump(obj, f, indent=4, separators=(',', ': '))


def read_image(path):
    """Read image from path using ``PIL.Image``.

    Args:
        path (str): path to an image.

    Returns:
        PIL image
    """
    if not osp.exists(path):
        raise IOError('No file exists at {}'.format(path))

    while True:
        try:
            img = Image.open(path).convert('RGB')
            return img
        except IOError:
            print(
                'Cannot read image from {}, '
                'probably due to heavy IO. Will re-try'.format(path)
            )


def listdir_nohidden(path, sort=False):
    """List non-hidden items in a directory.

    Args:
         path (str): directory path.
         sort (bool): sort the items.
    """
    items = [f for f in os.listdir(path) if not f.startswith('.') and 'sh' not in f]
    if sort:
        items.sort()
    return items

def read_split(filepath, path_prefix):
        def _convert(items):
            out = []
            for impath, label, classname in items:
                impath = os.path.join(path_prefix, impath)
                item = Datum(
                    impath=impath,
                    label=int(label),
                    classname=classname
                )
                out.append(item)
            return out
        
        print(f'Reading split from {filepath}')
        split = read_json(filepath)
        train = _convert(split['train'])
        val = _convert(split['val'])
        test = _convert(split['test'])

        return train, val, test

def subsample_classes(*args, subsample="all"):
        """Divide classes into two groups. The first group
        represents base classes while the second group represents
        new classes.

        Args:
            args: a list of datasets, e.g. train, val and test.
            subsample (str): what classes to subsample.
        """
        assert subsample in ["all", "base", "new"]
        
        if subsample == "all":
            return args
        
        dataset = args[0]
        labels = set()
        for item in dataset:
            labels.add(item.label)
        labels = list(labels)
        labels.sort()
        n = len(labels)
        # Divide classes into two halves
        m = math.ceil(n / 2)

        print(f"SUBSAMPLE {subsample.upper()} CLASSES!")
        if subsample == "base":
            selected = labels[:m]  # take the first half
        elif subsample == "new":
            selected = labels[m:]  # take the second half
        relabeler = {y: y_new for y_new, y in enumerate(selected)}
        
        output = []
        for dataset in args:
            dataset_new = []
            for item in dataset:
                if item.label not in selected:
                    continue
                item_new = Datum(
                    impath=item.impath,
                    label=relabeler[item.label],
                    classname=item.classname
                )
                dataset_new.append(item_new)
            output.append(dataset_new)
        
        return output

def generate_fewshot_dataset(*data_sources, num_shots=-1, repeat=True, seed=0):
    """Generate a few-shot dataset (typically for the training set).

    This function is useful when one wants to evaluate a model
    in a few-shot learning setting where each class only contains
    a few number of images.

    Args:
        data_sources: each individual is a list containing Datum objects.
        num_shots (int): number of instances per class to sample.
        repeat (bool): repeat images if needed.
    """
    if num_shots < 1:
        if len(data_sources) == 1:
            return data_sources[0]
        return data_sources

    print(f'Creating a {num_shots}-shot dataset')

    output = []

    for data_source in data_sources:
        tracker = split_dataset_by_label(data_source)
        dataset = []

        for label, items in tracker.items():
            if len(items) >= num_shots:
                random.seed(seed)
                sampled_items = random.sample(items, num_shots)
            else:
                if repeat:
                    random.seed(seed)
                    sampled_items = random.choices(items, k=num_shots)
                else:
                    sampled_items = items
            dataset.extend(sampled_items)

        output.append(dataset)

    if len(output) == 1:
        return output[0]

    return output

def split_dataset_by_label(data_source):
    """Split a dataset, i.e. a list of Datum objects,
    into class-specific groups stored in a dictionary.

    Args:
        data_source (list): a list of Datum objects.
    """
    output = defaultdict(list)

    for item in data_source:
        output[item.label].append(item)

    return output

def split_dataset_by_domain(data_source):
    """Split a dataset, i.e. a list of Datum objects,
    into domain-specific groups stored in a dictionary.

    Args:
        data_source (list): a list of Datum objects.
    """
    output = defaultdict(list)

    for item in data_source:
        output[item.domain].append(item)

    return output

def get_lab2cname(data_source):
        """Get a label-to-classname mapping (dict).
        Args:
            data_source (list): a list of Datum objects.
        """
        container = set()
        for item in data_source:
            container.add((item.label, item.classname))
        label2cname = {label: classname for label, classname in container}
        cname2label = {classname: label for label, classname in container}
        labels = list(label2cname.keys())
        labels.sort()
        classnames = [label2cname[label] for label in labels]
        return label2cname, cname2label, classnames



