import pathlib
import shutil
import json
import numpy as np
from PIL import Image

import torch
from torch.utils.data import Dataset






class CandidateDataset(Dataset):
    """Candidate dataset."""
    
    def __init__(self, pathname, transform=None, train=True):
        """
        Args:
            pathname (pathlib.Path): Path to the npz file.
            transform (callable, optional): Optional transform to be applied on a sample.
            train (boolean): Extract train (True) or test (False) set from the file.
        """
        self.samples, self.targets = np_loader(pathname.resolve(), train=train)
        self.transform = transform
        
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (sample, target) where target is class_index of the target class.
        """
        sample, target = self.samples[index], self.targets[index]
        sample = Image.fromarray(np.moveaxis(sample, 0, -1))
        
        if self.transform is not None:
            sample = self.transform(sample)
            
        # TODO: Target transform.
        
        return sample, target
    
def np_loader(filename, train=True):
    data = np.load(filename)
    if train:
        samples = data['X_train'].transpose(0, 3, 1, 2)
        targets = data['y_train']
    else:
        samples = data['X_test'].transpose(0, 3, 1, 2)
        targets = data['y_test']
    return samples, targets


def adjust_learning_rate(optimizer, epoch, args):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    lr = args.lr * (args.lrdecay ** (epoch // 30))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def save_checkpoint(state, is_best, filename='checkpoint.pth.tar'):
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, 'model_best.pth.tar')


def save_config(args):
    config_file = args.outpath / 'config.json'
    
    config_dict = {k:(str(v) if isinstance(v, pathlib.PosixPath) else v) for k,v in args.__dict__.items()}
    
    with open(config_file, 'w') as fn:
        json.dump(config_dict, fn, indent=2)