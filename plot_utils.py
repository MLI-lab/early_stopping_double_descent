import numpy as np
import json
import pathlib


def readJsonLog(*out_files, extract='acc1'):
    
    train_log = []
    
    for out_file in out_files:
        with open(out_file) as fn:
            train_log += json.load(fn)
        
    dataset_list = [k for k in train_log[0].keys() if not k == "epoch"]
    
    prog_log = {k:list() for k in dataset_list if extract in train_log[0][k]}
    
    set_flags = {k:True for k in prog_log.keys()}
    
    epoch_arr = []
    for epoch_log in train_log:
        epoch_arr.append(epoch_log['epoch'])
        
        for cur_set in prog_log.keys():
            if extract in epoch_log[cur_set]:
                prog_log[cur_set].append(epoch_log[cur_set][extract])
                
    return np.array(epoch_arr), prog_log


def readParamLogs(*out_files, extract='weight_change'):
    
    train_log = []
    
    for out_file in out_files:
        with open(out_file) as fn:
            train_log += json.load(fn)
    
    assert extract in train_log[0], "{} is not recorded in the log files".format(extract)
    
    layer_list = [k for k in train_log[0][extract].keys()]
    w_log = {k:list() for k in layer_list}
    
    epoch_arr = []
    for epoch_log in train_log:
        epoch_arr.append(epoch_log['epoch'])
        
        for cur_set in w_log.keys():
            w_log[cur_set].append(epoch_log[extract][cur_set])
                
    return np.array(epoch_arr), w_log


def find_experiments_by_config(resdir, conf_req=None, set_req=None):
    """ find experiments that have the required configurations
        resdir: results directory
        conf (dict): dictionary containing desired configurations
    """
    
    log_list = sorted([fn for fn in resdir.glob('**/log.json')])
    exp_idx_dict = {}
    
    for i, cur_log in enumerate(log_list):
        with open(cur_log.parent / 'config.json') as fn:
            conf_log = json.load(fn)
            
        conf_bool = True
        set_bool = True

        if conf_req:
            conf_bool = False
            if (conf_req.items() <= conf_log.items()):
                conf_bool = True
            
        if set_req:
            set_bool = False 
            set_insp = ['set' if (s in conf_log) and conf_log[s] else 'not_set' for s in set_req]
            if not 'not_set' in set_insp:
                set_bool = True

        if conf_bool and set_bool:
            exp_idx_dict[i] = ([conf_log[s] for s in set_req] if set_req else conf_req)

    return exp_idx_dict


def readProgressLog(out_file, two_tests=True):
    
    prog_log = {keyword:list() for keyword in ['train', 'test']}
    if two_tests:
        prog_log.update({'subset':list()})
    
    with open(out_file, 'r') as fn:
        new_lines = fn.readlines()
        
    cntr = 3
    sub_cnt = None
    if two_tests:
        sub_cnt = 2
        
    for new_line in new_lines:
        
        if cntr == 0:
            prog_log['train'].append(float(new_line.split(',')[0]))
            cntr += 1
        
        elif cntr == 1:
            prog_log['test'].append(float(new_line.split(',')[0]))
            cntr += 1
            
        elif cntr == sub_cnt:
            prog_log['subset'].append(float(new_line.split(',')[0]))
            cntr += 1
        
        if new_line[:5] == 'Epoch':
            cntr = 0
    
    return prog_log

def get_max_index(acc_dict, dataset='test'):
    acc_arr = np.array(acc_dict[dataset])
    acc_max = acc_arr.max()
    return acc_max, np.where(acc_arr == acc_max)

def get_min_index(acc_dict, dataset='test'):
    acc_arr = np.array(acc_dict[dataset])[40:]
    acc_min = acc_arr.min()
    return acc_min, np.where(acc_arr == acc_min)

def readRedundancyProgress(out_file):

    redun_dict = dict()
    cntr = 3
#     set_trace()

    with open(out_file, 'r') as fn:
        new_lines = fn.readlines()
        
    for new_line in new_lines:
        if new_line[:9] == 'cifar_len':
            cifar_len = new_line.split(', ')[0].split(' = ')[1]
            times_cifar = new_line.split(', ')[1].split(' = ')[1]
            tiny_frac = new_line.split(', ')[2].split(' = ')[1].strip()
            if not cifar_len in redun_dict.keys():
                redun_dict[cifar_len] = dict()
            if not times_cifar in redun_dict[cifar_len].keys():
                redun_dict[cifar_len][times_cifar] = dict()
            if not tiny_frac in redun_dict[cifar_len][times_cifar].keys():
                redun_dict[cifar_len][times_cifar][tiny_frac] = dict()
                redun_dict[cifar_len][times_cifar][tiny_frac]['train'] = list()
                redun_dict[cifar_len][times_cifar][tiny_frac]['test'] = list()
        
        if cntr == 0:
            redun_dict[cifar_len][times_cifar][tiny_frac]['train'].append(float(new_line.split(',')[0]))
            cntr += 1
        
        elif cntr == 1:
            redun_dict[cifar_len][times_cifar][tiny_frac]['test'].append(float(new_line.split(',')[0]))
            cntr += 1
        
        if new_line[:5] == 'Epoch':
            cntr = 0

    return redun_dict

def readGradResults(out_file, two_tests=False):
    
    prog_log = {keyword:list() for keyword in ['train', 'test', 'loss']}
    if two_tests:
        prog_log.update({'cifar_test':list()})
    
    with open(out_file, 'r') as fn:
        new_lines = fn.readlines()
        
    cntr = 4
    cifar_cnt = 5
    loss_cnt = 2
    if two_tests:
        cifar_cnt = 2
        loss_cnt = 3
    epoch_arr = []
        
    for new_line in new_lines:
        
        if cntr == 0:
            prog_log['train'].append(float(new_line.split(',')[0]))
            cntr += 1
        
        elif cntr == 1:
            prog_log['test'].append(float(new_line.split(',')[0]))
            cntr += 1
            
        elif cntr == cifar_cnt:
            prog_log['cifar_test'].append(float(new_line.split(',')[0]))
            cntr += 1
            
        elif cntr == loss_cnt:
            prog_log['loss'].append(float(new_line.split(',')[0]))
            cntr += 1
        
        if new_line[:5] == 'Epoch':
            cntr = 0
            epoch_arr.append(new_line.split(' ')[-1])
    
    return prog_log, epoch_arr
