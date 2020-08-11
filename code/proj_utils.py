import numpy as np
import torch



def grad_norm(net):  
    # returns the norm of the gradient corresponding to the convolutional parameters
    
    # count number of convolutional layers
    nconvnets = 0
    for p in list(filter(lambda p: len(p.data.shape)>1, net.parameters())):
        nconvnets += 1
    
    out_grads = np.zeros(nconvnets)
    p = [x for x in net.parameters() ]
    for ind,p in enumerate(list(filter(lambda p: p.grad is not None and len(p.data.shape)>1, net.parameters()))):
        out_grads[ind] = p.grad.data.norm(2).item()

    return out_grads


def get_jacobian_prod(train_loader, model, crit, gvec, args):
    model.zero_grad()
    for i, (input, target) in enumerate(train_loader):
        for cur_input, cur_target in zip(input, target):
            cur_target = torch.tensor([cur_target]).cuda(args.gpu, non_blocking=True)
            if args.gpu is not None:
                cur_input = cur_input.unsqueeze(0).cuda(args.gpu, non_blocking=True)

            cur_output = model(cur_input)
            loss = crit(cur_output, cur_target)
            
            gvec = (torch.randn((1, args.num_classes)) / len(train_loader.dataset)).cuda(args.gpu, non_blocking=True)
            
            cur_output.backward(gvec)
            # loss.backward(gvec)
        
    return grad_norm( model )

    
def get_jacobian_two_layer(X, y, model, crit):
    
    grads = []
    for cx, cy in zip(X, y):

        cur_grads = []
        model.zero_grad()
        co = model(cx)
        co.backward(torch.ones(len(cy)))

        for p in model.parameters():
            if p.grad is not None and len(p.data.shape)>1:
                cur_grads.append(p.grad.data.numpy().flatten())
        grads.append(np.concatenate(cur_grads))
    return np.array(grads)


def get_jacobian_svd(train_loader, model, args, average_batches=False):
    if average_batches:
        print('Averaging the Jacobian SVD across training set batches.\n' + 
            'This might take longer.')
    else:
        print('Computing the Jacobian for a single training batch.\n' + 
            'This might be faster, but less reliable.')
    gradient_mat = []

    npars = []
    for ind,p in enumerate(list(filter(lambda p: p.grad is not None and len(p.data.shape)>1, net.parameters()))):
        npars.append(list(p.data.shape))
    nconv = np.prod([ np.prod(i) for i in npars[:-1] ])
    nfc = np.prod(npars[-1])
    
    
    running_conv = RunningStats()
    running_fc = RunningStats()
    for i, (input, target) in enumerate(train_loader):
        
        grad_batch = []
        for cur_input, cur_target in zip(input, target):
            cur_gradient = []
            if args.gpu is not None:
                cur_input = cur_input.unsqueeze(0).cuda(args.gpu, non_blocking=True)
            
            for cur_lbl in range(args.num_classes):
                cur_one_hot = [0] * int(args.num_classes)
                cur_one_hot[cur_lbl] = 1
                cur_one_hot = torch.FloatTensor([cur_one_hot]).cuda(args.gpu, non_blocking=True)
                
                model.zero_grad()
                cur_output = model(cur_input)
                cur_output.backward(cur_one_hot)
                for para in model.parameters():
                    cur_gradient.append(para.grad.data.cpu().numpy().flatten())
                    
            grad_batch.append(np.concatenate(cur_gradient))  

        uv, sv, vtv = np.linalg.svd(grad_batch, full_matrices=False) 

        vconv, vfc = [], []
        for cur_v in vtv:
            vconv.append(np.linalg.norm(cur_v[:nconv]))
            vfc.append(np.linalg.norm(cur_v[-nfc:]))
        vconv = np.array(vconv)
        vfc = np.array(vfc)

        if not average_batches:
            return vconv, vfc

        running_conv.push(vconv)
        running_fc.push(vfc)


    return running_conv, running_fc


def get_jacobian(train_loader, model, args):
    gradient_mat = []
    
    for i, (input, target) in enumerate(train_loader):
        
        for cur_input, cur_target in zip(input, target):
            cur_gradient = []
            if args.gpu is not None:
                cur_input = cur_input.unsqueeze(0).cuda(args.gpu, non_blocking=True)
            
            for cur_lbl in range(args.num_classes):
                cur_one_hot = [0] * int(args.num_classes)
                cur_one_hot[cur_lbl] = 1
                cur_one_hot = torch.FloatTensor([cur_one_hot]).cuda(args.gpu, non_blocking=True)
                
                model.zero_grad()
                cur_output = model(cur_input)
                cur_output.backward(cur_one_hot)
                for para in model.parameters():
                    cur_gradient.append(para.grad.data.cpu().numpy().flatten())
                    
            gradient_mat.append(np.concatenate(cur_gradient))   
    return np.array(gradient_mat)
                

class AverageVector:
    """Computes and stores the mean and standard deviation vectors"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.avg = 0
        self.std = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.sum = [i + j for i,j in zip(self.sum, val * n)]
        self.count += n
        self.avg = self.sum / self.count


class RunningStats:
    """Based on `John Cook's`__ method for computing the variance of the data using Welford's algorithm in single pass.

    __ http://www.johndcook.com/standard_deviation.html
    """
    def __init__(self):
        self.n = 0
        self.old_m = 0
        self.new_m = 0
        self.old_s = 0
        self.new_s = 0

    def clear(self):
        self.n = 0

    def push(self, x):
        self.n += 1

        if self.n == 1:
            self.old_m = self.new_m = x
            self.old_s = 0
        else:
            self.new_m = self.old_m + (x - self.old_m) / self.n
            self.new_s = self.old_s + (x - self.old_m) * (x - self.new_m)

            self.old_m = self.new_m
            self.old_s = self.new_s

    def mean(self):
        return self.new_m if self.n else 0.0

    def variance(self):
        return self.new_s / (self.n - 1) if self.n > 1 else 0.0

    def standard_deviation(self):
        return np.sqrt(self.variance())


def scale_initialization(model, layers, gain=1):
    if isinstance(layers, int):
        layers = [layers]
        
    for cur_lay in layers:
        for p in model[cur_lay].parameters():
            p.data = torch.mul(p.data, gain)

    
def get_weights(model, layers):
    weight_list = []
    
    for i in layers:
        weight_list.append(model[i].weight.data.cpu().numpy())
    return weight_list


def rescale_weights(model, norm_dict):
    scaled_norm_dict = {k: (v / np.prod(model[int(k.split('_')[-1])].weight.data.shape)) for k, v in norm_dict.items()}
    base_scale = min(norm_dict.values())
    with torch.no_grad():
        for cur_l, cur_scale in norm_dict.items():
            cur_k = cur_l.split('_')[-1] + '.' + 'weight'
            cur_w = model.state_dict()[cur_k]
            model.state_dict()[cur_k].copy_(cur_w / (cur_scale / base_scale))
            
            
def get_lr_scales(model, base_lr, norm_dict):
    scaled_norm_dict = {k: (v / np.prod(model[int(k.split('_')[-1])].weight.data.shape)) for k, v in norm_dict.items()}
    base_scale = min(norm_dict.values())
    new_lr_dict = {}
    for cur_l, cur_scale in norm_dict.items():
        cur_k = cur_l.split('_')[-1]
        new_lr_dict[cur_k] = base_lr / (cur_scale / base_scale)
    return new_lr_dict