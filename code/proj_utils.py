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