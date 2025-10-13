import torch.nn as nn
import torch
from .common import log
from .random_matrix_gene import gene_random_matrix
from .subscaflinear import SubScafLinear
from .subscafconv2d import SubScafConv2d
import torch.distributed as dist
from .measure_comm import measure_all_reduce, measure_broadcast

def replace_with_subscaf_layer(model, target_modules_list, device, args, jump_modules_list=[], layer='linear'):
    def replace_module(model, target_modules_list, jump_modules_list=None):
        """replace Conv2d/Linear modules in model into SubScafConv2d/SubScafLinear module"""
        nonlocal num_subscaf_params, subscaf_params, lbd, comp_mat_rec
        for name, module in model.named_children():
            # only revise module with param_name has "mlp" or "attn"
            if isinstance(module, replace_class) and any(target_key in name for target_key in target_modules_list):
                log(f"enable Subspace Scaffold for weights in module: {name}")
                if layer == 'conv2d':
                    module.in_features = module.kernel_size[0]
                    module.out_features = module.kernel_size[1]
                if args.adaptive_cp_rate != 0:
                    record_cp_dim = args.comp_dim
                    args.comp_dim = min(max(int(module.in_features * args.adaptive_cp_rate), args.comp_dim), module.in_features)

                # create compression matrix only when new shape demand occur
                if (args.comp_dim, module.in_features) not in comp_mat_rec.keys():
                    #comp_mat = gene_random_matrix(module.out_features, args.comp_dim, args.gene_method).to(device)
                    comp_mat = gene_random_matrix(args.comp_dim, module.in_features, args.gene_method).to(device)
                    comp_mat = comp_mat.to(module.weight.dtype)
                    dist.broadcast(comp_mat, src=0)
                    comp_mat_rec[(args.comp_dim, module.in_features)] = comp_mat
                else:
                    comp_mat = comp_mat_rec[(args.comp_dim, module.in_features)]

                # substitue all Conv2d/Linear module into SubScafConv2d/SubScafLinear module
                new_layer = subscaf_class(args.comp_dim, comp_mat, module)

                setattr(model, name, new_layer)

                # record the subscaf module total parameters
                num_subscaf_params += sum(p.numel() for p in new_layer.parameters())

                # add parameter into trainable parameter
                subscaf_params += [p for p in new_layer.parameters()]

                # initialize lambda
                #lbd.append(torch.zeros((args.comp_dim, module.in_features), device=device, requires_grad=False))
                if subscaf_class == SubScafLinear:
                    layer_lbd = torch.zeros((module.out_features, args.comp_dim), device=device, requires_grad=False, dtype=module.weight.dtype)
                else:
                    layer_lbd = torch.zeros((module.out_channels, module.in_channels // module.groups, module.out_features, args.comp_dim), device=device, requires_grad=False, dtype=module.weight.dtype)
                # in_features is needed for optimizer later
                layer_lbd.in_features = module.in_features
                lbd.append(layer_lbd)
                if args.adaptive_cp_rate != 0:
                    # recover compression dimension
                    args.comp_dim = record_cp_dim

            else:
                if args.jump_certain_modules and name in jump_modules_list:
                    continue
                replace_module(module, target_modules_list, jump_modules_list)

    # set some variable for replace_module
    if args.gene_method == 'svd':
        # use cd at first
        use_svd = True
        args.gene_method = 'cd'
    else:
        use_svd = False
    # choose aim layer
    if layer.lower() == 'linear':
        replace_class = nn.Linear
        subscaf_class = SubScafLinear
    elif layer.lower() == 'conv2d':
        replace_class = nn.Conv2d
        subscaf_class = SubScafConv2d
    else:
        assert True, 'Only Support Conv2d and Linear'
    num_subscaf_params = 0
    subscaf_params = []
    lbd = []
    comp_mat_rec = {} 
    replace_module(model, target_modules_list, jump_modules_list)
    if use_svd:
        # recover svd gene method
        args.gene_method = 'svd'
    return num_subscaf_params, subscaf_params, lbd, comp_mat_rec


@torch.no_grad()
def outer_update(model, lbd, comp_mat_rec, target_modules_list, opt, subscaf_params, args, device, jump_modules_list=None, gene_new_cp=True, grad_dict={}, layer='linear'):
    def subscaf_outer_update(model, lbd, comp_mat_rec, target_modules_list, jump_modules_list, grad_dict):
        """carry out one outer update for subspace scaffold algorithm"""
        nonlocal idx, new_comp_mat_rec
        for name, module in model.named_children():
            if isinstance(module, replace_class) and any(target_key in name for target_key in target_modules_list):
                # all_reduce b
                avg_b = module.b.detach().clone()
                if not args.measure_comm:
                    dist.all_reduce(avg_b, op=dist.ReduceOp.AVG)
                else:
                    time_all_reduce = measure_all_reduce(avg_b, dist.ReduceOp.AVG)
                    all_reduce_times.append(time_all_reduce)
                    all_reduce_tensors.append(avg_b)
                if args.adaptive_cp_rate != 0:
                    record_cp_dim = args.comp_dim
                    args.comp_dim = min(max(int(module.in_features * args.adaptive_cp_rate), args.comp_dim), module.in_features)

                # generate new compression matrix
                if (args.comp_dim, module.in_features) not in new_comp_mat_rec.keys():
                    #new_comp_mat = gene_random_matrix(module.out_features, args.comp_dim, args.gene_method).to(device)
                    if args.gene_method == "svd":
                        new_comp_mat = gene_random_matrix(
                                                args.comp_dim, 
                                                module.in_features, 
                                                args.gene_method, 
                                                grad_dict[module.b] @ comp_mat_rec[(args.comp_dim, module.in_features)],
                                            ).to(device)
                    else:
                        new_comp_mat = gene_random_matrix(args.comp_dim, module.in_features, args.gene_method).to(device)
                    new_comp_mat = new_comp_mat.to(module.b.dtype)
                    if not args.measure_comm:
                        dist.broadcast(new_comp_mat, src=0)
                    else:
                        broadcast_time = measure_broadcast(new_comp_mat, src=0)
                        broadcast_times.append(broadcast_time)
                        broadcast_tensors.append(new_comp_mat)
                    new_comp_mat_rec[(args.comp_dim, module.in_features)] = new_comp_mat
                else:
                    new_comp_mat = new_comp_mat_rec[(args.comp_dim, module.in_features)]

                update_factor = comp_mat_rec[(args.comp_dim, module.in_features)] @ new_comp_mat.T

                # update momentum_buffer
                if args.momentum > 0 and gene_new_cp:
                    if not args.per_layer_weight_update:
                        opt.update_m(module.b, 
                                        - avg_b @ update_factor / (args.lr * args.tau), 
                                        avg_b, 
                                        comp_mat_rec[(args.comp_dim, module.in_features)],
                                        new_comp_mat)
                        #opt.update_m(module.b, update_factor = update_factor)
                    else:
                        opt[module.b].update_m(module.b, 
                                                - avg_b @ update_factor / (args.lr * args.tau), 
                                                avg_b, 
                                                comp_mat_rec[(args.comp_dim, module.in_features)],
                                                new_comp_mat)
                        #opt[module.b].update_m(module.b, update_factor=update_factor)
                                
                # update lbd for every modules
                if gene_new_cp:
                    lbd[idx] = (lbd[idx] + module.b - avg_b) @ update_factor 
                    lbd[idx].in_features = module.in_features
                else:
                    lbd[idx] = (lbd[idx] + module.b - avg_b)
                    lbd[idx].in_features = module.in_features
                assert lbd[idx].shape == (module.out_features, args.comp_dim) or lbd[idx].shape == (module.out_channels, module.in_channels // module.groups, module.out_features, module.comp_dim)

                # update compression matrix, b and x
                new_x = (module.x + avg_b @ comp_mat_rec[(args.comp_dim, module.in_features)])
                module.update(comp_mat=new_comp_mat, x=new_x, b=True)

                # update idx
                idx += 1
                if args.adaptive_cp_rate != 0:
                    # recover compression dimension
                    args.comp_dim = record_cp_dim
            else:
                if args.jump_certain_modules and name in jump_modules_list:
                    continue
                subscaf_outer_update(module, lbd, comp_mat_rec, target_modules_list, jump_modules_list, grad_dict)
    idx = 0
    if gene_new_cp == False:
        # not update compression matrix
        new_comp_mat_rec = comp_mat_rec
    else:
        new_comp_mat_rec = {}

    # choose aim layer
    if layer.lower() == 'linear':
        replace_class = nn.Linear
    elif layer.lower() == 'conv2d':
        replace_class = SubScafConv2d
    else:
        assert True, 'Only Support Conv2d and Linear'

    if args.measure_comm:
        all_reduce_times = []
        all_reduce_tensors = []
        broadcast_times = []
        broadcast_tensors = []

    subscaf_outer_update(model, lbd, comp_mat_rec, target_modules_list, jump_modules_list, grad_dict)
    comp_mat_rec = new_comp_mat_rec

    # update lbd
    if not args.per_layer_weight_update:
        opt.update_lbd(lbd)
    else:
        for (p, l) in zip(subscaf_params, lbd):
            opt[p].update_lbd(lbd=[l])

    if args.measure_comm:
        return all_reduce_times, all_reduce_tensors, broadcast_times, broadcast_tensors

