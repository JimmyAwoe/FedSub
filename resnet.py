import argparse
import os
import time
import torch.distributed as dist
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from transformers import get_cosine_schedule_with_warmup
from utils import (
    split_dataset_by_class,
    log, 
    init_process_group, 
    main_parse_args, 
    replace_with_subscaf_layer,
    get_subscaf_optimizer,
    resnet_module,
    outer_update,
    measure_all_reduce,
)


def parse_args(args, remaining_args):
    parser = argparse.ArgumentParser()
    # model
    parser.add_argument('--arch', default='resnet1202', choices=['resnet20', 'resnet32', 'resnet44', 
                                                                 'resnet56', 'resnet110', 'resnet1202'])

    # data
    parser.add_argument('--data_hete', action="store_true")

    # training
    parser.add_argument('--epochs', default=200, type=int, metavar='N',
                        help='number of total epochs to run')
    parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true',
                        help='evaluate model on validation set')
    parser.add_argument('--pretrained', dest='pretrained', action='store_true',
                        help='use pre-trained model')
    
    # log
    parser.add_argument('--print-freq', '-p', default=50, type=int,
                        metavar='N', help='print frequency (default: 50)')

    # measure communication
    parser.add_argument("--measure_comm", action="store_true", help="measure the time used for communication")

    new_args, _ = parser.parse_known_args(remaining_args)
    args = argparse.Namespace(**vars(args), **vars(new_args))
    return args

def main(args):
    best_prec1 = 0
    rank = int(os.environ.get("RANK"))
    world_size = int(os.environ.get("WORLD_SIZE"))
    device = f"cuda:{rank}"

    # model
    model = resnet_module.__dict__[args.arch]()
    model.to(device)

    cudnn.benchmark = True


    # load data 
    normalize = transforms.Normalize(mean=[0.507, 0.487, 0.441],
                                    std=[0.267, 0.256, 0.276])
    train_dataset = datasets.CIFAR100(root='./data', train=True, transform=transforms.Compose([
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                normalize,
            ]), download=True),
    val_dataset = datasets.CIFAR100(root='./data', train=False, transform=transforms.Compose([
                transforms.ToTensor(),
                normalize,
            ])),
    if args.data_hete:
        # set data heterogenity
        train_loader = split_dataset_by_class(train_dataset[0], rank, world_size, args.batch_size)
        val_loader = split_dataset_by_class(val_dataset[0], rank, world_size, args.batch_size)
    else:
        train_loader = torch.utils.data.DataLoader(
            train_dataset[0],
            batch_size=args.batch_size, shuffle=True,
            num_workers=4, pin_memory=True)

        val_loader = torch.utils.data.DataLoader(
            val_dataset[0],
            batch_size=128, shuffle=False,
            num_workers=4, pin_memory=True)

    # define loss function (criterion) and optimizer
    criterion = nn.CrossEntropyLoss().to(device)

    # optimizer
    trainable_param = [p for p in model.parameters() if p.requires_grad == True]
    param_before_comp = sum(p.numel() for p in model.parameters()) / 1_000_000
    trainable_param_before_comp = sum(p.numel() for p in model.parameters() if p.requires_grad == True) / 1_000_000
    if args.optimizer == 'subscafsgd':
        target_modules_list = ["conv1", "conv2"]
        jump_modules_list = []
        # define nonlocal variables for replace module
        num_subscaf_params, subscaf_params, lbd, comp_mat_rec = replace_with_subscaf_layer(model, 
                                                                                            target_modules_list, 
                                                                                            device, 
                                                                                            args, 
                                                                                            jump_modules_list,
                                                                                            'conv2d')
        id_subscaf_params = [id(p) for p in subscaf_params]
        # make parameters without "is_comp" to another group
        regular_params = [p for p in model.parameters() if id(p) not in id_subscaf_params]
        # make parameters with "is_comp" to a single group
        param_groups = [{'params': regular_params, 'is_comp': False}, 
                        {'params': subscaf_params, 'is_comp': True, 'lbd': lbd, 'tau': args.tau, 
                         'compression_dim': args.comp_dim}]

    log(f"\n{model}\n")
    log(f"Total params: {param_before_comp:.2f}M")
    if 'subscaf' in args.optimizer.lower():
        log(f"Trainable params before compression: {trainable_param_before_comp:.2f}M")
        log(f"Trainable params after compression: {sum(p.numel() for p in model.parameters()) / 1_000_000:.2f}M")
        log(f"Total params with Subspace Scaffold enabled: {num_subscaf_params / 1_000_000:.2f}M")
    else: 
        log(f"Trainable params: {sum(p.numel() for p in model.parameters() if p.requires_grad) / 1_000_000:.2f}M")

    if args.optimizer == 'subscafsgd':
        args.per_layer_weight_update = False
        optimizer, schedule = get_subscaf_optimizer(args, param_groups, regular_params, subscaf_params, lbd, model)
    elif args.optimizer == 'sgd' or args.optimizer == 'fedavgsgd':
        optimizer = torch.optim.SGD(model.parameters(), 
                                    args.lr,
                                    momentum=args.momentum,
                                    weight_decay=args.weight_decay,
                                    nesterov=args.nesterov)
        # schedule
        if not args.constant_lr:
            schedule = get_cosine_schedule_with_warmup(optimizer,
                                                    num_warmup_steps=args.warmup,
                                                    num_training_steps=len(train_loader) * args.epochs)
        else:
            schedule = None


    if args.evaluate:
        validate(val_loader, model, criterion, args, device)

    for epoch in range(args.epochs):

        # train for one epoch
        log('current lr {:.5e}'.format(optimizer.param_groups[0]['lr']))
        if args.optimizer == 'subscafsgd':
            train(train_loader, model, criterion, optimizer, epoch, schedule, args, device,
                lbd, comp_mat_rec, target_modules_list, subscaf_params, jump_modules_list)
        else:
            train(train_loader, model, criterion, optimizer, epoch, schedule, args, device)


        if args.evaluate:
            # evaluate on validation set
            prec1 = validate(val_loader, model, criterion, args, device)

            # remember best prec@1 and save checkpoint
            best_prec1 = max(prec1, best_prec1)

def train(train_loader, model, criterion, optimizer, epoch, schedule, args, device, *arg):
    """
        Run one train epoch
    """
    if args.optimizer == 'subscafsgd':
        lbd, comp_mat_rec, target_modules_list, subscaf_params, jump_modules_list = arg
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()

    # switch to train mode
    model.train()
    update_step = 0
    if args.measure_comm:
        all_reduce_times = []
        all_reduce_tensors = []
        broadcast_times = []
        broadcast_tensors = []

    end = time.time()
    for i, (input, target) in enumerate(train_loader):

        # measure data loading time
        data_time.update(time.time() - end)

        target = target.to(device)
        input_var = input.to(device)
        target_var = target

        # compute output
        output = model(input_var)
        loss = criterion(output, target_var)

        update_step += 1

        # compute gradient and update 
        if args.optimizer == 'sgd':
            optimizer.zero_grad()
            if not args.constant_lr:
                schedule.step()
            loss.backward()
            with torch.no_grad():
                for p in model.parameters():
                    dist.all_reduce(p.grad, op=dist.ReduceOp.AVG)
            optimizer.step()
        elif 'subscaf' in args.optimizer.lower():
            optimizer.zero_grad()
            if not args.constant_lr:
                schedule.step()
            loss.backward()
            optimizer.step()
        elif 'fedavg' in args.optimizer.lower():
            optimizer.zero_grad()
            if not args.constant_lr:
                schedule.step()
            loss.backward()
            optimizer.step()
            
        if "subscaf" in args.optimizer.lower() and update_step % args.tau == 0:
            if update_step % args.update_cp_freq != 0:
                gene_new_cp = False 
            else:
                gene_new_cp = True
            if args.measure_comm:
                temp_all_reduce_times, temp_all_reduce_tensors, temp_broadcast_times, temp_broadcast_tensors =\
                outer_update(model, lbd, comp_mat_rec, target_modules_list, optimizer, 
                                subscaf_params, args, device, jump_modules_list, gene_new_cp, layer='conv2d')
            else:
                outer_update(model, lbd, comp_mat_rec, target_modules_list, optimizer, 
                                subscaf_params, args, device, jump_modules_list, gene_new_cp, layer='conv2d')

            if args.measure_comm:
                all_reduce_times.extend(temp_all_reduce_times)
                all_reduce_tensors.extend(temp_all_reduce_tensors)
                broadcast_tensors.extend(temp_broadcast_tensors)
                broadcast_times.extend(temp_broadcast_times)
        
        if 'fedavg' in args.optimizer.lower() and update_step % args.tau == 0:
            for p in model.parameters():
                if not args.measure_comm:
                    dist.all_reduce(p, op=dist.ReduceOp.AVG)
                else:
                    times = measure_all_reduce(p, dist.ReduceOp.AVG)
                    all_reduce_times.append(times)
                    all_reduce_tensors.append(p)

        output = output.float()
        loss = loss.float()
        # measure accuracy and record loss
        prec1 = accuracy(output.data, target, (1,)) [0]
        losses.update(loss.item(), input.size(0))
        top1.update(prec1.item(), input.size(0))

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        #if args.use_log and i % args.print_freq == 0:
        if args.use_log and update_step % args.tau == 0:
            #log('Epoch: [{0}][{1}/{2}]\t'
                  #'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  #'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  #'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  #'Prec@1 {top1.val:.3f} ({top1.avg:.3f})'.format(
                      #epoch, i, len(train_loader), batch_time=batch_time,
                      #data_time=data_time, loss=losses, top1=top1))

            log('Epoch: [{0}][{1}/{2}]\t'
                  'Loss: {loss.avg:.4f}\t'
                  'Prec@5: {top1.avg:.3f}\t'
                  'lr: {lr:.4f}'.format(
                      epoch, i, len(train_loader), 
                      loss=losses, top1=top1, lr=optimizer.param_groups[0]["lr"]))
    
    if dist.get_rank() == 0:
        if args.measure_comm:
            # mix all reduce times and broadcast times 
            all_reduce_times.extend(broadcast_times)
            all_reduce_tensors.extend(broadcast_tensors)
            avg_time = sum(all_reduce_times) / args.num_training_steps
            min_time = min(all_reduce_times)
            max_time = max(all_reduce_times)
            communication_volumn = 0
            for p in all_reduce_tensors:
                communication_volumn += p.numel()
            log(f"avg_comm_time: {avg_time}ms, min_comm_time: {min_time}ms, max_time: {max_time}ms, total_comm_volumn: {communication_volumn}")


def validate(val_loader, model, criterion, args, device):
    """
    Run evaluation
    
    Note: Each call to validate creates new AverageMeter instances,
    so each evaluation is independent and does not accumulate across different evaluations.
    """
    # Create new AverageMeter instances for each evaluation
    # This ensures each evaluation is independent and does not accumulate across calls
    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()

    # switch to evaluate mode
    model.eval()

    end = time.time()
    with torch.no_grad():
        for i, (input, target) in enumerate(val_loader):
            target = target.to(device)
            input_var = input.to(device)
            target_var = target.to(device)


            # compute output
            output = model(input_var)
            loss = criterion(output, target_var)

            output = output.float()
            loss = loss.float()

            # measure accuracy and record loss
            prec1 = accuracy(output.data, target)[0]
            losses.update(loss.item(), input.size(0))
            top1.update(prec1.item(), input.size(0))

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if args.use_log and i % args.print_freq == 0:
                log('Test: [{0}/{1}]\t'
                      'Time: {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                      'Loss: {loss.avg:.4f}\t'
                      'Prec@5: {top1.avg:.3f}'.format(
                          i, len(val_loader), batch_time=batch_time, loss=losses,
                          top1=top1))

    log(' * Prec@5 {top1.avg:.3f}'.format(top1=top1))

    # switch back to train mode
    model.train()

    return top1.avg

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


def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        #correct_k = correct[:k].view(-1).float().sum(0)
        correct_k = correct[:k].flatten().float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res


if __name__ == '__main__':
    init_process_group()
    args, unknown = main_parse_args(None)
    args = parse_args(args, unknown)
    main(args)
    dist.destroy_process_group()
