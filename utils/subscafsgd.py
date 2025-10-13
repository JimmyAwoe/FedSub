from transformers.activations import ACT2FN
from torch.optim.optimizer import Optimizer, _use_grad_for_differentiable, _default_to_fused_or_foreach
from transformers import get_cosine_schedule_with_warmup
from torch import Tensor
import torch.nn as nn
from typing import Iterable, Callable, Optional, List 
import torch
from torch.optim import SGD

def get_subscaf_optimizer(args, param_groups=None, regular_params=None, subscaf_params=None, lbd=None, model=None, scaler_dict=None):
    if not args.per_layer_weight_update:
        assert param_groups is not None, "Must input param_groups"
        optimizer = SubScafSGD(param_groups, 
                            lr=args.lr, 
                            tau=args.tau, 
                            compression_dim=args.comp_dim,
                            foreach=False,
                            nesterov=args.nesterov,
                            weight_decay=args.weight_decay,
                            momentum=args.momentum,
                            dampening=args.dampening,
                            )
        # we add 1 to num_training_steps for avoiding lr become zero when training, which would cause
        # lbd to be nan
        if not args.constant_lr:
            schedule = get_cosine_schedule_with_warmup(optimizer,
                                                    num_warmup_steps=args.warmup,
                                                num_training_steps=args.num_training_steps + 1)
        else:
            schedule = None
        return optimizer, schedule
    else:
        assert regular_params is not None, "Must input regular_params if you want to use layerwise update"
        assert subscaf_params is not None, "Must input subscaf_params if you want to use layerwise update"
        assert lbd is not None, "Must input lbd if you want to use layerwise update"
        assert model is not None, "Must input model if you want to use layerwise update"

        grad_accumulation = args.total_batch_size // args.batch_size
        optimizer_dict = {p: SubScafSGD([{'params': p, 'is_comp': False}], 
                                        lr=args.lr, 
                                        tau=args.tau, 
                                        compression_dim=args.comp_dim,
                                        nesterov=args.nesterov,
                                        momentum=args.momentum,
                                        weight_decay=args.weight_decay,
                                        dampening=args.dampening,
                                        foreach=False) for p in regular_params}
        for (p, l) in zip(subscaf_params, lbd):
            optimizer_dict.update({p: SubScafSGD([{'params':p, 'is_comp': True, 'lbd': [l]}],
                                                lr=args.lr,
                                                tau=args.tau,
                                                compression_dim=args.comp_dim,
                                                foreach=False,
                                                weight_decay=args.weight_decay,
                                                nesterov=args.nesterov,
                                                dampening=args.dampening,
                                                momentum=args.momentum)})
        def optimizer_hook(p):
            if p.grad is None:
                return
            if not args.constant_lr:
                schedule_dict[p].step()
            if args.mixed_precision:
                scaler_dict[p].step(optimizer_dict[p])
                scaler_dict[p].update()
            else:
                optimizer_dict[p].step()
            optimizer_dict[p].zero_grad()

        schedule_dict = {}
        for p in model.parameters():
            if p.requires_grad:
                # NOTE
                # we add 1 to num_training_steps for avoiding lr become zero when training, which would cause
                # lbd to be nan
                # because in this condition, every backward would call optimizer_hook once, hence push the lr,
                # so in the case of gradient accumulation, we should correspondily longer the warmup and training 
                # step
                if not args.constant_lr:
                    schedule_dict[p] = get_cosine_schedule_with_warmup(optimizer_dict[p],
                                                                    num_warmup_steps=args.warmup * grad_accumulation,
                                                                    num_training_steps=args.num_training_steps * grad_accumulation + 1)
                p.register_post_accumulate_grad_hook(optimizer_hook)
        return optimizer_dict

# create Subspace Scaffold optimizer based on sgd 

class SubScafSGD(Optimizer):
    """
    Implement SGD optimize algorithm for Subspace Scaffold.

    Parameters:
        params (`Iterable[nn.parameter.Parameter]`):
            Iterable of parameters to optimize or dictionaries defining parameter groups.
        lr (`float`, *optional*, defaults to 0.001):
            The learning rate to use.
        tau (`int`, *optional*, defaults to 10):
            The frequency to synchronize among nodes.
        compression_dim (`int`, *optional*, default to 64):
            The expectd compression dimension.
    """
    def __init__(
        self,
        params: Iterable[nn.parameter.Parameter],
        lr: float = 1e-3,
        tau: int = 10,
        compression_dim: int = 64,
        momentum = 0,
        dampening = 0,
        weight_decay = 0,
        nesterov = False,
        *,
        maximize: bool = False, 
        foreach: Optional[bool] = None,
        differentiable: bool = False,
        fused: Optional[bool] = None,
    ):
        if lr < 0.0:
            raise ValueError(f"Invalid learning rate: {lr}")
        if momentum < 0.0:
            raise ValueError(f"Invalid momentum value: {momentum}")
        if weight_decay < 0.0:
            raise ValueError(f"Invalid weight_decay value: {weight_decay}")
        if nesterov and (momentum <= 0 or dampening != 0):
            raise ValueError("Nesterov momentum requires a momentum and zero dampening")

        defaults = dict(lr=lr, momentum=momentum, dampening=dampening,
                        weight_decay=weight_decay, nesterov=nesterov,
                        maximize=maximize, foreach=foreach,
                        differentiable=differentiable, tau=tau,
                        compression_dim=compression_dim,fused=fused)
        super().__init__(params, defaults)

    # directly inherit form torch.optim.SGD
    def __setstate__(self, state):
        SGD.__setstate__(self, state)

    # directly inherit form torch.optim.SGD
    def _init_group(self, group, params_with_grad, d_p_list, momentum_buffer_list):
        return SGD._init_group(self, group, params_with_grad, d_p_list, momentum_buffer_list)

    @torch.no_grad()
    def update_lbd(self, lbd):
        for group in self.param_groups:
            if 'lbd' in group.keys():
                group['lbd'] = lbd

    @torch.no_grad()
    def update_m(self, params, m, avg_b=None, past_p=None, new_p=None):
        for group in self.param_groups:
            if group['is_comp'] == True:
                #if avg_b is not None:
                    ## carry out error feedback
                    #past_m = self.state[params]['momentum_buffer'] @ past_p

                #error = m 
                #m = - avg_b @ update_factor / (args.lr * args.tau)
                #self.state[params]['momentum_buffer'] = self.state[params]['momentum_buffer'] @ update_factor
                # align magnitude 
                # SubScaf-CA
                m_mean = torch.mean(m, dim=0, keepdim=True)
                m_std = torch.std(m, dim=0, keepdim=True) + 1e-10
                past_m_mean = torch.mean(self.state[params]['momentum_buffer'], dim=0, keepdim=True)
                past_m_std = torch.std(self.state[params]['momentum_buffer'], dim=0, keepdim=True)
                # below is SubScaf-MA
                #m_mean = torch.mean(m)
                #m_std = torch.std(m)
                #past_m_mean = torch.mean(self.state[params]['momentum_buffer'])
                #past_m_std = torch.std(self.state[params]['momentum_buffer'])

                m = ((m - m_mean) / m_std + past_m_mean) * past_m_std
                #self.state[params]['momentum_buffer'] = m
                self.state[params]['momentum_buffer'] = None

    @_use_grad_for_differentiable
    def step(self, closure: Callable = None):
        """
        Performs a single optimization step.

        Arguments:
            closure (`Callable`, *optional*): A closure that reevaluates the model and returns the loss.
        """
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            params_with_grad = []
            d_p_list = []
            momentum_buffer_list = []

            has_sparse_grad = self._init_group(group, params_with_grad, d_p_list, momentum_buffer_list)

            subscafsgd(params_with_grad,
                d_p_list,
                momentum_buffer_list,
                weight_decay=group['weight_decay'],
                momentum=group['momentum'],
                lr=group['lr'],
                dampening=group['dampening'],
                nesterov=group['nesterov'],
                maximize=group['maximize'],
                has_sparse_grad=has_sparse_grad,
                foreach=group['foreach'],
                tau=group['tau'],
                lbd = group.get('lbd', None),
                is_comp = group['is_comp'],
                )

            # update momentum_buffers in state
            for p, momentum_buffer in zip(params_with_grad, momentum_buffer_list):
                state = self.state[p]
                state['momentum_buffer'] = momentum_buffer

        return loss

# below are three functions copied from torch.optim.SGD
# Only few changes implemented to achieve subspace scaffold optimize.
def subscafsgd(params: List[Tensor],
        d_p_list: List[Tensor],
        momentum_buffer_list: List[Optional[Tensor]],
        # kwonly args with defaults are not supported by functions compiled with torchscript issue #70627
        # setting this as kwarg for now as functional API is compiled by torch/distributed/optim
        has_sparse_grad: bool = None,
        foreach: Optional[bool] = None,
        tau: int = 10,
        lbd: List[Tensor] = None,
        is_comp: bool = False,
        *,
        weight_decay: float,
        momentum: float,
        lr: float,
        dampening: float,
        nesterov: bool,
        maximize: bool,
        ):
    r"""Functional API that performs SGD algorithm computation.

    See :class:`~torch.optim.SGD` for details.
    """

    if foreach is None:
        # why must we be explicit about an if statement for torch.jit.is_scripting here?
        # because JIT can't handle Optionals nor fancy conditionals when scripting
        if not torch.jit.is_scripting():
            _, foreach = _default_to_fused_or_foreach(params, differentiable=False, use_fused=False)
        else:
            foreach = False

    if foreach and torch.jit.is_scripting():
        raise RuntimeError('torch.jit.script not supported with foreach optimizers')

    if foreach and not torch.jit.is_scripting():
        func = _subscaf_multi_tensor_sgd
    else:
        func = _subscaf_single_tensor_sgd

    func(params,
        d_p_list,
        momentum_buffer_list,
        tau,
        lbd,
        is_comp=is_comp,
        weight_decay=weight_decay,
        momentum=momentum,
        lr=lr,
        dampening=dampening,
        nesterov=nesterov,
        has_sparse_grad=has_sparse_grad,
        maximize=maximize,
        )

def _subscaf_single_tensor_sgd(params: List[Tensor],
                    d_p_list: List[Tensor],
                    momentum_buffer_list: List[Optional[Tensor]],
                    tau,
                    lbd,
                    is_comp: bool = False,
                    *,
                    weight_decay: float,
                    momentum: float,
                    lr: float,
                    dampening: float,
                    nesterov: bool,
                    maximize: bool,
                    has_sparse_grad: bool):

    for i, param in enumerate(params):
        d_p = d_p_list[i] if not maximize else -d_p_list[i]

        if weight_decay != 0:
            d_p = d_p.add(param, alpha=weight_decay)

        if is_comp == True:
            # r / m coefficient
            if len(lbd[i].shape) == 2:
                m, r = lbd[i].shape
            elif len(lbd[i].shape) == 4:
                m, r  = lbd[i].shape[-2:]
            else:
                assert False, "The shape of lambda is not support."
            #m = lbd[i].in_features
            d_p.mul_(r / m)
            
        if momentum != 0:
            buf = momentum_buffer_list[i]

            if buf is None:
                buf = torch.clone(d_p).detach()
                momentum_buffer_list[i] = buf
            else:
                buf.mul_(momentum).add_(d_p, alpha=1 - dampening)
            
            if nesterov:
                d_p = d_p.add(buf, alpha=momentum)
            else:
                d_p = buf
        if is_comp == True:
            d_p.add_(lbd[i] / (lr * tau))

        param.add_(d_p, alpha=-lr)


def _subscaf_multi_tensor_sgd(params: List[Tensor],
                    grads: List[Tensor],
                    momentum_buffer_list: List[Optional[Tensor]],
                    tau,
                    lbd,
                    is_comp: bool = False,
                    *,
                    weight_decay: float,
                    momentum: float,
                    lr: float,
                    dampening: float,
                    nesterov: bool,
                    maximize: bool,
                    has_sparse_grad: bool):

    if len(params) == 0:
        return
    if is_comp:
        grouped_tensors = Optimizer._group_tensors_by_device_and_dtype([params, 
                                                                        grads, 
                                                                        momentum_buffer_list, 
                                                                        lbd], with_indices=True)
    else:
        grouped_tensors = Optimizer._group_tensors_by_device_and_dtype([params, 
                                                                        grads, 
                                                                        momentum_buffer_list], with_indices=True)
    for item in grouped_tensors.values():
        if is_comp:
            ((device_params, device_grads, device_momentum_buffer_list, lbd), indices) = item
        else:
            ((device_params, device_grads, device_momentum_buffer_list), indices) = item
        device_has_sparse_grad = has_sparse_grad and any(grad.is_sparse for grad in device_grads)

        if maximize:
            device_grads = torch._foreach_neg(device_grads)

        if weight_decay != 0:
            # Re-use the intermediate memory (device_grads) already allocated for maximize
            if maximize:
                torch._foreach_add_(device_grads, device_params, alpha=weight_decay)
            else:
                device_grads = torch._foreach_add(device_grads, device_params, alpha=weight_decay)
        
        if is_comp == True:
            m, r = lbd[0].shape
            # r/m coffeicient
            torch._foreach_mul_(device_grads, r / m)

        if momentum != 0:
            bufs = []

            all_states_with_momentum_buffer = True
            for i in range(len(device_momentum_buffer_list)):
                if device_momentum_buffer_list[i] is None:
                    all_states_with_momentum_buffer = False
                    break
                else:
                    bufs.append(device_momentum_buffer_list[i])

            if all_states_with_momentum_buffer:
                torch._foreach_mul_(bufs, momentum)
                torch._foreach_add_(bufs, device_grads, alpha=1 - dampening)
            else:
                bufs = []
                for i in range(len(device_momentum_buffer_list)):
                    if device_momentum_buffer_list[i] is None:
                        buf = device_momentum_buffer_list[i] = momentum_buffer_list[indices[i]] = \
                            torch.clone(device_grads[i]).detach()
                    else:
                        buf = device_momentum_buffer_list[i]
                        buf.mul_(momentum).add_(device_grads[i], alpha=1 - dampening)

                    bufs.append(buf)

            if nesterov:
                torch._foreach_add_(device_grads, bufs, alpha=momentum)
            else:
                device_grads = bufs
        # carry out subspace scaffold
        if is_comp:
            # HACK revise to avoid repeat computation
            scaled_lbd = torch._foreach_div(lbd, lr * tau)
            torch._foreach_add_(device_grads, scaled_lbd, alpha=1)
        if not device_has_sparse_grad:
            torch._foreach_add_(device_params, device_grads, alpha=-lr)
        else:
            # foreach APIs don't support sparse
            for i in range(len(device_params)):
                device_params[i].add_(device_grads[i], alpha=-lr)