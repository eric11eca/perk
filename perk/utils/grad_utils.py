import torch

def replace_none_with_zero(tensor_list, reference):
    out = []
    for t, r in zip(tensor_list, reference):
        fixed = t if t is not None else torch.zeros_like(r)
        out.append(fixed)
    return tuple(out)

def to_vec(tensor_list, alpha=1.0):
        return torch.cat(
            [alpha * t.reshape(-1) for t in tensor_list])

def grad(loss, parameters, retain_graph=False, allow_unused=False):
    return torch.autograd.grad(
        loss, parameters,
        retain_graph=retain_graph,
        allow_unused=allow_unused)

def get_opt_param_group_for_param(param, optimizer):
    """
    Get optimizer param_group for specific parameter

    :param param: Parameter for which optimizer param_group is inquired
    :param optimizer: Optimizer instance
    :type param: torch.nn.Parameter
    :return: param_group for the given parameter
    :rtype: dict
    """
    param_groups = optimizer.param_groups
    for group in param_groups:
        if group["name"] == param:
            return group

def get_opt_state_for_param(param, optimizer):
    """
    Get optimizer state for specific parameter

    :param param: Parameter for which optimizer state is inquired
    :param optimizer: Optimizer instance
    :type param: torch.nn.Parameter
    :return: state for the given parameter
    :rtype: dict
    """
    for state in optimizer.state:
        if param in state:
            return state[param]

def adapted_params(model, peft=True):
    if peft:
        params = [
            p for n, p in model.named_parameters()
            if "lora" in n
        ]
    else:
        params = list(model.parameters())
    return params

def meta_base_params(model):
    return [
        p for n, p in model.named_parameters()
        if "lora" not in n
    ]

def set_grads(params, grads):
    """
    Set gradients for trainable parameters. ``params.grad = grads``

    :param params: Trainable parameters
    :type params: Sequence of Tensor
    :param grads: Calculated gradient
    :type grads: Sequence of Tensor
    """
    grad_norm = 0
    for param, grad in zip(params, grads):
        if grad is not None:
            if hasattr(param, "grad") and param.grad is not None:
                param.grad = param.grad + grad.detach()
            else:
                param.grad = grad.detach()
            grad_norm += torch.sum(grad**2)
    grad_norm = grad_norm ** (1/2)
    return grad_norm

def count_grads(params):
    num_none_grad = 0
    num_grad = 0
    num_require_grad = 0
    num_freeze = 0
    for p in params:
        if p.grad is not None:
            num_grad += 1
        else:
            num_none_grad += 1

        if p.requires_grad:
            num_require_grad += 1
        else:
            num_freeze += 1
    print(f"# of grads: {num_grad}")
    print(f"# of None grads: {num_none_grad}")
    print(f"# of require_grad: {num_require_grad}")
    print(f"# of freeze: {num_freeze}")

def neg_with_none(a):
    if a is None:
        return None
    else:
        return -a