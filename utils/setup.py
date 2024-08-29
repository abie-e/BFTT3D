from copy import deepcopy
import torch.jit
import torch.optim as optim
import torch
import torch.nn as nn


def setup_tent_shot(args, model):
    """Set up tent adaptation.

    Configure the model for training + feature modulation by batch statistics,
    collect the parameters for feature modulation by gradient optimization,
    set up the optimizer, and then tent the model.
    """
    model = configure_model(args, model) 
    params, param_names = collect_params(model, args)  
    optimizer = setup_optimizer(args, params)  
    return model, optimizer

def collect_stats_bn(model):
    """Collect the normalization stats from batch norms.

    Walk the model's modules and collect all batch normalization stats.
    Return the stats and their names.
    """
    stats = []
    names = []
    for nm, m in model.named_modules():
        if isinstance(m, nn.BatchNorm2d):
            state = m.state_dict()
            if m.affine:
                del state['weight'], state['bias']
            for ns, s in state.items():
                stats.append(s)
                names.append(f"{nm}.{ns}")
    return stats, names

def setup_BFTT3D(model):
    return model


def setup_optimizer(tent_args, params):
    """Set up optimizer for tent adaptation.

    Tent needs an optimizer for test-time entropy minimization.
    In principle, tent could make use of any gradient optimizer.
    In practice, we advise choosing Adam or SGD+momentum.
    For optimization settings, we advise to use the settings from the end of
    trainig, if known, or start with a low learning rate (like 0.001) if not.

    For best results, try tuning the learning rate and batch size.
    """
    
    # return optim.SGD(params, lr=0.0001, momentum=0.9)
    return optim.Adam(params, lr=1e-3, betas=(0.9, 0.999), weight_decay=0.)

def Entropy(input_):
    bs = input_.size(0)
    entropy = -input_ * torch.log(input_ + 1e-5)
    entropy = torch.sum(entropy, dim=1)
    return entropy 

@torch.jit.script
def softmax_entropy(x: torch.Tensor) -> torch.Tensor:
    """Entropy of softmax distribution from logits."""
    return -(x.softmax(1) * x.log_softmax(1)).sum(1)


def safe_log(x, ver):
    if ver == 1:
        return torch.log(x + 1e-5)
    elif ver == 2:
        return torch.log(x + 1e-7)
    elif ver == 3:
        return torch.clamp(torch.log(x), min=-100)
    else:
        raise ValueError("safe_log version is not properly defined !!!")


def softmax_diversity_regularizer(x):
    x2 = x.softmax(-1).mean(0)  # [b, c] -> [c]
    return (x2 * safe_log(x2, ver=1)).sum()


@torch.enable_grad()  # ensure grads in possible no grad context for testing
def forward_and_adapt_shot(x, model, optimizer):
    """Forward and adapt model on batch of data.

    Measure entropy of the model prediction, take gradients, and update params.
    """
    # forward
    # (batch * n_views, 3, T, 224,224 )  -> (batch * n_views, n_class ) todo clip-level prediction
    optimizer.zero_grad()
    outputs = model(x)
    loss = 1*(softmax_entropy(outputs).mean(0) - softmax_diversity_regularizer(outputs).mean(0)) 
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()
    return outputs


@torch.enable_grad()  # ensure grads in possible no grad context for testing
def forward_and_adapt_tent(x, model, optimizer):
    """Forward and adapt model on batch of data.

    Measure entropy of the model prediction, take gradients, and update params.
    """
    # forward
    
    # outputs = model.module.classification_only(x, only_unmasked=False)  # (batch * n_views, 3, T, 224,224 )  -> (batch * n_views, n_class ) todo clip-level prediction
    outputs = model(x)
    loss = softmax_entropy(outputs).mean(0)   #   todo compute the entropy for all clip-level predictions   then take the averaga among all samples
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()
    return outputs


def collect_params(model, args):
    """Collect the affine scale + shift parameters from batch norms.

    Walk the model's modules and collect all batch normalization parameters.
    Return the parameters and their names.

    Note: other choices of parameterization are possible!
    """
    params = []
    names = []
    if args.tta == 'tent':
        for nm, m in model.named_modules():
            if isinstance(m, torch.nn.modules.batchnorm._BatchNorm):
                for np, p in m.named_parameters():
                    if np in ['weight', 'bias']:  # weight is scale gamma, bias is shift beta
                        params.append(p)
                        names.append(f"{nm}.{np}")
    if args.tta == 'shot':
        for nm, m in model.named_modules():
            for np, p in m.named_parameters():
                params.append(p)
                names.append(f"{nm}.{np}")
    return params, names


def copy_model_and_optimizer(model, optimizer):
    """Copy the model and optimizer states for resetting after adaptation."""
    model_state = deepcopy(model.state_dict())
    optimizer_state = deepcopy(optimizer.state_dict())
    tta_model = deepcopy(model)
    for param in tta_model.parameters():
        param.detach_()
    return tta_model, model_state, optimizer_state


def load_model_and_optimizer(model, optimizer, model_state, optimizer_state):
    """Restore the model and optimizer states from copies."""
    model.load_state_dict(model_state, strict=True)
    optimizer.load_state_dict(optimizer_state)


def configure_model(model):
    """Configure model for use with tent."""
    model.train()
    model.requires_grad_(False)
    for m in model.modules():
        if isinstance(m, torch.nn.modules.batchnorm._BatchNorm):
            m.requires_grad_(True)
            # m.track_running_stats = True
            m.track_running_stats = False #
            m.running_mean = None 
            m.running_var = None
    return model


def check_model(model):
    """Check model for compatability with tent."""
    is_training = model.training
    assert is_training, "tent needs train mode: call model.train()"
    param_grads = [p.requires_grad for p in model.parameters()]
    has_any_params = any(param_grads)
    has_all_params = all(param_grads)
    assert has_any_params, "tent needs params to update: " \
                           "check which require grad"
    assert not has_all_params, "tent should not update all params: " \
                               "check which require grad"

    has_bn = any([isinstance(m, torch.nn.modules.batchnorm._BatchNorm) for m in model.modules()])
    assert has_bn, "tent needs normalization for its optimization"

