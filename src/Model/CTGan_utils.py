from torch import nn, autograd
import torch



def format_time(seconds):
    minutes = int((seconds % 3600) // 60)
    seconds_int = int(seconds % 60)
    milliseconds = int((seconds % 1) * 1000)
    return f"{minutes:02d}min:{seconds_int:02d}s:{milliseconds:03d}ms"

def format_time_with_h(seconds):
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    seconds_int = int(seconds % 60)
    milliseconds = int((seconds % 1) * 1000)
    return f"{hours:02d}h:{minutes:02d}min:{seconds_int:02d}s:{milliseconds:03d}ms"


def get_nolin_akt(name : str):
    if name == "relu":
        return nn.ReLU()
    elif name == "leaky_relu":
        return nn.LeakyReLU()
    elif name == "sigmoid":
        return nn.Sigmoid()
    elif name == "tanh":
        return nn.Tanh()
    elif name == "softmax":
        return nn.Softmax()
    else:
        raise ValueError(f"Unknown activation function: {name}")
    

def get_loss_function(name : str):
    if name == "cross_entropy":
        return nn.CrossEntropyLoss()
    elif name == "mse":
        return nn.MSELoss()
    else:
        raise ValueError(f"Unknown loss function: {name}")


def gradient_penalty(discriminator, X_real_cond, X_fake_cond, device):
    bs = X_real_cond.size(0)

    alpha = torch.rand(bs, 1, device=device)
    alpha = alpha.expand_as(X_real_cond)  

    X_interpolated = alpha * X_real_cond.detach() + (1 - alpha) * X_fake_cond.detach()
    X_interpolated.requires_grad_(True)
    
    y_hat_interpolated = discriminator(X_interpolated)
    
    grad_outputs = torch.ones_like(y_hat_interpolated, device=device)
    gradients = autograd.grad(
        outputs=y_hat_interpolated,
        inputs=X_interpolated,
        grad_outputs=grad_outputs,
        create_graph=True,
        retain_graph=True
    )[0]

    gradients = gradients.view(bs, -1)
    gradient_norm = gradients.norm(2, dim=1)

    gradient_penalty = ((gradient_norm - 1) ** 2).mean()

    return gradient_penalty

def wasserstein_loss(y_real, y_fake):
    return torch.mean(y_fake) - torch.mean(y_real)