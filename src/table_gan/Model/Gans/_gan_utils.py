from torch import nn, autograd
import torch
import matplotlib.pyplot as plt


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


def get_nolin_act(name : str):
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
    return torch.mean(y_real) - torch.mean(y_fake)


def find_w(input_size):
    n = 0
    while True:
        w = 2 * n 
        if w ** 2 > input_size:
            extra = w**2 - input_size  
            return w, extra
        n += 1


def init_weights(m):
    if isinstance(m, nn.Linear):
        nn.init.xavier_uniform_(m.weight)  
        if m.bias is not None:
            nn.init.zeros_(m.bias)  
    elif isinstance(m, nn.Conv2d):
        nn.init.kaiming_normal_(m.weight, a=0.2, nonlinearity='leaky_relu')  # Kaiming initialization for Conv2d layers
        if m.bias is not None:
            nn.init.zeros_(m.bias)


def plot_gan_losses(crit_loss, gen_loss,
                    critic_color='blue',
                    gen_color='orange',  
                    save_path=None
    ):
    fig, ax = plt.subplots(figsize=(10, 6))

    ax.plot(crit_loss, label="Critic Loss", color=critic_color, linestyle='-', linewidth=1.5)
    ax.plot(gen_loss, label="Generator Loss", color=gen_color, linestyle='-', linewidth=1.5)
  
    ax.set_xlabel("Epochs")
    ax.set_ylabel("Loss")
    ax.set_title("Train and Validation Losses for Generator and Critic")
    ax.legend()

    if save_path:
        fig.savefig(save_path)

    return fig