from matplotlib import pyplot as plt
import numpy as np
import torch
from torch import nn
import typing as t


def set_activation_hooks(network: nn.Module) -> t.Tuple[t.Dict, t.Dict]:

    def get_activated_neurons(name):
        def hook(model, input):
            activated_neurons[name] = input[0].detach()
        return hook
    
    
    def get_unactivated_neurons(name):
        def hook(model, input, output):
            unactivated_neurons[name] = output.detach()
        return hook
    
    activated_neurons = {}
    unactivated_neurons = {}
    
    for i, (name, _) in enumerate(network.named_children()):
        getattr(network, name).register_forward_pre_hook(get_activated_neurons(i))
        getattr(network, name).register_forward_hook(get_unactivated_neurons(i))

    return activated_neurons, unactivated_neurons


def get_number_of_layers(network: nn.Module) -> int:
    return len([module for module in network.modules() if not isinstance(module, network.__class__)])



def visualize_layer_activations(fig: plt.Figure, network: nn.Module, 
                     X: torch.Tensor, y: torch.Tensor, 
                     y_labels: t.List, X_labels: t.List = None) -> None:

    if X_labels is None:
        X_labels = []
    
    # Set Hooks for Activation and Non-Activation Layers
    activated_neurons, unactivated_neurons = set_activation_hooks(network=network)

    # Make Prediction
    y_hat = network(X)

    # Create Title based on if prediction is correct or not
    if torch.argmax(y_hat) == y:
        title_text = f"Correct, Prediction: {torch.argmax(y_hat)}, Truth: {y}"
        fig.suptitle(title_text, color="g", fontsize=20)
    else:
        title_text = f"Incorrect, Prediction: {torch.argmax(y_hat)}, Truth: {y}"
        fig.suptitle(title_text, color="r", fontsize=20)

    # Get number of layers to determine grid size required
    N_LAYERS = get_number_of_layers(network=network)
    GRID_SIZE = 1 + 2 * N_LAYERS

    # Create number of axes
    axs = [plt.subplot2grid((1, GRID_SIZE), (0, i), rowspan=1, colspan=1) for i in range(GRID_SIZE)]

    # Show X
    axs[0].imshow(X[0], cmap='gray')
    axs[0].set_xticklabels([])
    axs[0].set_yticklabels(X_labels)

    # Loop over layers of the network
    for i, (name, _) in enumerate(network.named_children()):

        layer = np.rot90(unactivated_neurons[i], k=3, axes=(0, 1))
        if (i + 1) == N_LAYERS:
            layer_activated = np.rot90(y_hat.detach().numpy(), k=3, axes=(0, 1))
        else:
            layer_activated = np.rot90(activated_neurons[i+1], k=3, axes=(0,1))
    
        axs[2 * i + 1].imshow(layer, cmap='RdYlGn')
        axs[2 * (i + 1)].imshow(layer_activated, cmap='YlGn')
        axs[2 * i + 1].set_title(f"L{i} Out")
        axs[2 * (i + 1)].set_title(f"L{i} Activated")
        axs[2 * i + 1].set_xticklabels([])
        axs[2 * (i + 1)].set_xticklabels([])
        axs[2 * i + 1].set_yticklabels([])
    
        if (i + 1) == N_LAYERS:
            axs[2 * (i + 1)].set_yticklabels(y_labels)
            axs[2 * (i + 1)].set_yticks([*range(len(y_labels))])
            axs[2 * (i + 1)].yaxis.set_label_position("right")
            axs[2 * (i + 1)].yaxis.tick_right()
        else:
            axs[2 * (i + 1)].set_yticklabels([])

    return
