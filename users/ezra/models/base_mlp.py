from typing import Callable, Type, Sequence, Tuple

import torch
from torch import nn
from torch.optim import Optimizer

from users.ezra.models.wrappers import PyTorchWrapper


class _mlp(nn.Module):
    def __init__(self, layer_sizes: Sequence[int], activation: nn.Module = nn.ReLU):
        """
        Initializes a simple Multilayer Perceptron from a list of layer sizes. The first layer is the input dimension
        and the last layer is the output dimension. Activation functions are inserted after each layer except the last

        Args:
            layer_sizes (Sequence[int]): List containing the sizes of each layer.
            activation (nn.Module): Activation function

        """
        super().__init__()
        layers = []
        for i in range(len(layer_sizes) - 1):
            layers.append(nn.Linear(layer_sizes[i], layer_sizes[i + 1]))
            if i < len(layer_sizes) - 2:
                layers.append(activation())

        self.model = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the MLP.

        Args:
            x (Tensor): Input tensor.

        Returns:
            Tensor: Output tensor.
        """
        return self.model(x)


class BaseMLP(PyTorchWrapper):
    """
    A simple Multilayer Perceptron implementation wrapped in PyTorchWrapper
    """
    def __init__(self, model_parameters: dict, optimizer_parameters: dict, optimizer: Type[Optimizer],
                 loss_function: Callable[[torch.Tensor, torch.Tensor], torch.tensor],
                 training_parameters: Sequence[int, int, str] = (1, 8, "cpu")):
        """
        Initializes a wrapped Multilayer Perceptron
        Args:
            model_parameters (dict): Parameters to initialize the MLP
            optimizer_parameters (dict): Parameters to initialize the optimizer
            optimizer (Type[Optimizer]): Which optimizer to use
            loss_function: criterion to optimize on
            training_parameters (Tuple[int, int, str]): Epoch, Batch Size, Device
        """
        model = _mlp(**model_parameters)
        optimizer = optimizer(model.parameters(), **optimizer_parameters)
        super().__init__(model, loss_function, optimizer, training_parameters)

