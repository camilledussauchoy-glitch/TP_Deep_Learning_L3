import torch
from torch import nn


class FastGradientSignMethod:

    """
    Implements the Fast Gradient Sign Method (FGSM) attack for adversarial example generation.

    Attributes:
        model (torch.nn.Module): The model to attack.
        eps (float): Maximum perturbation (L-infinity norm bound).
    """
    def __init__(self, model, eps):
        """
        Initializes the FGSM attack.

        Args:
            model (torch.nn.Module): The model to attack.
            eps (float): Maximum L-infinity norm of the perturbation.
        """
        self.model = model
        self.eps = eps
        self.name = f'FGSM_{eps:.2e}'

    def compute(self, x, y):
        """
        Generates an adversarial perturbation using FGSM.

        Args:
            x (torch.Tensor): Original input images.
            y (torch.Tensor): True labels for the input images.

         Returns:
            torch.Tensor: The computed adversarial perturbations.
        """
        # initialize the perturbation delta to zero, and require gradient for optimization
        delta = torch.zeros_like(x, requires_grad=True)

        # get model output and compute loss (cross-entropy)
        loss = nn.CrossEntropyLoss()(self.model(x + delta), y)
        loss.backward()

        ## To do 12
        #apply one step of sign gradient ascent to the input
        perturbation = self.eps * delta.grad.data.sign()
        
        #raise NotImplementedError 
        return perturbation

class ProjectedGradientDescent:
    """
    Implements the Projected Gradient Descent (PGD) attack in L-infinity norm for adversarial example generation.

    Attributes:
        model (torch.nn.Module): The model to attack.
        eps (float): Maximum perturbation (L-infinity norm bound).
        alpha (float): Step size for each iteration.
        num_iter (int): Number of iterations for the attack.
    """
    def __init__(self, model, eps, alpha, num_iter):
        """
        Initializes the PGD attack.

        Args:
            model (torch.nn.Module): The model to attack.
            eps (float): Maximum L-infinity norm of the perturbation.
            alpha (float): Step size for the attack.
            num_iter (int): Number of attack iterations.
        """
        self.model = model
        self.eps = eps
        self.num_iter = num_iter

        ## To do 19
        if alpha is None : 
            self.alpha = eps / num_iter 
        else : 
            self.alpha = alpha
        self.name = f'PGDLinf_{eps:.2e}_{alpha:.2e}_{num_iter}'


    def compute(self, x, y):
        """
        Generates an adversarial perturbation using PGD with L2 norm.

        Args:
            x (torch.Tensor): Original input images.
            y (torch.Tensor): True labels for the input images.

        Returns:
            torch.Tensor: The computed adversarial perturbations.
        """
        # initialize the perturbation delta to zero, and require gradient for optimization
        delta = torch.zeros_like(x, requires_grad=True)

        for t in range(self.num_iter):
            
            # iteratively compute adversarial perturbations
            outputs = self.model(x + delta)
            loss = nn.CrossEntropyLoss()(outputs, y)
            loss.backward()
            # Apply one step of sign gradient ascent to the input
            delta.data = delta + self.alpha * delta.grad.data.sign()
            # Project the perturbation to be within the L-infinity norm ball of radius eps
            delta.data = torch.clamp(delta.data, -self.eps, self.eps)
            ## To do 16 
            break
            # Reset the gradients of the model
            delta.grad.zero_()

        return delta.detach()
