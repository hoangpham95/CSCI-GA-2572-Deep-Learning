from typing import Any
from torch.func import grad
import torch

from collections.abc import Mapping

class MLP:
  def __init__(
    self,
    linear_1_in_features: int,
    linear_1_out_features: int,
    f_function: str,
    linear_2_in_features: int,
    linear_2_out_features: int,
    g_function: str
  ):
    """
    Args:
        linear_1_in_features: the in features of first linear layer
        linear_1_out_features: the out features of first linear layer
        linear_2_in_features: the in features of second linear layer
        linear_2_out_features: the out features of second linear layer
        f_function: string for the f function: relu | sigmoid | identity
        g_function: string for the g function: relu | sigmoid | identity
    """
    self.f_function = f_function
    self.g_function = g_function

    self.parameters: Mapping[str, torch.Tensor] = dict(
        W1 = torch.randn(linear_1_out_features, linear_1_in_features),
        # torch.randn with single parameter produce a row vector. use (size, 1)
        # to produce a column vector.
        b1 = torch.randn(linear_1_out_features),
        
        W2 = torch.randn(linear_2_out_features, linear_2_in_features),
        # torch.randn with single parameter produce a row vector. use (size, 1)
        # to produce a column vector. 
        b2 = torch.randn(linear_2_out_features),
    )
    self.grads: Mapping[str, torch.Tensor] = dict(
        dJdW1 = torch.zeros(linear_1_out_features, linear_1_in_features),
        dJdb1 = torch.zeros(linear_1_out_features),
        dJdW2 = torch.zeros(linear_2_out_features, linear_2_in_features),
        dJdb2 = torch.zeros(linear_2_out_features),
    )
    
    self.activation: Mapping[str, torch.Module] = dict(
      relu = torch.nn.ReLU(),
      sigmoid = torch.nn.Sigmoid(),
      identity = torch.nn.Identity()
    )
    
    self.derivative: Mapping[str, Any] = dict(
      relu = lambda x: 1.0 * (x > 0.0),
      sigmoid = lambda x: torch.exp(-x) / (1.0 + torch.exp(-x))**2,
      identity = lambda x: 1.0
    )

    # put all the cache value you need in self.cache
    self.cache = dict()
   
    # caching the size of the model 
    self.in_features = linear_1_in_features
    self.out_features = linear_2_out_features
    self.mid_features = linear_1_out_features
    
    assert self.mid_features == linear_2_in_features
    

  def forward(self, x: torch.Tensor):
    """
    Args:
        x: tensor shape (batch_size, linear_1_in_features)
    """
    self.cache["batch_sz"] = x.shape[0]
    params = self.parameters
 
    f = self.activation[self.f_function]
    g = self.activation[self.g_function]
  
    # Unpacking parameters 
    W1, W2, b1, b2 = params["W1"], params["W2"], params["b1"], params["b2"]
   
    # Forwarding input. Each row in x is a feature. 
    # The output of the first layer is W1 * x.T. Need to transpose b1 and b2 to
    # match with the output as well.
    z1 = W1 @ x.T + b1.unsqueeze(1)
    z2 = f(z1)
    z3 = W2 @ z2 + b2.unsqueeze(1)
    y_hat = g(z3)
    
    # Persists the value to calculate gradient
    self.cache["x"] = x
    self.cache["z1"] = z1
    self.cache["z2"] = z2
    self.cache["z3"] = z3
   
    # output needs to be size batch x output, so transpose here
    return y_hat.T 
    
  
  def backward(self, dJdy_hat):
    """
    Args:
        dJdy_hat: The gradient tensor of shape (batch_size, 
        linear_2_out_features)
    """
    df = self.derivative[self.f_function]
    dg = self.derivative[self.g_function]
    
    # We cannot use dy_hatdz3 since there's no mathematical concept of partial
    # derivative of matrix wrt to matrix, so calculate dy_hatdb2 directly.
    dy_hatdb2 = dg(self.cache["z3"].T)
    
    # identity returns 1, other functions return tensor for dy_hatdb2. using *
    # will also override tensor multiplication. Will need to use torch.sum()
    # to returns the correct size of b2.
    dJdb2 = dJdy_hat * dy_hatdb2 
    
    dJdW2 = dJdb2.T @ self.cache["z2"].T
    dJdz2 = dJdb2 @ self.parameters["W2"]
    
    # identity returns 1, other functions return tensor for dy_hatdb2. using *
    # will also override tensor multiplication. Will need to use torch.sum()
    # to returns the correct size of b1.
    dJdb1 = dJdz2 * df(self.cache["z1"].T)
    dJdW1 = dJdb1.T @ self.cache["x"]
    
    self.grads["dJdb2"] = torch.sum(dJdb2, dim=0)
    self.grads["dJdW2"] = dJdW2
    self.grads["dJdb1"] = torch.sum(dJdb1, dim=0)
    self.grads["dJdW1"] = dJdW1
    
 
  def clear_grad_and_cache(self):
    for grad in self.grads:
        self.grads[grad].zero_()
    self.cache = dict()
 
    
################################################################################
# Loss functions
################################################################################
def mse_loss(y, y_hat):
  """
  Args:
      y: the label tensor (batch_size, linear_2_out_features)
      y_hat: the prediction tensor (batch_size, linear_2_out_features)

  Return:
      J: scalar of loss
      dJdy_hat: The gradient tensor of shape (batch_size, linear_2_out_features)
  """
  J = torch.nn.functional.mse_loss(y_hat, y)
  
  # Need to use numel for the entire batch loss avg
  dJdy_hat = 2./torch.numel(y) * (y_hat - y)
   
  return J, dJdy_hat 
  

def bce_loss(y, y_hat):
  """
  Args:
      y_hat: the prediction tensor
      y: the label tensor
      
  Return:
      loss: scalar of loss
      dJdy_hat: The gradient tensor of shape (batch_size, linear_2_out_features)
  """
  eps = 1e-100
  J = torch.nn.functional.binary_cross_entropy(y_hat, y)
  n = torch.numel(y)
  
  # Need to use numel for entire batch loss avg. Using epsilon to avoid 
  # undefined gradient.
  dJdy_hat = -1./n * (
    y/(y_hat + eps) - (1. - y)/(1. - y_hat + eps)
  )

  return J, dJdy_hat
  