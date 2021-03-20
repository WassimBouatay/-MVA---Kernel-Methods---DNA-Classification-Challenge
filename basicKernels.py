import numpy as np

def linear_kernel(x,x_hat):
  return float(x @ x_hat.T)

def rbf_kernel(x,x_hat, var = 1):
  return np.exp(-np.linalg.norm(x-x_hat)/(2*var))

def poly_kernel(x,x_hat,alpha = 1 , c=0 , d = 2):
  return (alpha * float(x @ x_hat.T) + c)**d