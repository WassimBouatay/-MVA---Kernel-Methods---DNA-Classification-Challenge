import numpy as np
from numba import prange


def normalize_kernel_matrix(K):
    diag_K=np.diag(K)
    root_K=np.sqrt(diag_K)
    K_inv=1/root_K
    K_inv = np.diag( K_inv )
    return K_inv @ K @ K_inv

def compute_Ker_mat(k, X, m=None, size=5, d=None, spectrum_size=None, normalize=False ):
  K = np.zeros((X.shape[0],X.shape[0]))

  for i in prange(len(X)):
    for j in prange(i, len(X)):
      x1, x2 = X[i], X[j]
      if spectrum_size != None:
        K[i,j] = k(x1,x2,spectrum_size)
        K[j,i] = K[i,j]
      elif d !=None:
        K[i,j] = k(x1,x2,d)
        K[j,i] = K[i,j]
      elif m!=None:
        K[i,j] = k(x1,x2, m,size)
        K[j,i] = K[i,j]
      else: 
        K[i,j] = k(x1,x2)
        K[j,i] = K[i,j]
  if normalize:
    return  1/10*normalize_kernel_matrix(K)
  else:
    return K