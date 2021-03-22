import numpy as np
from numba import njit, prange
       
@njit
def SW_kernel(x1,x2, w = 0.02, s = 0.03):
      x1 = [char for char in x1] 
      x2 = [char for char in x2] 
      
      n1 = len(x1)
      n2 = len(x2)

      S = np.zeros((n1,n2))

      for i in prange(n1):
        for j in prange(n2):
          if x1[i] == x2[j]:
            S[i,j] = s
          else:
            S[i,j] = -s

      H = np.zeros((n1+1,n2+1))
      """
      # original version 

      max1 = np.zeros(n1 +1)

      for j in range(1,n2+1):
        for t in prange(n1+1):
          max1[t] = np.max(np.array([max1[t],H[t,j-1]])) - w
        max2  = 0
        for i in range(1,n1+1):
          max2 = np.max(np.array([H[i-1,j],max2])) - w 
          H[i,j] = np.max(np.array([ H[i-1,j-1] + S[i-1,j-1] , max2 , max1[i], 0]))
      """
      # faster version
      for j in range(1,n1+1):
        for i in range(1,n2+1):
          H[i,j] = np.max(np.array([ H[i-1,j-1]+S[i-1,j-1], H[i-1,j]-w, H[i,j-1]-w, 0 ]))
      return np.max(H)

@njit
def SW_K_Mat(X,w = 0.02, s = 0.03 ):
    n = len(X)
    K = np.zeros((n,n))
    for h in prange(n):
      for l in prange(h , n):
        x1, x2 = X[h], X[l]
        K[h,l] = SW_kernel(x1,x2, w = w, s = s)
        K[l,h] = K[h,l]
    return K

