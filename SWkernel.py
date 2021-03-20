import numpy as np
from numba import njit, prange
 
"""
@njit
def LA_K_Mat(X,w = 0.02, s = 0.03 ):
    n = len(X)
    K = np.zeros((n,n))
    for h in prange(n):
      for l in prange(h, n):
        x1, x2 = X[h], X[l]

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
        max2 = np.zeros(n1 +1)
      
        for j in range(1,n2+1):
          for t in prange(n1+1):
            max2[t] = np.max(np.array([max2[t],H[t,j-1]])) -w
          max1 = 0
          for i in range(1,n1+1):
            max1 = np.max(np.array([H[i-1,j],max1])) - w 
            H[i,j] = np.max(np.array([ H[i-1,j-1] + S[i-1,j-1] , max1 , max2[i], 0]))
     
        for j in range(1,n1+1):
          for i in range(1,n2+1):
            H[i,j] = np.max(np.array([ H[i-1,j-1]+S[i-1,j-1], H[i-1,j]-w, H[i,j-1]-w, 0 ]))
        
        K[h,l] = np.max(H)
        K[l,h] = K[h,l]
    return K

"""
@njit
def LA_K_Mat(X1,X2,w = 0.02, s = 0.03 ):
    n1 = len(X1)
    n2 = len(X2)
    K = np.zeros((n1,n2))
    for h in prange(n1):
      for l in prange(n2):
        x1, x2 = X1[h], X2[l]
        K[h,l] = LA_kernel(x1,x2, w = 0.02, s = 0.03)
    return K
"""

@njit
def LA_kernel(x1,x2, w = 0.02, s = 0.03):
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
      max2 = np.zeros(n1 +1)


      for j in range(1,n2+1):
        for t in prange(n1+1):
          max2[t] = np.max(np.array([max2[t],H[t,j-1]])) -w
        max1 = 0
        for i in range(1,n1+1):
          max1 = np.max(np.array([H[i-1,j],max1])) - w 
          H[i,j] = np.max(np.array([ H[i-1,j-1] + S[i-1,j-1] , max1 , max2[i], 0]))
      
      for j in range(1,n1+1):
        for i in range(1,n2+1):
          H[i,j] = np.max(np.array([ H[i-1,j-1]+S[i-1,j-1], H[i-1,j]-w, H[i,j-1]-w, 0 ]))
  
      return np.max(H)"""
      
@njit
def LA_kernel(x1,x2, w = 0.02, s = 0.03):
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
      max2 = np.zeros(n1 +1)


      for j in range(1,n2+1):
        for t in prange(n1+1):
          max2[t] = np.max(np.array([max2[t],H[t,j-1]])) -w
        max1 = 0
        for i in range(1,n1+1):
          max1 = np.max(np.array([H[i-1,j],max1])) - w 
          H[i,j] = np.max(np.array([ H[i-1,j-1] + S[i-1,j-1] , max1 , max2[i], 0]))
      """
      for j in range(1,n1+1):
        for i in range(1,n2+1):
          H[i,j] = np.max(np.array([ H[i-1,j-1]+S[i-1,j-1], H[i-1,j]-w, H[i,j-1]-w, 0 ]))
      
  
      return np.max(H)