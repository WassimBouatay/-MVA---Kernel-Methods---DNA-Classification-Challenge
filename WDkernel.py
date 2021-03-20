from numba import njit, prange

@njit
def WD_kernel(x1, x2, d=3):
  res = 0
  for k in prange(1,d+1):
    res_k = 0
    for i in prange(len(x1)-k):
      if x1[i:i+k]==x2[i:i+k]:
        res_k += 1
    beta_k = 2*(d-k+1)/(d*(d+1))
    res += beta_k*res_k
  return res