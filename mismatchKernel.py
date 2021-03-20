import numpy as np
from numba import prange

def m_mismatch(s, m=1 , no_touch_ind=-1):
  #S_gamma_k 
  '''
  function to make at most m changes in the string s
  input:
    - s : string
    - m : number of changes (=0 or =1 or = 2)
    - no_touch_ind: index in s that we musn't change
  output:
    - list of possible generated words with the indices of changes
  '''
  L0=[]
  L1=[]
  L2=[]
  if m==0:
    L0.append([s,-1])
    return L0, L1,L2
  if m==1:  
    for i in range(len(s)):
      if i==no_touch_ind:
        continue
      for c in ['A', 'C', 'G', 'T']:
        if s[i]==c:
          continue
        else:
          ch=s[:i]+ c + s[i+1:] 
          L1.append([ch,i])
    L0 = m_mismatch(s, 0)[0]
    return L0,L1,L2
        
  if m==2:
    L0,L1 = m_mismatch(s, 1)[0] , m_mismatch(s, 1)[1]
    for i in range(len(L1)):
      word, ind = L1[i][0] , L1[i][1]
      L21= m_mismatch(word, 1 , no_touch_ind = ind )[1]
      L2.append( L21 )
    return L0, L1,L2

def generate_spectrum(x1 , x2,k):
  d1 = dict()
  d2 = dict()
  for i in range(len(x1)-k):
    if x1[i:i+k] in d1.keys():
      d1[x1[i:i+k]] += 1
    else:
      d1[x1[i:i+k]] = 1
    if x1[i:i+k] not in d2.keys():
      d2[x1[i:i+k]] = 0
    
  for i in range(len(x2)-k):
    if x2[i:i+k] in d2.keys():
      d2[x2[i:i+k]] +=1
    else:
      d2[x2[i:i+k]] =1
    if x2[i:i+k] not in d1.keys():
      d1[x2[i:i+k]] = 0
  
  return d1, d2

def mismatch_coefficient(gamma, x , m,k, d):
  L0,L1,L2 = m_mismatch(gamma, m , no_touch_ind=-1)
  L0 = [L0[0][0]]
  L1 = [L[0] for L in L1 ]
  L=[]
  for i in prange( len(L2) ):
    for j in prange( len(L2[0]) ):
      L.append(L2[i][j][0])
  L2=L
  L = np.concatenate([L0,L1,L2])
  res = 0
  for g in L:
    if g in d.keys():
      res += d[g]
  return res 

def mismatchKernel(x1,x2,m=1,size=5):
  d1, d2 = generate_spectrum(x1,x2,size)
  spectrum = list(d1.keys())
  res = 0
  for gamma in spectrum:
    res+= mismatch_coefficient(gamma, x1 , m,size, d1) * mismatch_coefficient(gamma, x2 , m,size, d2)
  return res