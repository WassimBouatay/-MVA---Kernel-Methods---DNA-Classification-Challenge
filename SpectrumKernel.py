import numpy as np

def spectrum_kernel(x1, x2, spectrum_size=8):
  d1 = dict()
  d2 = dict()
  for k in range(1, spectrum_size+1):
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
    
  phi1 = np.array(list(d1.values()))
  phi2 = np.array(list(d2.values()))
  return phi1 @ phi2.T / np.sqrt((np.sum(phi1**2)*np.sum(phi2**2)))