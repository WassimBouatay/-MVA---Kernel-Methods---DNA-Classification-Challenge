import numpy as np
import numba as nb
from cvxopt import matrix, solvers
from cvxopt.solvers import qp
from scipy.optimize import minimize
from numba import njit, prange
import time 
from SWkernel import LA_K_Mat, LA_kernel
from SpectrumKernel import spectrum_kernel
from mismatchKernel import mismatchKernel
from WDkernel import WD_kernel
from basicKernels import linear_kernel, rbf_kernel, poly_kernel
from CreateKernelMatrix import compute_Ker_mat


solvers.options['show_progress'] = False


def log_rg_loss(x):
  return np.log(1 + np.exp(-x))


class Ridge_Classifier():
  def  __init__(self,lam = 0.1, kernel_name = 'linear',Kernel_mat = None , kernel = linear_kernel, 
                spectrum_size=8, 
                loss_func = log_rg_loss , m=0, size=4, d=3):
    super(Ridge_Classifier ,self).__init__()

    self.kernel_name = kernel_name
    self.kernel = kernel
    self.loss_func = loss_func
    self.lam = lam
    self.alpha = None
    self.K = Kernel_mat
    self.data = None
    self.spectrum_size = spectrum_size
    self.size = size
    self.m = m
    self.d = d

  def fit(self, data, Y):

    L = Y.copy()
    L[L==0] = -1
    self.data = data.copy()

    if (self.K == None).all():

      if self.kernel_name == 'linear' :
        self.K = self.data @ self.data.T

      elif self.kernel_name == 'rbf':
        var = 1
        self.K = np.zeros((self.data.shape[0], self.data.shape[0]))
        for i , x in enumerate(self.data):
            self.K[i,:] = np.linalg.norm(self.data-x,axis=1)**2
        self.K = np.exp(-self.K/(2*var))

      elif self.kernel_name == 'WD_kernel':
        self.K = compute_Ker_mat(self.kernel, self.data, d=self.d)
      elif self.kernel_name == 'spectrum_kernel':
        self.K = compute_Ker_mat(self.kernel, self.data, spectrum_size=self.spectrum_size, normalize=False)
      elif self.kernel_name == 'mismatchKernel':
        self.K = compute_Ker_mat(self.kernel, self.data, m=self.m , size =self.size)
      elif self.kernel_name == 'LA_kernel':
        listed = nb.typed.List(self.data)
        self.K = LA_K_Mat(listed)

      else:
        self.K = compute_Ker_mat(self.kernel, self.data)

    print('Kernel computed')
    
    f = lambda alpha : np.mean(self.loss_func(L * (self.K @ alpha.T)))  + self.lam * alpha @ self.K @ alpha.T

    self.alpha = minimize(f, np.zeros(self.data.shape[0]))['x']

  def predict(self, X , predict_train=False):
    f = []

    for x in X:
      f_x = 0
      for i in prange(self.alpha.shape[0]):
        if self.kernel_name == 'spectrum_kernel':
          f_x += self.kernel(x ,self.data[i], spectrum_size=self.spectrum_size) * self.alpha[i]
        elif self.kernel_name == 'WD_kernel':
          f_x += self.kernel(x ,self.data[i], d=self.d) * self.alpha[i]
        elif self.kernel_name =='LA_kernel':
          y = self.data[i]
          f_x += self.kernel(x,y) * self.alpha[i]

        else:
          f_x += self.kernel(x ,self.data[i]) * self.alpha[i]
      
      if f_x > 0:
        f.append(1)
      else:
        f.append(0)
    if predict_train == True:
      f_train = np.maximum(0, np.sign(self.K @ self.alpha))
      return f , f_train
    else:
      return f
  


class SVM():
  def  __init__(self, kernel_name = None, kernel=linear_kernel , spectrum_size=8, C=1.0, m=1 , size=5 , scale=1, d=3):
    super(SVM ,self).__init__()

    self.kernel_name = kernel_name
    self.kernel = kernel
    self.C = C
    self.alpha = None
    self.w = None
    self.bias = None
    self.K = None
    self.data = None
    self.scale = scale

    self.spectrum_size = spectrum_size
    self.size = size
    self.m = m
    self.d = d

  def fit(self, data, Y , Kernel_train = None):
    L = Y.copy()
    L[L==0] = -1
    if not (data is None):
      N = len(data)
      self.data = data.copy()
    if not (Kernel_train is None):
      N = len(Kernel_train)
      self.K = Kernel_train.copy()

    if Kernel_train is None:
      start = time.time()
      if self.kernel_name == 'linear' :
        self.K = self.data @ self.data.T

      elif self.kernel_name == 'rbf':
        var = 1
        self.K = np.zeros((self.data.shape[0], self.data.shape[0]))
        for i , x in enumerate(self.data):
            self.K[i,:] = np.linalg.norm(self.data-x,axis=1)**2
        self.K = np.exp(-self.K/(2*var))

      elif self.kernel_name == 'WD_kernel':
        self.K = compute_Ker_mat(self.kernel, self.data, d=self.d)
      elif self.kernel_name == 'spectrum_kernel':
        self.K = compute_Ker_mat(self.kernel, self.data, spectrum_size=self.spectrum_size, normalize=False)
      elif self.kernel_name == 'mismatchKernel':
        self.K = compute_Ker_mat(self.kernel, self.data, m=self.m , size =self.size)
      elif self.kernel_name == 'LA_kernel':
        listed = nb.typed.List(self.data)
        self.K = LA_K_Mat(listed)

      else:
        self.K = compute_Ker_mat(self.kernel, self.data)

      print('Kernel computed')
      print("time:", time.time()-start)    

    self.K = self.K/self.scale

    P = matrix(self.K)
    q = matrix(-L, (N, 1), 'd')
    G = matrix(np.vstack((np.diag(L), -np.diag(L))), (2*N, N), 'd')
    h = matrix(np.vstack((np.ones((N, 1))*self.C, np.zeros((N, 1)))))
    sol = qp(P, q, G,	h)

    self.alpha = np.array(sol['x']).reshape(N)

  def predict(self, X, predict_train = False ,Kernel_val_train= None):
    f = []
    if Kernel_val_train is None:
      for x in X:
        f_x = 0
        for i in range(self.alpha.shape[0]):
          if self.kernel_name == 'spectrum_kernel':
            f_x += self.kernel(x ,self.data[i], spectrum_size=self.spectrum_size) * self.alpha[i]
          elif self.kernel_name == 'WD_kernel':
            f_x += self.kernel(x ,self.data[i], d=self.d) * self.alpha[i]
          elif self.kernel_name == 'mismatchKernel':
            f_x += self.kernel(x ,self.data[i], m=self.m, size = self.size) * self.alpha[i]
          elif self.kernel_name =='LA_kernel':
            y = self.data[i]
            f_x += self.kernel(x,y) * self.alpha[i]
          else:
            f_x += self.kernel(x ,self.data[i]) * self.alpha[i]
        if f_x > 0:
          f.append(1)
        else:
          f.append(0)
    else:
      f = np.maximum(0, np.sign(Kernel_val_train @ self.alpha))
    if predict_train == True:
      f_train = np.maximum(0, np.sign(self.K @ self.alpha))
      return f , f_train
    else:
      return f
  
  
  '''def predict_mismatch(self,X_val,n , normalize = True):
    l = len(self.alpha)
    f = np.zeros(n)
    kernel_val = np.zeros((n,l))
    for i in range(n):
      x = X_val[i]
      for j in range(l):
        y = self.data[j]
        kernel_val[i][j] = mismatchKernel(x,y,m=self.m,size=self.size)
    if normalize==True:
      print("*****")
      kernel_val = normalize_kernel_matrix(kernel_val.T @ kernel_val)
    else:
      kernel_val =  kernel_val @ kernel_val.T

    for i in range(n):
      f_x = [kernel_val[i][j] * self.alpha[j%l] for j in range(n)]
      s = np.sum(f_x)
      if s > 0:
        f[i] = 1
      else:
        f[i] = 0
    return f'''
  
  def predict_mismatch(self , X_val):
    n = len(X_val)
    l = len(self.data)
    f = np.zeros(n)
    for i in range(n):
      f_x = np.zeros(l)
      x = X_val[i]
      for j in range(l):
        y = self.data[j]
        f_x[j] = (mismatchKernel(x,y,self.m,self.size) / np.sqrt(mismatchKernel(x,x,self.m,self.size)*mismatchKernel(y,y,self.m,self.size)) )* self.alpha[j]
      s = np.sum(f_x)
      if s > 0:
        f[i] = 1
      else:
        f[i] = 0
    return f


  @staticmethod
  @njit
  def predict_LA(X_val,data,alpha):
    n = len(X_val)
    m = len(data)
    f = np.zeros(n)
    for i in prange(n):
      f_x = np.zeros(m)
      x = X_val[i]
      for j in prange(m):
        y = data[j]
        f_x[j] = LA_kernel(x,y) * alpha[j]
      s = np.sum(f_x)
      if s > 0:
        f[i] = 1
      else:
        f[i] = 0
    return f