import os
import numpy as np
import scipy
import sys

if(len(sys.argv) < 4):
  print("Usage: python procrustes_align.py source_embeddings target_embeddings output_file")
  sys.exit()

s_file=sys.argv[1]
t_file=sys.argv[2]
o_file=sys.argv[3]

A=np.loadtxt(s_file)
B=np.loadtxt(t_file)
if len(A) > 1000000:
  idx = np.random.choice(np.arange(len(A)), 1000000, replace=False)
  A = A[idx,:]
  B = B[idx,:]

A_S=np.sum(A, axis=1)
B_S=np.sum(B, axis=1)
for i in range(A_S.size):
   if A_S[i] == 0:
      B[i,:]= A[i,:]
   if B_S[i] == 0:
      A[i,:]=B[i,:]

A=np.transpose(A)
B=np.transpose(B)

M=np.matmul(B,np.transpose(A))
U,S,V_t = np.linalg.svd(M, full_matrices=True)
R=np.matmul(U, V_t)

np.savetxt(o_file, R, delimiter=' ', fmt='%0.6f')  
