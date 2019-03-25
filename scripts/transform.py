import os
import numpy as np
import scipy
import sys

if(len(sys.argv) < 4):
  print("Usage: python transform.py source_embeddings R output_file")
  sys.exit()

s_file=sys.argv[1]
r_file=sys.argv[2]
o_file=sys.argv[3]

A=np.loadtxt(s_file)
R=np.loadtxt(r_file)
R=np.transpose(R)

B=np.matmul(A,R)

np.savetxt(o_file, B, delimiter=' ', fmt='%0.6f')  
