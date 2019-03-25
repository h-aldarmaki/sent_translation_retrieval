import os
import numpy as np
import scipy
import sys

if(len(sys.argv) < 3):
  print("Usage: python eval.py source_embeddings target_embeddings")
  sys.exit()

s_file=sys.argv[1]
t_file=sys.argv[2]


A=np.loadtxt(s_file)
B=np.loadtxt(t_file)

#calculate average cosine similarity
A_norms=np.linalg.norm(A, axis=1)
B_norms= np.linalg.norm(B, axis=1)
with np.errstate(divide='ignore'):
   cos_all=A.dot(B.T) / np.outer(A_norms,B_norms)

cos_all[np.isnan(cos_all)] = 0

sim_avg=cos_all.diagonal().mean()

print("Average cosine similariy:")
print(sim_avg)
#calculate acc of nearest enighbor search

nn=np.argmax(cos_all, axis=1)
#print(nn)
nn_res=np.equal(nn, np.arange(A.shape[0]))
nn_acc=np.sum(nn_res)/A.shape[0]

#for each sentence in s, find nearest neighbor in t
print("Nearest neighbor accuracy (s->t):")
print(nn_acc)
#calculate correaltion of pair-wise similarity


nn=np.argmax(cos_all, axis=0)
nn_res=np.equal(nn, np.arange(A.shape[0]))
nn_acc=np.sum(nn_res)/A.shape[0]

#for each sentence in t, find nearest neighbor in s
print("Nearest neighbor accuracy (t->s):")
print(nn_acc)
#calculate correaltion of pair-wise similarity


