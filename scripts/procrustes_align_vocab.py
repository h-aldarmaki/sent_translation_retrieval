import os
import numpy as np
import scipy
import sys

if(len(sys.argv) < 5):
  print("Usage: python procrustes_align_vocab.py source_embeddings target_embeddings dictionary output_file")
  sys.exit()

s_file=sys.argv[1]
t_file=sys.argv[2]
dictionary_file=sys.argv[3]
o_file=sys.argv[4]


#read alignment file
#if both source and target words exists in dicgtionary
#append to A and B. 



def get_glove(glove_path):
    # create word_vec with glove vectors
    word_vec = {}
    with open(glove_path) as f:
        for line in f:
            word, vec = line.split(' ', 1)
            if len(vec.split()) < 100 :
               continue
            word_vec[word] = np.array(list(map(float, vec.split())))
    return word_vec

s_vecs=get_glove(s_file)
t_vecs=get_glove(t_file)

A=[]
B=[]
with open(dictionary_file) as f1:
   for line in f1:
     s_str, t_str=line.strip().split();
     if s_str in s_vecs and t_str in t_vecs :
       A.append(s_vecs[s_str])
       B.append(t_vecs[t_str])

A=np.asarray(A, dtype=np.float32)
B=np.asarray(B, dtype=np.float32)
A=np.transpose(A)
B=np.transpose(B)

M=np.matmul(B,np.transpose(A))
U,S,V_t = np.linalg.svd(M, full_matrices=True)
R=np.matmul(U, V_t)

np.savetxt(o_file, R, delimiter=' ', fmt='%0.6f')  
