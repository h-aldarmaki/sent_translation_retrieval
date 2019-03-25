'''
generate sentence reps (avg) with pre-train word embeddings

'''

import os
import numpy as np
import sys


#adapted from https://github.com/allenai/bilm-tf/blob/master/bilm/data.py
class Vocabulary(object):
    def __init__(self, filename):
        '''
        filename = the vocabulary file.  It is a flat text file with one
            (normalized) token per line
        '''
        self._word_to_id = {}
        self._unk = -1

        with open(filename, 'r') as f: #, encoding="ISO-8859-1") as f:
            idx = 0
            for line in f:
                word_name = line.strip()
                self._word_to_id[word_name] = idx
                idx += 1
    def word_to_id(self, word):
        if word in self._word_to_id:
            return self._word_to_id[word]
        return self._unk
    def encode(self, sentence, split=True):
        if split:
            words = [curr_word for curr_word in sentence.split() if self.word_to_id(curr_word) != -1]
        else:
            words = [curr_word for curr_word in sentence if self.word_to_id(curr_word) !=-1]
        
        word_ids = [self.word_to_id(curr_word) for curr_word in words] 
        return np.array(word_ids, dtype=np.int32), words


class Weights(object):
    def __init__(self, filename):
       '''
       filename = the weight file. It is a flat text file with one word followed by its weight
       '''     
       self._word_to_w={}
       self._unk = 1 #weight for unseen words (assuming sif weighting)
       with open(filename, 'r') as f: #, encoding="ISO-8859-1") as f:
           for line in f:
              toks = line.strip().split()
              self._word_to_w[toks[0]] = toks[1]
    def encode(self, sentence, split=True):
        if split: #TODO only use with above return
            word_ws = [self.word_to_w(curr_word.lower()) for curr_word in sentence.split()]
        else:
            word_ws = [self.word_to_w(curr_word.lower()) for curr_word in sentence]
        return np.array(word_ws, dtype=np.float32)
    def word_to_w(self, word):
        if word in self._word_to_w:
           return self._word_to_w[word]   
        else:
           return self._unk


#c number of commandline args
if (len(sys.argv) < 7 or len(sys.argv) > 7):
    print(len(sys.argv))
    print("Usage: python avg.py input_file vocab_file sif_file embedding_file output_file R")
    sys.exit(1)

data_file=sys.argv[1]
vocab_file=sys.argv[2]
sif_file=sys.argv[3]
embedding_file=sys.argv[4]
output_file=sys.argv[5]
R_file=sys.argv[6]
R=np.loadtxt(R_file)
R=np.transpose(R)

embeddings = np.loadtxt(embedding_file)
#print(embeddings.shape)

#read text from file
raw_txt = [line.rstrip() for line in open(data_file, 'r')]#, encoding="ISO-8859-1")]
tokenized_txt = [sentence.split() for sentence in raw_txt]

vocab = Vocabulary(vocab_file)
weights = Weights(sif_file)
idx=0
final_res = np.zeros((len(tokenized_txt), len(embeddings[0])), dtype=np.float32)

for x in tokenized_txt:
    ids, toks=vocab.encode(x, split=False)
    w = weights.encode(toks, split=False)
    final_res[idx] = np.average(np.matmul(embeddings[ids,:],R), weights=w, axis=0)
    idx=idx+1
#print(final_res.shape)
np.savetxt(output_file, final_res, delimiter=' ', fmt='%0.6f')  
