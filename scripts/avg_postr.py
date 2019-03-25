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
            word_ids = [
                self.word_to_id(cur_word) for cur_word in sentence.split() if self.word_to_id(curr_word) !=-1
            ]
        else:
            word_ids = [self.word_to_id(curr_word) for curr_word in sentence if self.word_to_id(curr_word) !=-1]

        return np.array(word_ids, dtype=np.int32)


#c number of commandline args
if (len(sys.argv) < 6 or len(sys.argv) > 6):
    print(len(sys.argv))
    print("Usage: python avg.py input_file vocab_file embedding_file output_file R")
    sys.exit(1)

data_file=sys.argv[1]
vocab_file=sys.argv[2]
embedding_file=sys.argv[3]
output_file=sys.argv[4]
R_file=sys.argv[5]
R=np.loadtxt(R_file)
R=np.transpose(R)

#TODO separate vocab and embeddings, then read them both here,
weights = np.loadtxt(embedding_file)
print(weights.shape)

#read text from file
raw_txt = [line.rstrip() for line in open(data_file, 'r')]#, encoding="ISO-8859-1")]
tokenized_txt = [sentence.split() for sentence in raw_txt]

vocab = Vocabulary(vocab_file)
idx=0
final_res = np.zeros((len(tokenized_txt), len(weights[0])), dtype=np.float32)

for x in tokenized_txt:
    ids=vocab.encode(x, split=False)
    final_res[idx] = np.average(weights[ids,:], axis=0)
    idx=idx+1

final_res=np.matmul(final_res, R)
print(final_res.shape)
np.savetxt(output_file, final_res, delimiter=' ', fmt='%0.6f')  
