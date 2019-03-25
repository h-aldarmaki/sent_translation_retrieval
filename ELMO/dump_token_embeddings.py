'''
pre-compute and cache context independent token representations
Usage: python dump_token_embeddings.py input_file vocab_file token_embedding_file 

'''

import tensorflow as tf
import os
from bilm import TokenBatcher, BidirectionalLanguageModel, weight_layers, \
    dump_token_embeddings
import sys

#check that three args are given
if (len(sys.argv) < 5 or len(sys.argv) > 5):
    print(len(sys.argv))
    print("Usage: python dump_token_embeddings.py input_file vocab_file token_embedding_file lang")
    sys.exit(1)

data_file=sys.argv[1]
vocab_file=sys.argv[2]
token_embedding_file=sys.argv[3]
lang=sys.argv[4]
#read text from file
raw_txt = [line.rstrip() for line in open(data_file, 'r')]#, encoding = "ISO-8859-1")]
tokenized_txt = [sentence.split() for sentence in raw_txt]

# Create the vocabulary file with all unique tokens and
# the special <S>, </S> tokens (case sensitive).
all_tokens = set(['<S>', '</S>'])
for sentence in tokenized_txt:
    for token in sentence:
        all_tokens.add(token)

with open(vocab_file, 'w') as fout:
    fout.write('\n'.join(all_tokens))

#pretrained LM
#options_file = "pretrained/elmo_2x4096_512_2048cnn_2xhighway_options.json"
#weight_file = "pretrained/elmo_2x4096_512_2048cnn_2xhighway_weights.hdf5"
#TODO set file names
options_file = "repeat/en_saved/options.json"
weight_file = "repeat/en/weights.hdf5"
if lang == 'es':
  options_file = "repeat/es/options.json"
  weight_file = "repeat/es/weights.hdf5"
if lang == 'de':
  options_file = "repeat/de/options.json"
  weight_file = "repeat/de/weights.hdf5"

# Dump the token embeddings to a file. Run this once for your dataset.

dump_token_embeddings(
    vocab_file, options_file, weight_file, token_embedding_file
)
tf.reset_default_graph()

