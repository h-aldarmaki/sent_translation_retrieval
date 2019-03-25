'''
generate ELMO reps (avg) with pre-computed and cached context independent
token representations

'''

import tensorflow as tf
import os
from bilm import TokenBatcher, BidirectionalLanguageModel, weight_layers, \
    dump_token_embeddings
import numpy as np
import sys

#check number of commandline args
if (len(sys.argv) < 7 or len(sys.argv) > 8):
    print(len(sys.argv))
    print("Usage: python dump_token_embeddings.py input_file vocab_file token_embedding_file output_file R_file lang [top_layer]")
    sys.exit(1)

data_file=sys.argv[1]
vocab_file=sys.argv[2]
token_embedding_file=sys.argv[3]
output_file=sys.argv[4]
R_file=sys.argv[5]
R=np.loadtxt(R_file)
lang=sys.argv[6]

top_layer=False
if len(sys.argv) == 8:
    top_layer=sys.argv[7]

#pretrained LM
#TODO
options_file = "repeat/es/options.json"
weight_file = "repeat/es/weights.hdf5"
if lang == 'de':
  options_file = "repeat/de/options.json"
  weight_file = "repeat/de/weights.hdf5"

#read text from file
raw_txt = [line.rstrip() for line in open(data_file, 'r')]#, encoding="ISO-8859-1")]
tokenized_txt = [sentence.split()[:500] for sentence in raw_txt]

## Now we can do inference.
# Create a TokenBatcher to map text to token ids.
batcher = TokenBatcher(vocab_file)
#batcher = okenWeightBatcher(vocab_file, sif_file)
# Input placeholders to the biLM.
input_token_ids = tf.placeholder('int32', shape=(None, None))

# Build the biLM graph.
bilm = BidirectionalLanguageModel(
    options_file,
    weight_file,
    use_character_inputs=False,
    embedding_weight_file=token_embedding_file
)

# Get ops to compute the LM embeddings.
input_embeddings_op = bilm(input_token_ids)

# Get an op to compute ELMo (weighted average of the internal biLM layers)
elmo_emb = weight_layers('input', input_embeddings_op, l2_coef=0.0, use_top_only=top_layer)

batch_size=32
elmo_size=1024
with tf.Session() as sess:
    # It is necessary to initialize variables once before running inference.
    sess.run(tf.global_variables_initializer()) 

    # Create batches of data.
    input_ids = batcher.batch_sentences(tokenized_txt)
    #context_ids, weights = batcher.batch_sentences(tokenized_context)
    final_res=np.zeros((len(tokenized_txt), elmo_size), dtype=np.float32)
    #run batches of size 128
    for i in range(0,len(tokenized_txt), batch_size): 
      j=i+batch_size if i+batch_size <= len(tokenized_txt) else len(tokenized_txt)
      elmo_emb_= sess.run(
        elmo_emb['weighted_op'],
        feed_dict={input_token_ids: input_ids[i:j,:]}
      )
     #perform averaging here ... 
      res = np.array(elmo_emb_)
      idx=i
      for x, sen in zip(res, tokenized_txt[i:j]):
       x_r=np.transpose(np.matmul(R, np.transpose(x)))       
       avg=np.sum(x_r, axis=0)/len(sen)
       final_res[idx]=avg
       idx=idx+1


np.savetxt(output_file, final_res, delimiter=' ', fmt='%0.6f')
