#!/bin/sh

set -x
lang=$1
home_dir="$(pwd)"
datadir=$home_dir/en-${lang}_train
elmodir=$home_dir/ELMO #TODO your ELMO directory

#process English ELMO avg embeddings (once per model)
test_en=$home_dir/test/newsstest2013.en
cd $elmodir
python dump_token_embeddings.py $test_en $datadir/common/elmo.vocab.en $datadir/common/elmo_token_embeddings.en.hdf5 en
python embed_avg.py $test_en $datadir/common/elmo.vocab.en $datadir/common/elmo_token_embeddings.en.hdf5 $datadir/common/en.elmo_embeddings en

#process other language test vocab
test_file=$home_dir/test/newsstest2013.$lang
python dump_token_embeddings.py $test_file $datadir/common/vocab.test.$lang $datadir/common/elmo_token_embeddings.test.${lang}.hdf5 $lang
cd $home_dir/experiments/

for n in 100 200 500 1000 2000 5000 10000 20000 50000 100000 200000 500000 1000000; do
    thisdir=$datadir/train_$n 
    scratch=$thisdir/elmo
    mkdir $scratch
    perl $home_dir/scripts/extract_aligned_pairs.pl $thisdir/align/sym.align $scratch/pairs $lang
    cd $elmodir
    python dump_token_embeddings.py $thisdir/en.tok $scratch/vocab.en $scratch/elmo_token_embeddings.en.hdf5 en
    python dump_token_embeddings.py $thisdir/${lang}.tok $scratch/vocab.$lang $scratch/elmo_token_embeddings.${lang}.hdf5 $lang
    python embed_word_in_context.py $thisdir/en.tok $scratch/pairs.en  $scratch/vocab.en $scratch/elmo_token_embeddings.en.hdf5 $scratch/en.embeddings en 
    python embed_word_in_context.py $thisdir/${lang}.tok $scratch/pairs.$lang  $scratch/vocab.$lang $scratch/elmo_token_embeddings.${lang}.hdf5 $scratch/${lang}.embeddings $lang
    #CHECK that dictionary is <= 1M words, otherwise sample down
    python $home_dir/scripts/procrustes_align.py $scratch/${lang}.embeddings $scratch/en.embeddings $scratch/R

    #transform test file
    python embed_avg_transform.py $test_file $datadir/common/vocab.test.$lang $datadir/common/elmo_token_embeddings.test.${lang}.hdf5 $scratch/${lang}.final_embeddings $scratch/R $lang

    cd $home_dir/experiments/
    #evaluate performance
    python scripts/eval.py $scratch/${lang}.final_embeddings $datadir/common/en.elmo_embeddings
    
done
