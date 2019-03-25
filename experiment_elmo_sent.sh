#!/bin/sh

set -x
lang=$1
home_dir="$(pwd)"
datadir=$home_dir/en-${lang}_train
elmodir=$home_dir/ELMO #TODO
test_file=$home_dir/test/newsstest2013.$lang

for n in 100 200 500 1000 2000 5000 10000 20000 50000; do # 100000 200000 500000 1000000; do
    thisdir=$datadir/train_$n 
    scratch=$thisdir/elmo
    mkdir $scratch/sent
    
    cd $elmodir
    python embed_avg.py $thisdir/en.tok $scratch/vocab.en $scratch/elmo_token_embeddings.en.hdf5 $scratch/sent/en.embeddings en 
    python embed_avg.py $thisdir/${lang}.tok $scratch/vocab.$lang $scratch/elmo_token_embeddings.${lang}.hdf5 $scratch/sent/${lang}.embeddings $lang 

    python $home_dir/scripts/procrustes_align.py $scratch/sent/${lang}.embeddings $scratch/sent/en.embeddings $scratch/sent/R

    #transform test file
    python embed_avg_post_transform.py $test_file $datadir/common/vocab.test.$lang $datadir/common/elmo_token_embeddings.test.${lang}.hdf5 $scratch/sent/${lang}.final_embeddings $scratch/sent/R $lang

    cd $home_dir/experiments/
    #evaluate performance
    python scripts/eval.py $scratch/sent/${lang}.final_embeddings $datadir/common/en.elmo_embeddings
    
done
