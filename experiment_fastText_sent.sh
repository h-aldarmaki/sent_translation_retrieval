#!/bin/sh


set -x
lang=$1
home_dir="$(pwd)"
fasttext_dir=/path/to/your/fastText
datadir=$home_dir/en-${lang}_train
#process English  embeddings (once per model)
test_dir=$home_dir/test
test_en=$test_dir/newsstest2013.en
test_file=$test_dir/newsstest2013.${lang}
mkdir $datadir/common
#extract vocab
cat $test_en | tr " " "\n" | sort | uniq | sed 1d > $datadir/common/test.vocab.en
#extract fastText vectors
cat $datadir/common/test.vocab.en | $fasttext_dir/fasttext print-vectors $fasttext_dir/output/en.bin | cut -f2- -d' ' > $datadir/common/test.vectors.en
#generate english test set embeddings
python $home_dir/scripts/avg.py $test_en $datadir/common/test.vocab.en $datadir/common/test.vectors.en $datadir/common/en.ft_embeddings

#process other language test vocab and fastText vectors
cat $test_file | tr " " "\n" | sort | uniq | sed 1d > $datadir/common/test.vocab
cat $datadir/common/test.vocab | $fasttext_dir/fasttext print-vectors $fasttext_dir/output/${lang}.bin | cut -f2- -d' ' > $datadir/common/test.vectors


for n in 100 200 500 1000 2000 5000 10000 20000 50000 100000 200000 500000 1000000; do
    thisdir=$datadir/train_$n 
    scratch=$thisdir/avg
    mkdir $scratch
    #extract dictionary and generate embeddings for training set
    cat $thisdir/en.tok | tr " " "\n" | sort | uniq | sed 1d > $scratch/vocab.en
    cat $scratch/vocab.en | $fasttext_dir/fasttext print-vectors $fasttext_dir/output/en.bin | cut -f2- -d' ' > $scratch/vectors.en
    python $home_dir/scripts/avg.py $thisdir/en.tok $scratch/vocab.en $scratch/vectors.en $scratch/en.embeddings
    #generate embeddings for other language
    cat $thisdir/${lang}.tok | tr " " "\n" | sort | uniq | sed 1d > $scratch/vocab.$lang
    cat $scratch/vocab.$lang | $fasttext_dir/fasttext print-vectors $fasttext_dir/output/${lang}.bin | cut -f2- -d' ' > $scratch/vectors.$lang
    python $home_dir/scripts/avg.py $thisdir/${lang}.tok $scratch/vocab.$lang $scratch/vectors.$lang $scratch/${lang}.embeddings

    #git orthogonal transformation between sentence embeddings
    python $home_dir/scripts/procrustes_align.py $scratch/${lang}.embeddings $scratch/en.embeddings $scratch/R

    #transform test file
    python $home_dir/scripts/sif_postr.py $test_file $datadir/common/test.vocab $datadir/common/test.vectors $scratch/${lang}.final_embeddings $scratch/R
    #evaluate performance
    python scripts/eval.py $scratch/${lang}.final_embeddings $datadir/common/en.ft_embeddings
     
done
