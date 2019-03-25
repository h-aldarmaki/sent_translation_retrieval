# Evaluation of Cross-Lingual Sentence Embeddings

Evaluation scripts and data as described in "Context-Aware Crosslingual Mapping". NAACL 2019. https://arxiv.org/pdf/1903.03243.pdf

If you use the data or scripts, please cite: 

```
@inproceedings{aldarmaki2019,
  title={Context-Aware Crosslingual Mapping},
  author={Aldarmaki, Hanan and Diab, Mona},
  booktitle={Proceedings of the 2019 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies},
  year={2019}
}
```

## Requirements ##

Python 3.4 or larger

If you're running the ELMo scripts, download the tensorflow version: 

https://github.com/allenai/bilm-tf

## Instructions ##

* Download parallel data for the alignment: 

   * https://figshare.com/articles/Subsets_of_WMT13_Parallel_Data_for_Evaluating_Cross-Lingual_Embeddings/7886240/2

   * (use version 1 if you don't need word alignments for ELMo)

* unzip the data files

* Add ELMo scripts (from the directory ELMO/) to your ELMo installation directory. 

* Modify the scripts above to point to your pre-trained ELMo models for each language. 

* Modify experiment....sh files according to your needs. 

I'm providing the bash scripts for FastText sentence mapping (averaging) and ELMo word and sentence mapping. For other options, check the scripts/ directory. 
