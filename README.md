# Evaluation of Cross-Lingual Sentence Embeddings

Evaluation scripts and data as described in "Context-Aware Crosslingual Mapping". NAACL 2019. https://arxiv.org/pdf/1903.03243.pdf

The data are derived from WMT'13 parallel sets (common crawl) for Spanish-English and German-English: https://www.statmt.org/wmt13/translation-task.html 

If you use the data or scripts, please cite: 

```
@inproceedings{aldarmaki2019,
  title={Context-Aware Crosslingual Mapping},
  author={Aldarmaki, Hanan and Diab, Mona},
  booktitle={Proceedings of the 2019 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies},
  year={2019}
}
```
And

```

@inproceedings{bojar2013findings,
  title={Findings of the 2013 workshop on statistical machine translation},
  author={Bojar, Ondrej and Buck, Christian and Federmann, Christian and Haddow, Barry and Koehn, Philipp and Leveling, Johannes and Monz, Christof and Pecina, Pavel and Post, Matt and Saint-Amand, Herve and others},
  booktitle={Proceedings of the eighth workshop on statistical machine translation},
  pages={1--44},
  year={2013}
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
