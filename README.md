# qebrain

This repository provides an unofficial released implementation of paper ["Bilingual Expert" Can Find Translation Errors](https://arxiv.org/abs/1807.09433). Since the implementation details, data preprocessing, and other possibilities, it is not guaranteed to reproduce the results in [WMT 2018 QE task](http://www.statmt.org/wmt18/quality-estimation-task.html#results).

## Requirements
1. TensorFlow `pip install tensorflow-gpu`
2. OpenNMT-tf `pip install OpenNMT-tf`

## Basic Usage
1. Download the [parallel datasets](http://www.statmt.org/wmt18/translation-task.html#download) from WMT website.
2. Preprocessing including tokenization, lowercasing, and vocabulary files.
3. The parallel data can be stored in `data/para`, and the example vocab files are in folder `data/vocab`.
4. Run `./expert_train.sh` to train bilingual expert model, and due to the large dataset, we provide the multi GPU implementation.
5. Download the [QE dataset](https://lindat.mff.cuni.cz/repository/xmlui/handle/11372/LRT-2619). An example dataset of De-En has been downloaded and preprocessed in folder `data/qe`.
6. Run `./qe_train.sh` to train the quality estimation model, and due to the small dataset, we only provide the single GPU implementation.
7. Run `./qe_infer.sh` to make the inference on dataset without labels.

## Citation
If you use this code for your research, please cite our papers.
```
@article{fan2018bilingual,
  title={" Bilingual Expert" Can Find Translation Errors},
  author={Fan, Kai and Li, Bo and Zhou, Fengming and Wang, Jiayi},
  journal={arXiv preprint arXiv:1807.09433},
  year={2018}
}
```
