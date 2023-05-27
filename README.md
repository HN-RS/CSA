# CSA
This repo is a tf implementation of "[Contrastive State Augmentations for Reinforcement Learning-Based Recommender Systems](http://arxiv.org/abs/2305.11081)" (SIGIR 2023).
If you have any question, please open an issue or contact crushna@163.com.

## Dependencies

- cuda 11.2
- Python 3.7
- Tensorflow-gpu 2.6.0
- pandas
- numpy


## Datasets
We provide a runnable version of RC15 and RetailRocket. You can run this program directly using dataset in "[Google Drive](https://drive.google.com/drive/folders/19OoKDiXXr0JKnnizGKDNx0KPugbKhiG-?usp=drive_link)". You should download and move it under "./data/."

If you want to get "meituan" dataset and more processing details, please contact crushna@163.com.

## Setup
Make sure the following packages are installed with the correct version.
```bash
conda install tensorflow-gpu==2.6.0
conda install pandas
conda install numpy
pip install trfl
pip install tensorflow-probability==0.14.0 
```

## Get started
The following commands can be used to train and evaluate CSA based on GRU4Rec:
```bash
cd code/GRU4REC/
python CSA_N.py
```
You need to modify the directory of dataset and set hyperparameters for different base models and data according to the data provided in the paper.


## Reference

```
@article{ren2023contrastive,
  title={Contrastive State Augmentations for Reinforcement Learning-Based Recommender Systems},
  author={Ren, Zhaochun and Huang, Na and Wang, Yidan and Ren, Pengjie and Ma, Jun and Lei, Jiahuan and Shi, Xinlei and Luo, Hengliang and Jose, Joemon M and Xin, Xin},
  journal={arXiv preprint arXiv:2305.11081},
  year={2023}
}
```

