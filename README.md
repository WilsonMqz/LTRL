# LTRL

This is the implemention of our ICMR'24 paper (Learning from Reduced Labels for Long-Tailed Data).

Requirements: 
Python 3.8.5, 
numpy 1.24.3, 
torch 2.0.1,
torchvision 0.15.2.

You need to:
1. Download CIFAR-10 datasets into './data/'.
2. Run the following demos:
```
python3 train.py --dataset cifar10 --method SL --imb-factor 0.02 --seed {} --fix-size 4 --extend-size 1
python3 train.py --dataset cifar10 --method SL --imb-factor 0.01 --seed {} --fix-size 4 --extend-size 1
```

## Citation
If you find this repository useful, please consider giving a star ⭐ and citation
```
@inproceedings{wei2024learning,
  title={Learning from Reduced Labels for Long-Tailed Data},
  author={Wei, Meng and Li, Zhongnian and Zhou, Yong and Xu, Xinzheng},
  booktitle={Proceedings of the 2024 International Conference on Multimedia Retrieval},
  pages={111--119},
  year={2024}
}
```

