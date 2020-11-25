## Introduction
Volkscv is a foundational python library for computer vision research and deployment projects

It provides the following functionalities.

- Analyzer
- Metrics.

## LicenseNow

This project is released under the [Apache 2.0 license](LICENSE).

## Installation


### Requirements
- Linux
- Python >= 3.6
- Numpy >= 1.13.3

We have tested the following versions of OS and softwares:

- OS: Ubuntu 16.04.6 LTS
- Python: 3.7.3
- Numpy: 1.16.4

### Install volkscv
1.If your platform is x86 or x64, you can create a conda virtual environment and activate it.
```shell
conda create -n volkscv python=3.7 -y
conda activate volkscv
```
2.Setup
```shell
pip install "git+https://github.com/Media-Smart/volkscv.git"
```

## Support
### Analyzer
- [x] [Statistics](https://github.com/ChaseMonsterAway/volkscv/tree/master/volkscv/analyzer/statistics)
- [x] [Visualization](https://github.com/ChaseMonsterAway/volkscv/tree/master/volkscv/analyzer/visualization)

### Metrics
- [x] [Classification](https://github.com/Media-Smart/volkscv/tree/master/volkscv/metrics/classification)
- [x] [Segmentation](https://github.com/Media-Smart/volkscv/tree/master/volkscv/metrics/segmentation)
- [x] [Detection](https://github.com/Media-Smart/volkscv/tree/master/volkscv/metrics/detection)

## Contact

This repository is currently maintained by 
Chenhao Wang ([@C-H-Wong](http://github.com/C-H-Wong)), 
Yuxin Zou ([@Yuxin Zou](https://github.com/YuxinZou)),
Jun Sun([@ChaseMonsterAway](https://github.com/ChaseMonsterAway)),
Hongxiang Cai ([@hxcai](http://github.com/hxcai)), 
Yichao Xiong ([@mileistone](https://github.com/mileistone)).

## Credits
We got and modified much code from [scikit-learn](https://github.com/scikit-learn/scikit-learn), [cocoapi](https://github.com/cocodataset/cocoapi), 
thanks to [scikit-learn](https://github.com/scikit-learn/scikit-learn), [COCO](https://github.com/cocodataset).
