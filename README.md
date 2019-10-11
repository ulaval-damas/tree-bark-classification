# TreeBarkClassification

## Article

If you use BarkNet 1.0 or this code in your work, please cite the following article:</br>
https://arxiv.org/abs/1803.00949

### Bibtex entry
@INPROCEEDINGS{8593514, 
author={M. {Carpentier} and P. {Giguère} and J. {Gaudreault}}, 
booktitle={2018 IEEE/RSJ International Conference on Intelligent Robots and Systems (IROS)}, 
title={Tree Species Identification from Bark Images Using Convolutional Neural Networks}, 
year={2018}, 
volume={}, 
number={}, 
pages={1075-1081}, 
keywords={feature extraction;forestry;geophysical image processing;image classification;learning (artificial intelligence);neural nets;vegetation mapping;bark images;tree individual number;high-resolution bark images;species recognition;tree diameters;tree bark species classification;standard vision problems;deep learning;forestry related tasks;convolutional neural networks;tree species identification;Vegetation;Forestry;Deep learning;Feature extraction;Training;Cameras;Task analysis}, 
doi={10.1109/IROS.2018.8593514}, 
ISSN={2153-0866}, 
month={Oct},}

## BarkNet 1.0 Database

October 11th, 2019: fixed corrupted BOJ+BOP pictures.
https://storage.googleapis.com/barknet-1/BarkNet%201.0001.zip

## How to run
`python3 train.py --config PATH_TO_CONFIG_FILE`

## How to test
`python3 test.py`
