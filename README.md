# MOE-RRG

This is the code of Mixture of Experts for Radiology Report Generation .


## Requirements

- `torch==1.7.1`
- `torchvision==0.8.2`
- `opencv-python==4.4.0.42`

## Download R2GenCMN
You can download the models we trained for each dataset from [here](https://github.com/zhjohnchan/R2GenCMN/blob/main/data/r2gencmn.md).

## Datasets
 you can download the dataset from [here](https://drive.google.com/file/d/1DS6NYirOXQf8qYieSVMvqNwuOlgAbM_E/view?usp=sharing) and then put the files in `data/mimic_cxr`. You can apply the dataset [here](https://drive.google.com/file/d/1DS6NYirOXQf8qYieSVMvqNwuOlgAbM_E/view?usp=sharing) with your license of [PhysioNet](https://physionet.org/content/mimic-cxr-jpg/2.0.0/).


## Train

Run `bash train_mimic_cxr.sh` to train a model on the MIMIC-CXR data.

## Test

Run `bash test_mimic_cxr.sh` to test a model on the MIMIC-CXR data.


