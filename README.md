# Multi-Correlation Siamese Transformer Network with Dense Connection for 3D Single Object Tracking

## Introduction
This repository is the official implementation of our paper "Multi-Correlation Siamese Transformer Network with Dense Connection for 3D Single Object Tracking, IEEE Robotics and Automation Letters, 2023". Our code is based on [V2B](https://github.com/fpthink/V2B) and [SST](https://github.com/tusen-ai/SST).



## Environment settings
* Create an environment for MCSTN
```
conda create -n MCSTN python=3.7
conda activate MCSTN
```

* Install pytorch and torchvision
```
conda install pytorch==1.9.1 torchvision==0.10.1 torchaudio==0.9.1 cudatoolkit=10.2 -c pytorch
```

* Install dependencies.
```
pip install -r requirements.txt
```

## Data preparation
We use the datasets KITTI, nuScenes, and Waymo in the same way as [V2B](https://github.com/fpthink/V2B) and [STNet](https://github.com/fpthink/STNet). Please refer to [V2B](https://github.com/fpthink/V2B) for the detail of data preparation.



**Note**: After you get the dataset ready, please modify the path variable ```data_dir``` and ```val_data_dir``` about the dataset under configuration file ```MCSTN_main/utils/options```.

## Evaluation

Train a new model:
```
python main.py --which_dataset KITTI/NUSCENES --category_name category_name
```

Test a model:
```
python main.py --which_dataset KITTI/NUSCENES/WAYMO --category_name category_name --train_test test
```
Please refer to the relevant code for more details of the parameter setting.

## Visualization
```
cd MCSTN_main/visualization/
python visual.py
```

## Citation

If you find the code or trained models useful, please consider citing:

```
@ARTICLE{mcstn2023,
  author={Feng, Shihao and Liang, Pengpeng and Gao, Jin and Cheng, Erkang},
  journal={IEEE Robotics and Automation Letters}, 
  title={Multi-Correlation Siamese Transformer Network With Dense Connection for 3D Single Object Tracking}, 
  year={2023},
  volume={8},
  number={12},
  pages={8066-8073},
  doi={10.1109/LRA.2023.3325715}}
```

## Acknowledgements

- Thank Le Hui et al. for their implementation of [V2B](https://github.com/fpthink/V2B).
- Thank Ziqi Pang et al. for the  [ Waymo 3D-SOT benchmark](https://arxiv.org/pdf/2103.06028.pdf).


