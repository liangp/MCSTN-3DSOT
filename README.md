# Multi-Correlation Siamese Transformer Network with Dense Connection for 3D Single Object Tracking

## Introduction
This repository is released for MCSTN-3DSOT in our [IEEE ROBOTICS AND AUTOMATION LETTERS 2023 paper (poster)](https://ieeexplore.ieee.org/document/10287541).

**Note**: Our code is an improvement on previous work [V2B](https://github.com/fpthink/V2B), if you are more familiar with [V2B](https://github.com/fpthink/V2B), you can also refer to their code.

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
**Note**: We treated the three datasets KITTI, nuScenes, and Waymo in the same way as [V2B](https://github.com/fpthink/V2B) and [STNet](https://github.com/fpthink/STNet).
### [KITTI dataset](https://projet.liris.cnrs.fr/imagine/pub/proceedings/CVPR2012/data/papers/424_O3C-04.pdf)
* Download the [velodyne](http://www.cvlibs.net/download.php?file=data_tracking_velodyne.zip), [calib](http://www.cvlibs.net/download.php?file=data_tracking_calib.zip) and [label_02](http://www.cvlibs.net/download.php?file=data_tracking_label_2.zip) from [KITTI Tracking](http://www.cvlibs.net/datasets/kitti/eval_tracking.php). Unzip the downloaded files and place them under the same parent folder.

### [nuScenes dataset](https://openaccess.thecvf.com/content_CVPR_2020/papers/Caesar_nuScenes_A_Multimodal_Dataset_for_Autonomous_Driving_CVPR_2020_paper.pdf)
* Download the Full dataset (v1.0) from [nuScenes](https://www.nuscenes.org/).
  
    Note that base on the offical code [nuscenes-devkit](https://github.com/nutonomy/nuscenes-devkit), we modify and use it to convert nuScenes format to KITTI format. It requires metadata from nuScenes-lidarseg. Thus, you should replace *category.json* and *lidarseg.json* in the Full dataset (v1.0). We provide these two json files in the nuscenes_json folder.

    Executing the following code to convert nuScenes format to KITTI format
    ```
    cd nuscenes-devkit-master/python-sdk/nuscenes/scripts
    python export_kitti.py --nusc_dir=<nuScenes dataset path> --nusc_kitti_dir=<output dir> --split=<dataset split>
    ```

    Note that the parameter of "split" should be "train_track" or "val". In our paper, we use the model trained on the KITTI dataset to evaluate the generalization of the model on the nuScenes dataset.
	
### [Waymo open dataset](https://openaccess.thecvf.com/content_CVPR_2020/papers/Sun_Scalability_in_Perception_for_Autonomous_Driving_Waymo_Open_Dataset_CVPR_2020_paper.pdf)
* We follow the benchmark created by [LiDAR-SOT](https://github.com/TuSimple/LiDAR_SOT) based on the waymo open dataset. You can download and process the waymo dataset as guided by [their code](https://github.com/TuSimple/LiDAR_SOT), and use our code to test model performance on this benchmark.
* The benchmark they built have many things that we don't use, but the following processing results are necessary:
```
[waymo_sot]
    [benchmark]
        [validation]
            [vehicle]
                bench_list.json
                easy.json
                medium.json
                hard.json
            [pedestrian]
                bench_list.json
                easy.json
                medium.json
                hard.json
    [pc]
        [raw_pc]
            Here are some segment.npz files containing raw point cloud data
    [gt_info]
        Here are some segment.npz files containing tracklet and bbox data
```

**Node**: After you get the dataset, please modify the path variable ```data_dir&val_data_dir``` about the dataset under configuration file ```MCSTN_main/utils/options```.

## Evaluation

Train a new model:
```
python main.py --which_dataset KITTI/NUSCENES --category_name category_name
```

Test a model:
```
python main.py --which_dataset KITTI/NUSCENES/WAYMO --category_name category_name --train_test test
```
For more preset parameters or command debugging parameters, please refer to the relevant code and change it according to your needs.

## Visualization
```
cd MCSTN_main/visualization/
python visual.py
```

## Citation

If you find the code or trained models useful, please consider citing:

```
@ARTICLE{10287541,
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

- Thank Le Hui for his implementation of [V2B](https://github.com/fpthink/V2B).
- Thank Pang for the [3D-SOT benchmark](https://arxiv.org/pdf/2103.06028.pdf) based on the waymo open dataset.




