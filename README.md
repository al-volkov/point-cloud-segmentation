# Point cloud segmentation

## Structure

* src
  * image_loader - Classes responsible for loading images
  * point_projector - Classes responsible for projecting points onto a planeClasses responsible for projecting points onto a plane
  * predictor - Classes responsible for projecting points onto a plane
  * utils
  * point_cloud_segmentor.py - Main class responsible for user interaction
* example_config.yaml
* test.py - Script for evaluation
## Usage

1 **Install**
```
git clone https://github.com/al-volkov/point-cloud-segmentation.git
conda env create -f environment.yml
conda activate point-cloud-segmentation
```
If you plan to perform semantic segmentation on the fly, then you will need to install the appropriate dependencies for the library of your choice:
* [Detectron2](https://detectron2.readthedocs.io/en/latest/tutorials/install.html)
* [MMSegmentation](https://mmsegmentation.readthedocs.io/en/latest/get_started.html)

2 **Configure**
    
You should create `.yaml` config with parameters.

4 **Evaluate**
```
python test.py --config-path path_to_config
```