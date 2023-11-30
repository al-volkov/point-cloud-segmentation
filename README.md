# Point Cloud Segmentation



## Structure

* `src`
  * `image_loader` - Classes for loading images.
  * `point_projector` - Classes for projecting points onto a plane.
  * `predictor` - Classes for making predictions.
  * `utils` - Utility functions and classes.
  * `point_cloud_segmentor.py` - Main class for user interaction.
* `example_config.yaml` - Example configuration file.
* `test.py` - Script for evaluation.
* `requirements.txt` - Alternative setup for dependencies.
* `environment.yml` - Environment setup file (for conda users).

## Usage

### 1. Install

Clone the repository and set up the environment:

```bash
git clone https://github.com/al-volkov/point-cloud-segmentation.git
conda env create -f environment.yml
conda activate point-cloud-segmentation
```

Alternatively, you can use requirements.txt to install Python dependencies:

```bash
git clone https://github.com/al-volkov/point-cloud-segmentation.git
pip install -r requirements.txt
```

If you plan to perform semantic segmentation of images on the fly, then you will need to install the appropriate dependencies for the library of your choice:
* [Detectron2](https://detectron2.readthedocs.io/en/latest/tutorials/install.html)
* [MMSegmentation](https://mmsegmentation.readthedocs.io/en/latest/get_started.html)

### 2 **Configure**
    
You should create `.yaml` config with parameters. See example_config.yaml for a template.

### 3 **Evaluate**
```
python test.py --config-path path_to_config
```