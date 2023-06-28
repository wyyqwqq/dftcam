# DFT-CAM

## Official code for the ICIP 2023 paper "DFT-CAM: Discrete Fourier Transform Driven Class Activation Map"

## How to run 

1. Install required environment using conda virtual environment:
```
conda env create --name ptc3 --file=ptc3.yml
```
   Torch version should be >= '1.13.1+cu117'

2. Run the main.py file with your specification:
 * -a: deep learning model supported in torchvision
 * -t: target layer
 * -k: topk class
 * -i: input image folder path (should be JPEG images)
 * -cam: CAM type (dftcam or convcam)
 * -l: numbers of selected layers

```
python main.py demo -a vgg16 -t 'features.29' -k 1 -i /PATH/TO/YOUR/IMAGE/FOLDER/ -cam dftcam -l 5
```
Outputs will be saved in the "results" folder.
