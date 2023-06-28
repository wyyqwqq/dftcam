# DFT-CAM

## Official code for the ICIP 2023 paper "DFT-CAM: Discrete Fourier Transform Driven Class Activation Map"

## How to run 

1. Install required environment using conda virtual environment:
```
conda env create --name envname --file=ptc3.yml
```

2. Run the .py file with your specification:
   * -a deep learning model supported in torchvision
   * -t target layer
   * -k topk class
   * -i input image folder path
   * -cam CAM type (dftcam or convcam)
   * -l numbers of selected layers
```
python main.py demo -a vgg16 -t 'features.29' -k 1 -i /PATH/TO/YOUR/IMAGE/FOLDER/ -cam dftcam -l 5
```
