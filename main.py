#!/usr/bin/env python
# coding: utf-8
#
# Author:   Yangyang Wang
# URL:      https://github.com/wyyqwqq/dftcam


from __future__ import print_function

import copy
import os.path as osp

import click
import cv2
import glob, os
import matplotlib.cm as cm
import numpy as np
from PIL import Image
from tifffile import imsave
import xml.etree.ElementTree as ET
from scipy.ndimage import label, generate_binary_structure
import torch
import torch.nn.functional as F
from torchvision import models, transforms
from tqdm import tqdm
import yaml
import albumentations
from albumentations.core.composition import Compose

from dft_cam import (
    BackPropagation,
    DFTCAM,
    GuidedBackPropagation,
)



# if a model includes LSTM, such as in image captioning,
# torch.backends.cudnn.enabled = False


def get_device(cuda):
    cuda = cuda and torch.cuda.is_available()
    device = torch.device("cuda" if cuda else "cpu")
    if cuda:
        current_device = torch.cuda.current_device()
        print("Device:", torch.cuda.get_device_name(current_device))
    else:
        print("Device: CPU")
    return device


def load_images(image_paths):
    images = []
    raw_images = []
    print("Images:")
    for i, image_path in enumerate(image_paths):
        print("\t#{}: {}".format(i, image_path))
        image, raw_image = preprocess(image_path)
        images.append(image)
        raw_images.append(raw_image)
    return images, raw_images



def preprocess(image_path):
    raw_image = cv2.imread(image_path)
    raw_image = cv2.resize(raw_image, (224,) * 2)
    image = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )(raw_image[..., ::-1].copy())

    return image, raw_image



def save_gradient(filename, gradient):
    gradient = gradient.cpu().numpy().transpose(1, 2, 0)
    gradient -= gradient.min()
    gradient /= gradient.max()
    gradient *= 255.0
    cv2.imwrite(filename, np.uint8(gradient))


def save_cam(filename, dcam, raw_image, paper_cmap=False):
    dcam = dcam.cpu().numpy()
    cmap = cm.jet_r(dcam)[..., :3] * 255.0
    if paper_cmap:
        alpha = dcam[..., None]
        dcam = alpha * cmap + (1 - alpha) * raw_image
    else:
        dcam = (cmap.astype(np.float) + raw_image.astype(np.float)) / 2

    while 1 in np.shape(dcam):
        dcam = np.squeeze(dcam, axis=0)

    cv2.imwrite(filename, np.uint8(dcam))



# torchvision models
model_names = sorted(
    name
    for name in models.__dict__
    if name.islower() and not name.startswith("__") and callable(models.__dict__[name])
)


@click.group()
@click.pass_context
def main(ctx):
    print("Mode:", ctx.invoked_subcommand)


@main.command()
@click.option("-i", "--image-paths", type=str, multiple=True, required=True)
@click.option("-a", "--arch", type=click.Choice(model_names), required=True)
@click.option("-t", "--target-layer", type=str, required=True)
@click.option("-k", "--topk", type=int, default=3)
@click.option("-l", "--layers", type=int, default=1)
@click.option("-cam", "--cam-type", type=str, default="dftcam")
@click.option("--cuda/--cpu", default=True)
def demo(image_paths, target_layer, arch, topk, cam_type, cuda, layers):
    """
    Visualize model responses
    sample command:
        python main.py demo -a vgg16 -t 'features.29' -k 1 -i /PATH/TO/YOUR/IMAGE/FOLDER/ -cam dftcam -l 5
    """

    device = get_device(cuda)

    # Model from torchvision
    model = models.__dict__[arch](pretrained=True)
    # print(model)
    # model = torch.nn.DataParallel(model)
    model.to(device)
    model.eval()
    # print(model)

    dcam = DFTCAM(model=model)
    gbp = GuidedBackPropagation(model=model)

    error_score_list = []
    iou_score_list = []

    results_folder = "./results/"
    if not os.path.exists(results_folder):
        os.makedirs(results_folder)

    # Loop through every image in folder
    for filename in sorted(glob.glob(os.path.join(image_paths[0], "*.jpeg"))):
        tmp = []
        tmp.append(filename)
        # Images
        images, raw_images = load_images(tmp)
        images = torch.stack(images).to(device)

        img_name = filename.split('/')[-1]


        bp = BackPropagation(model=model)
        probs, ids = bp.forward(images)  # sorted, predict class, [image_batch, top-K]

        # =========================================================================
        _ = dcam.forward(images)
        _ = gbp.forward(images)


        for i in range(topk):
            # Guided Backpropagation
            gbp.backward(ids=ids[:, [i]])
            gradients = gbp.generate()

            # DFT-CAM
            dcam.backward(ids=ids[:, [i]])
            if cam_type == 'dftcam':
                regions = dcam.generateDFT(target_layer=target_layer, num_layers=layers)
            elif cam_type == 'convcam':
                regions = dcam.generateCONV(target_layer=target_layer)


            save_cam(results_folder + "{}-{}-{}-{}.png".format(
                                img_name[:-5], arch, cam_type, i),
                                regions, raw_images[0])

            # save CAM image
            # cam_image = regions.cpu().numpy().astype(np.float32)
            # imsave('./results/'+ cam_type + '_' + arch + '_' + img_name[:-5] + '.tif', cam_image[0][0])



    print('CAM type:', cam_type)
    print('# layers:', layers)
    print('architecture:', arch)




if __name__ == "__main__":
    main()
