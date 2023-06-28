#!/usr/bin/env python
# coding: utf-8
#
# Author:   Yangyang Wang
# URL:      https://github.com/wyyqwqq/dftcam


from collections import Sequence

import numpy as np
import torch
import torch.nn as nn
from torch.nn import functional as F
from tqdm import tqdm


class _BaseWrapper(object):
    def __init__(self, model):
        super(_BaseWrapper, self).__init__()
        self.device = next(model.parameters()).device
        self.model = model
        self.handlers = []  # a set of hook function handlers

    def _encode_one_hot(self, ids):
        one_hot = torch.zeros_like(self.logits).to(self.device)
        one_hot.scatter_(1, ids, 1.0)
        return one_hot

    def forward(self, image):
        # self.image_shape = image.shape[2:]
        self.image_shape = image.shape[-2:]
        self.logits = self.model(image)
        self.probs = F.softmax(self.logits, dim=1)
        return self.probs.sort(dim=1, descending=True)  # ordered results


    def backward(self, ids):
        """
        Class-specific backpropagation
        """
        one_hot = self._encode_one_hot(ids)
        self.model.zero_grad()
        self.logits.backward(gradient=one_hot, retain_graph=True)

    def generate(self):
        raise NotImplementedError

    def remove_hook(self):
        """
        Remove all the forward/backward hook functions
        """
        for handle in self.handlers:
            handle.remove()


class BackPropagation(_BaseWrapper):
    def forward(self, image):
        self.image = image.requires_grad_()
        return super(BackPropagation, self).forward(self.image)

    def generate(self):
        gradient = self.image.grad.clone()
        self.image.grad.zero_()
        return gradient


class GuidedBackPropagation(BackPropagation):

    def __init__(self, model):
        super(GuidedBackPropagation, self).__init__(model)

        def backward_hook(module, grad_in, grad_out):
            # Cut off negative gradients
            if isinstance(module, nn.ReLU):
                return (F.relu(grad_in[0]),)

        for module in self.model.named_modules():
            self.handlers.append(module[1].register_backward_hook(backward_hook))


class DFTCAM(_BaseWrapper):

    def __init__(self, model, candidate_layers=None):
        super(GradCAM, self).__init__(model)
        self.fmap_pool = {}
        self.grad_pool = {}
        self.candidate_layers = candidate_layers  # list

        def save_fmaps(key):
            def forward_hook(module, input, output):
                self.fmap_pool[key] = output.detach()

            return forward_hook

        def save_grads(key):
            def backward_hook(module, grad_in, grad_out):
                self.grad_pool[key] = grad_out[0].detach()

            return backward_hook

        # If any candidates are not specified, the hook is registered to all the layers.
        for name, module in self.model.named_modules():
            if self.candidate_layers is None or name in self.candidate_layers:
                self.handlers.append(module.register_forward_hook(save_fmaps(name)))
                self.handlers.append(module.register_backward_hook(save_grads(name)))

    def _find(self, pool, target_layer):
        # print(pool.keys())
        if target_layer in pool.keys():
            return pool[target_layer]
        else:
            raise ValueError("Invalid layer name: {}".format(target_layer))


    def getCorners(self, input_tensor):
        B, H, W = input_tensor.shape
        output = []
        # upper left
        for i in range(5): # h
            for j in range(5-i): # w
                for b in range(B):
                    output.append(input_tensor[b,i,j].unsqueeze(0))
        # upper right
        for i in range(5): # h
            for j in range(W - 5+i, W): # w
                for b in range(B):
                    output.append(input_tensor[b,i,j].unsqueeze(0))
        # bottom left
        for i in range(H - 5, H): # h
            for j in range(i-H+5+1): # w
                for b in range(B):
                    output.append(input_tensor[b,i,j].unsqueeze(0))
        # bottom right
        for i in range(H - 5, H): # h
            for j in range(W-i+H-6, W): # w
                for b in range(B):
                    output.append(input_tensor[b,i,j].unsqueeze(0))

        upper_left = torch.cat(output, dim=0)
        return upper_left


    def generateDFT(self, target_layer, num_layers=1):
        fmaps = self._find(self.fmap_pool, target_layer)
        grads = fmaps

        # DFT
        for i in range(grads.size()[1]):
            if i == 0:
                dft = torch.fft.fft2(grads[:,i,:,:], dim=(-2, -1))
                dft = dft.real
                dft = self.getCorners(dft)
                DFT_matrix = dft.unsqueeze(0)
            else:
                dft = torch.fft.fft2(grads[:,i,:,:], dim=(-2, -1))
                dft = dft.real
                dft = self.getCorners(dft) 
                DFT_matrix = torch.cat((DFT_matrix, dft.unsqueeze(0)), dim=0)

        dot_product = torch.mm(DFT_matrix, torch.transpose(DFT_matrix, 0, 1))
        dot_product = dot_product - torch.diag(torch.diag(dot_product)) # remove diagonal elements
        sum_product = torch.sum(dot_product, 1) # sum by row, reduce col to 1, [n]

        if num_layers == 1:
            max_idx = [i for i in range(len(sum_product)) if sum_product[i]==torch.max(sum_product)] # find the least unique tensor

            pickOne = fmaps[:,max_idx,:,:]
        else:
            topk_sum, topk_idx = torch.topk(sum_product, num_layers)
            tmp_tensor = torch.zeros_like(fmaps[:,0:1,:,:])
            count = 0
            for idx in topk_idx:
                tmp_tensor += fmaps[:,idx,:,:]
                count += 1
            pickOne = torch.mean(tmp_tensor, 0, True)


        gcam = F.relu(pickOne)
        gcam = F.interpolate(
            gcam, self.image_shape, mode="bilinear", align_corners=False
        )

        B, C, H, W = gcam.shape
        gcam = gcam.view(B, -1)
        gcam -= gcam.min(dim=1, keepdim=True)[0]
        gcam /= gcam.max(dim=1, keepdim=True)[0]
        gcam = gcam.view(B, C, H, W)

        return gcam


    def generateCONV(self, target_layer, num_layers=1):
        fmaps = self._find(self.fmap_pool, target_layer)
        grads = fmaps

        # DFT
        for i in range(grads.size()[1]):
            if i == 0:
                dft = grads[:,i,:,:]

                DFT_matrix = torch.flatten(dft).unsqueeze(0)
            else:
                dft = grads[:,i,:,:]

                DFT_matrix = torch.cat((DFT_matrix, torch.flatten(dft).unsqueeze(0)), dim=0)

        dot_product = torch.mm(DFT_matrix, torch.transpose(DFT_matrix, 0, 1))
        dot_product = dot_product - torch.diag(torch.diag(dot_product)) # remove diagonal elements
        sum_product = torch.sum(dot_product, 1) # sum by row, reduce col to 1, [n]

        if num_layers == 1:
            max_idx = [i for i in range(len(sum_product)) if sum_product[i]==torch.max(sum_product)] # find the least unique tensor

            pickOne = fmaps[:,max_idx,:,:]
        else:
            topk_sum, topk_idx = torch.topk(sum_product, num_layers)

            tmp_tensor = torch.zeros_like(fmaps[:,0:1,:,:])
            for idx in topk_idx:
                tmp_tensor += fmaps[:,idx,:,:]
            pickOne = torch.mean(tmp_tensor, 1, True)

        gcam = F.relu(pickOne)
        gcam = F.interpolate(
            gcam, self.image_shape, mode="bilinear", align_corners=False
        )

        B, C, H, W = gcam.shape
        gcam = gcam.view(B, -1)
        gcam -= gcam.min(dim=1, keepdim=True)[0]
        gcam /= gcam.max(dim=1, keepdim=True)[0]
        gcam = gcam.view(B, C, H, W)

        return gcam
