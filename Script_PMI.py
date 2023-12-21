import matplotlib.pyplot as plt
import torch
import numpy as np
import os

from DiffusionFreeGuidence.DiffusionCondition import GaussianDiffusionSampler, GaussianDiffusionTrainer
from DiffusionFreeGuidence.ModelCondition import UNet
from Utils import *

modelConfig = {
    "state": "eval",
    "epoch": 1,
    "batch_size": 100,
    "T": 4000, # 4000
    "channel": 128,
    "channel_mult": [1, 2, 2, 2],
    "num_res_blocks": 2,
    "dropout": 0.15,
    "lr": 1e-4,
    "multiplier": 2.5,
    "beta_1": 1e-4,
    "beta_T": 0.028,
    "img_size": 32,
    "grad_clip": 1.,
    "device": "cuda:0",
    "w": 1.0,
    "save_dir": "./CheckpointsCondition4000/Model_0/", # 4000
    "training_load_weight": None,
    "test_load_weight": "ckpt_99_.pt", # 99
    "sampled_dir": "./SampledImgs/",
    "sampledNoisyImgName": "NoisyGuidenceImgs.png",
    "sampledImgName": "SampledGuidenceImgs.png",
    "nrow": 8
}

device = torch.device(modelConfig["device"])

for label in range(1, 11):

    torch.cuda.empty_cache()

    sampledImgs = torch.load(f'Images/sampled_images_label_{label}_4000_steps_weight_1.0.pt').to("cpu")

    pmi_list = []

    for i in range(len(sampledImgs)):
        Img = sampledImgs[i]
        pmi = get_est(Img.to(device), label, 0, 1)
        print("index =", i, "pmi =", pmi)
        pmi_list.append(pmi)

    torch.save(torch.tensor(pmi_list), f'PMI/pmi_values_label_{label}_4000_steps_weight_1.0')