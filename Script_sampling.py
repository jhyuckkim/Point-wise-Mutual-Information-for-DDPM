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
    "T": 4000, 
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
    "save_dir": "./CheckpointsCondition4000/Model_0/", 
    "training_load_weight": None,
    "test_load_weight": "ckpt_99_.pt", 
    "sampled_dir": "./SampledImgs/",
    "sampledNoisyImgName": "NoisyGuidenceImgs.png",
    "sampledImgName": "SampledGuidenceImgs.png",
    "nrow": 8
}

device = torch.device(modelConfig["device"])

for label in range(1, 11):

    torch.cuda.empty_cache()

    with torch.no_grad():
        model = UNet(T=modelConfig["T"], num_labels=10, ch=modelConfig["channel"], ch_mult=modelConfig["channel_mult"],
                        num_res_blocks=modelConfig["num_res_blocks"], dropout=modelConfig["dropout"]).to(device)
        ckpt = torch.load(os.path.join(modelConfig["save_dir"], modelConfig["test_load_weight"]), map_location=device)
        model.load_state_dict(ckpt)
        model.eval()

        sampler = GaussianDiffusionSampler(
            model, modelConfig["beta_1"], modelConfig["beta_T"], modelConfig["T"], w=modelConfig["w"]).to(device)

        for i in range(60):
            noisyImage = torch.randn(
                size=[modelConfig["batch_size"], 3, modelConfig["img_size"], modelConfig["img_size"]], device=device)
            new_sampledImgs, _, _, _, _ = sampler(noisyImage, torch.tensor([label]*modelConfig["batch_size"]).to(device), tracking_mode=True)
            
            torch.save(new_sampledImgs, f"sampled_images_label_{label}_4000_steps_weight_1.0_batch_{i}.pt")
            print("batch", i, "saved")
            

# Create the Images directory if it doesn't exist
os.makedirs('Images', exist_ok=True)

# Iterate over labels
for label in range(1,11):
    # Initialize an empty list to hold all batches for the current label
    all_batches = []

    # Iterate over 60 batches
    for batch in range(60):
        # Load the batch
        batch_data = torch.load(f'sampled_images_label_{label}_4000_steps_weight_1.0_batch_{batch}.pt')

        # Append the loaded batch to the list
        all_batches.append(batch_data)

    # Concatenate all batches along the 0th dimension
    concatenated_images = torch.cat(all_batches, dim=0)

    # Check if the concatenated tensor has the expected shape (6000, 3, 32, 32)
    assert concatenated_images.shape == (6000, 3, 32, 32), f"Unexpected shape for label {label}"

    # Save the concatenated tensor
    torch.save(concatenated_images, f'Images/sampled_images_label_{label}_4000_steps_weight_1.0.pt')

print("All tensors saved successfully.")