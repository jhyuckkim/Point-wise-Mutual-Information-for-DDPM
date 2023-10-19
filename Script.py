import matplotlib.pyplot as plt
import torch
import numpy as np
import os

from DiffusionFreeGuidence.DiffusionCondition import GaussianDiffusionSampler, GaussianDiffusionTrainer
from DiffusionFreeGuidence.ModelCondition import UNet
from Utils import get_pointwise_mutual_info_2, get_pointwise_mutual_info_3, get_intermediate_pointwise_mutual_info

def visualize_single_image(image_tensor):
    if image_tensor.shape != (3, 32, 32):
        raise ValueError("Invalid shape")
    image_tensor = (image_tensor + 1) / 2.0
    image_array = image_tensor.numpy().transpose((1, 2, 0))
    plt.imshow(image_array)
    plt.axis('off')
    plt.show()

def visualize_100_images(image_tensor):
    if len(image_tensor) != 100 or image_tensor.shape[1:] != (3, 32, 32):
        raise ValueError("Invalid shape")
        
    fig, axes = plt.subplots(10, 10, figsize=(15, 15),
                             subplot_kw={'xticks':[], 'yticks':[]},
                             gridspec_kw=dict(hspace=0.1, wspace=0.1))

    for i, ax in enumerate(axes.flat):
        img = image_tensor[i].numpy().transpose((1, 2, 0))
        img = (img + 1) / 2.0  # Assuming image was in [-1, 1], scale it back to [0, 1]
        ax.imshow(img)

    plt.show()

def visualize_100_images_with_est(image_tensor, num_array):
    if len(image_tensor) != 100 or image_tensor.shape[1:] != (3, 32, 32) or len(num_array) != 100:
        raise ValueError("Invalid shape")
        
    fig, axes = plt.subplots(10, 10, figsize=(15, 18),  # Adjust figure size
                             subplot_kw={'xticks':[], 'yticks':[]},
                             gridspec_kw=dict(hspace=0.4, wspace=0.1))

    for i, ax in enumerate(axes.flat):
        img = image_tensor[i].numpy().transpose((1, 2, 0))
        img = (img + 1) / 2.0  # Assuming image was in [-1, 1], scale it back to [0, 1]
        ax.imshow(img)
        ax.text(0.5, -0.2, str(round(num_array[i], 2)), size=12, ha="center", transform=ax.transAxes)
        
    plt.show()

def generate_and_classify_misgenerated_images(modelConfig, label, num_misgenerated_required):
    misgenerated_images = []
    count = 0

    device = torch.device(modelConfig["device"])
    
    with torch.no_grad():
        model = UNet(T=modelConfig["T"], num_labels=10, ch=modelConfig["channel"], ch_mult=modelConfig["channel_mult"],
                        num_res_blocks=modelConfig["num_res_blocks"], dropout=modelConfig["dropout"]).to(device)
        ckpt = torch.load(os.path.join(modelConfig["save_dir"], modelConfig["test_load_weight"]), map_location=device)
        model.load_state_dict(ckpt)
        model.eval()

        classifier = torch.hub.load("chenyaofo/pytorch-cifar-models", "cifar10_repvgg_a2", pretrained=True)
        classifier = classifier.to(modelConfig["device"])
        classifier.eval()

        sampler = GaussianDiffusionSampler(
            model, modelConfig["beta_1"], modelConfig["beta_T"], modelConfig["T"], w=modelConfig["w"]).to(device)

        while count < num_misgenerated_required:
            noisyImage = torch.randn(
                size=[modelConfig["batch_size"], 3, modelConfig["img_size"], modelConfig["img_size"]], device=device)
            sampledImgs = sampler(noisyImage, torch.tensor([label]*modelConfig["batch_size"]).to(device))

            output = classifier(sampledImgs)
            _, predicted = torch.max(output.data, 1)
            predicted_labels = predicted + 1

            indices_not_equal = torch.nonzero(predicted_labels != label).squeeze()
            if len(indices_not_equal) > 0:
                misgenerated_images.extend(sampledImgs[indices_not_equal])
                count += len(indices_not_equal)

    # Save the misgenerated images
    torch.save(misgenerated_images, f"misgenerated_images_label_{label}_4000steps.pt") # remove

    return misgenerated_images

def get_est(Img, label, model_num, t, iter=5):
    result = 0
    for _ in range(iter):
        _, est = get_intermediate_pointwise_mutual_info(Img, label, model_num, t)
        result += est
    return result/iter

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
torch.cuda.empty_cache()

for label in range(1,11):
    # Generate 100 images of the given label

    # device = torch.device(modelConfig["device"])
    # # load model and evaluate
    # with torch.no_grad():
    #     model = UNet(T=modelConfig["T"], num_labels=10, ch=modelConfig["channel"], ch_mult=modelConfig["channel_mult"],
    #                     num_res_blocks=modelConfig["num_res_blocks"], dropout=modelConfig["dropout"]).to(device)
    #     ckpt = torch.load(os.path.join(
    #         modelConfig["save_dir"], modelConfig["test_load_weight"]), map_location=device)
    #     model.load_state_dict(ckpt)
    #     print("model load weight done.")
    #     model.eval()
    #     sampler = GaussianDiffusionSampler(
    #         model, modelConfig["beta_1"], modelConfig["beta_T"], modelConfig["T"], w=modelConfig["w"]).to(device)
    #     # Sampled from standard normal distribution
    #     noisyImage = torch.randn(
    #         size=[modelConfig["batch_size"], 3, modelConfig["img_size"], modelConfig["img_size"]], device=device)
    #     sampledImgs, x_hat_tensor, x_t_tensor, snr, alphas_bar = sampler(noisyImage, torch.tensor([label]*modelConfig["batch_size"]).to(device), tracking_mode=True)

    # torch.save(sampledImgs, '100_sampled_images_label_'+str(label)+'_model_0.pt')
    # torch.save(x_t_tensor, '100_generation_steps_label_'+str(label)+'_model_0.pt')

    # del model
    # del sampler

    print("label", label, "started")
    sampledImgs = torch.load('100_sampled_images_label_'+str(label)+'_model_0.pt')
    x_t_tensor = torch.load('100_generation_steps_label_'+str(label)+'_model_0.pt')

    est_list_list = []

    for index in range(10):
        est_list = []

        for t in range(3750, 250 - 1, -250):
            est = get_est(x_t_tensor[index][t-1], label, model_num=0, t=t, iter=5).cpu().numpy()
            est_list.append(est)
            print(est)

        est = get_est(sampledImgs[index], label, model_num=0, t=0, iter=5).cpu().numpy()
        est_list.append(est)
        # print("PMI =", est)
        est_list = np.array(est_list)
        print(est_list)

        est_list_list.append(est_list)
        torch.save(est_list, 'PMI_along_generation_label_'+str(label)+'index'+str(index)+'_model_0.pt')
    
    est_list_list = torch.tensor(np.array(est_list_list))
    print(est_list_list)
    torch.save(est_list_list, 'PMI_along_generation_label_'+str(label)+'_model_0.pt')
    print("saved")

    
