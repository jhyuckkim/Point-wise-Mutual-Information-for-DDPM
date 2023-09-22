# import torch
# import os

# from DiffusionFreeGuidence.ModelCondition import UNet
# # from Diffusion.Model import UNet as UncondUNet

# def simulate_BM(x, gamma_list):
#     # Ensure gamma_list is a tensor on the same device as x
#     if not isinstance(gamma_list, torch.Tensor):
#         gamma_list = torch.tensor(gamma_list).to(x.device)
    
#     delta_gamma = torch.diff(gamma_list).to(x.device)

#     # Calculate increments of the Brownian motion using delta_gamma
#     delta_W = torch.randn((len(delta_gamma),) + x.shape, device=x.device) * torch.sqrt(delta_gamma).view(-1, 1, 1, 1).to(x.device)

#     # Initialize the first value of W_gamma using the variance provided by the first component of gamma_list
#     W_initial = torch.randn_like(x) * torch.sqrt(gamma_list[0]).to(x.device)
#     W_gamma = W_initial.unsqueeze(0)
#     W_gamma = torch.cat([W_gamma, torch.cumsum(delta_W, dim=0)], dim=0).to(x.device)

#     # Calculate z_gamma
#     z_gamma = gamma_list.view(-1, 1, 1, 1).to(x.device) * x + W_gamma

#     return W_gamma, delta_W, z_gamma


# def get_pointwise_mutual_info(img, label, w=1.8):
#     # img : (3, 32, 32)
#     # label : int

#     with torch.no_grad():
#         device = torch.device("cuda:0")

#         # # load unconditional UNet
#         # uncond_model = UncondUNet(T=1000, ch=128, ch_mult=[1, 2, 3, 4], attn=[2],
#         #                 num_res_blocks=2, dropout=0.)
#         # ckpt = torch.load(os.path.join(
#         #     "./Checkpoints/", "ckpt_199_.pt"), map_location=device)
#         # uncond_model.load_state_dict(ckpt)
#         # uncond_model.eval()
#         # uncond_model.to(device)

#         # load conditional UNet
#         model = UNet(T=500, num_labels=10, ch=128, ch_mult=[1, 2, 2, 2],
#                         num_res_blocks=2, dropout=0.15).to(device)
#         ckpt = torch.load(os.path.join(
#             "./CheckpointsCondition/", "ckpt_63_.pt"), map_location=device)
#         model.load_state_dict(ckpt)
#         model.eval()
#         model.to(device)

#         t = torch.arange(499, -1, -1).to(device)
#         betas = torch.linspace(1e-4, 0.028, 500).double()
#         alphas = 1. - betas
#         alphas_bar = torch.flip(torch.cumprod(alphas, dim=0), [0]).to(device)
#         snr = (alphas_bar / torch.sqrt(1. - alphas_bar))

#         W_gamma, delta_W, z_gamma = simulate_BM(img, snr)

#         alphas_bar_reshaped = alphas_bar[:, None, None, None].to(torch.float32)
#         z_gamma = z_gamma.to(torch.float32)
#         x_t = z_gamma * torch.sqrt((1 - alphas_bar_reshaped) / alphas_bar_reshaped)
#         labels = torch.full((500,), label, dtype=torch.int).to(device)

#         eps_list = []
#         noneps_list = []

#         for i in range(500):
#             x_t_single = x_t[i:i+1]
#             t_single = t[i:i+1]
#             labels_single = labels[i:i+1]
            
#             eps_single = model(x_t_single, t_single, labels_single)
#             nonEps_single = model(x_t_single, t_single, torch.zeros_like(labels_single))
            
#             eps_comb = (1. + w) * eps_single - w * nonEps_single
            
#             eps_list.append(eps_comb.cpu())
#             noneps_list.append(nonEps_single.cpu())

#         eps_concatenated = torch.cat(eps_list, dim=0).to(device)
#         noneps_concatenated = torch.cat(noneps_list, dim=0).to(device)

#         alphas_bar_reshaped = alphas_bar[:, None, None, None]
#         x_hat_cond = 1 / torch.sqrt(alphas_bar_reshaped) * (x_t - torch.sqrt(1 - alphas_bar_reshaped) * eps_concatenated)
#         x_hat_uncond = 1 / torch.sqrt(alphas_bar_reshaped) * (x_t - torch.sqrt(1 - alphas_bar_reshaped) * noneps_concatenated)

#         # Reshape sampledImg for broadcasting
#         img_reshaped = img[None, :, :, :]

#         # Compute the squared L2 norms
#         diff_unconditional = img_reshaped - x_hat_uncond
#         diff_tensor = img_reshaped - x_hat_cond

#         squared_l2_unconditional = (diff_unconditional ** 2).sum(dim=(1, 2, 3))
#         squared_l2_tensor = (diff_tensor ** 2).sum(dim=(1, 2, 3))

#         # Compute the differences
#         difference = squared_l2_unconditional - squared_l2_tensor

#         # Compute the SNR differences; note the inversion for decreasing snr
#         snr_diff = snr[1:] - snr[:-1]

#         # Perform the Riemann sum
#         integral_approximation = torch.sum(difference[:-1] * snr_diff, dim=0)

#         difference = x_hat_cond[:-1, :, :, :] - x_hat_uncond[:-1, :, :, :]
#         ito = (difference * delta_W).sum()

#     return 0.5 * integral_approximation, 0.5 * integral_approximation + ito

import torch
import os
from DiffusionFreeGuidence.ModelCondition import UNet

def simulate_BM(x, gamma_list):
    gamma_list = gamma_list.clone().detach().to(x.device)
    delta_gamma = torch.diff(gamma_list).to(x.device)
    delta_W = torch.randn((len(delta_gamma),) + x.shape, device=x.device) * torch.sqrt(delta_gamma).view(-1, 1, 1, 1).to(x.device)
    W_initial = torch.randn_like(x) * torch.sqrt(gamma_list[0]).to(x.device)
    W_gamma = torch.cat([W_initial.unsqueeze(0), torch.cumsum(delta_W, dim=0)], dim=0).to(x.device)
    z_gamma = gamma_list.view(-1, 1, 1, 1).to(x.device) * x + W_gamma
    return W_gamma, delta_W, z_gamma

def get_pointwise_mutual_info(img, label, w=1.8):
    with torch.no_grad():
        device = torch.device("cuda:0")
        model = UNet(T=500, num_labels=10, ch=128, ch_mult=[1, 2, 2, 2], num_res_blocks=2, dropout=0.15).to(device)
        ckpt = torch.load(os.path.join("./CheckpointsCondition/", "ckpt_63_.pt"), map_location=device)
        model.load_state_dict(ckpt)
        model.eval().to(device)
        
        t = torch.arange(499, -1, -1).to(device)
        betas = torch.linspace(1e-4, 0.028, 500).double()
        alphas = 1. - betas
        alphas_bar = torch.flip(torch.cumprod(alphas, dim=0), [0]).to(device)
        snr = (alphas_bar / torch.sqrt(1. - alphas_bar))
        
        W_gamma, delta_W, z_gamma = simulate_BM(img, snr)
        
        alphas_bar_reshaped = alphas_bar[:, None, None, None].to(torch.float32)
        z_gamma = z_gamma.to(torch.float32)
        x_t = z_gamma * torch.sqrt((1 - alphas_bar_reshaped) / alphas_bar_reshaped)
        
        labels = torch.full((500,), label, dtype=torch.int).to(device)
        eps_list = []
        noneps_list = []

        for i in range(500):
            x_t_single = x_t[i:i+1]
            t_single = t[i:i+1]
            labels_single = labels[i:i+1]
            eps_single = model(x_t_single, t_single, labels_single)
            nonEps_single = model(x_t_single, t_single, torch.zeros_like(labels_single))
            eps_comb = (1. + w) * eps_single - w * nonEps_single
            eps_list.append(eps_comb.cpu())
            noneps_list.append(nonEps_single.cpu())

        eps_concatenated = torch.cat(eps_list, dim=0).to(device)
        noneps_concatenated = torch.cat(noneps_list, dim=0).to(device)

        alphas_bar_reshaped = alphas_bar[:, None, None, None]
        x_hat_cond = 1 / torch.sqrt(alphas_bar_reshaped) * (x_t - torch.sqrt(1 - alphas_bar_reshaped) * eps_concatenated)
        x_hat_uncond = 1 / torch.sqrt(alphas_bar_reshaped) * (x_t - torch.sqrt(1 - alphas_bar_reshaped) * noneps_concatenated)

        img_reshaped = img[None, :, :, :]

        diff_unconditional = img_reshaped - x_hat_uncond
        diff_tensor = img_reshaped - x_hat_cond

        squared_l2_unconditional = (diff_unconditional ** 2).sum(dim=(1, 2, 3))
        squared_l2_tensor = (diff_tensor ** 2).sum(dim=(1, 2, 3))

        difference = squared_l2_unconditional - squared_l2_tensor

        snr_diff = snr[1:] - snr[:-1]

        integral_approximation = torch.sum(difference[:-1] * snr_diff, dim=0)

        difference = x_hat_cond[:-1, :, :, :] - x_hat_uncond[:-1, :, :, :]
        ito = (difference * delta_W).sum()

    return 0.5 * integral_approximation, 0.5 * integral_approximation + ito

def get_pointwise_mutual_info_2(img, label, w=1.8):
    with torch.no_grad():
        device = torch.device("cuda:0")
        model = UNet(T=500, num_labels=10, ch=128, ch_mult=[1, 2, 2, 2], num_res_blocks=2, dropout=0.15).to(device)
        ckpt = torch.load(os.path.join("./CheckpointsCondition/", "ckpt_63_.pt"), map_location=device)
        model.load_state_dict(ckpt)
        model.eval()
        
        t = torch.arange(499, -1, -1).to(device)
        betas = torch.linspace(1e-4, 0.028, 500).double()
        alphas = 1. - betas
        alphas_bar = torch.flip(torch.cumprod(alphas, dim=0), [0]).to(device)
        snr = (alphas_bar / torch.sqrt(1. - alphas_bar))
        
        W_gamma, delta_W, z_gamma = simulate_BM(img, snr)
        
        alphas_bar_reshaped = alphas_bar[:, None, None, None].to(torch.float32)
        z_gamma = z_gamma.to(torch.float32)
        x_t = z_gamma * torch.sqrt((1 - alphas_bar_reshaped) / alphas_bar_reshaped)
        
        labels = torch.full((500,), label, dtype=torch.int).to(device)
        
        eps = model(x_t, t, labels)
        nonEps = model(x_t, t, torch.zeros_like(labels))

        eps_comb = (1. + w) * eps - w * nonEps
        
        alphas_bar_reshaped = alphas_bar[:, None, None, None]
        x_hat_cond = 1 / torch.sqrt(alphas_bar_reshaped) * (x_t - torch.sqrt(1 - alphas_bar_reshaped) * eps_comb)
        x_hat_uncond = 1 / torch.sqrt(alphas_bar_reshaped) * (x_t - torch.sqrt(1 - alphas_bar_reshaped) * nonEps)
        
        img_reshaped = img[None, :, :, :]

        diff_unconditional = img_reshaped - x_hat_uncond
        diff_tensor = img_reshaped - x_hat_cond

        squared_l2_unconditional = (diff_unconditional ** 2).sum(dim=(1, 2, 3))
        squared_l2_tensor = (diff_tensor ** 2).sum(dim=(1, 2, 3))

        difference = squared_l2_unconditional - squared_l2_tensor

        snr_diff = snr[1:] - snr[:-1]

        integral_approximation = torch.sum(difference[:-1] * snr_diff, dim=0)

        difference = x_hat_cond[:-1, :, :, :] - x_hat_uncond[:-1, :, :, :]
        ito = (difference * delta_W).sum()

    return 0.5 * integral_approximation, 0.5 * integral_approximation + ito

# def get_pointwise_mutual_info_3(img, label, w=1.0):
#     with torch.no_grad():
#         device = torch.device("cuda:0")
#         model = UNet(T=4000, num_labels=10, ch=128, ch_mult=[1, 2, 2, 2], num_res_blocks=2, dropout=0.15).to(device)
#         ckpt = torch.load(os.path.join("./CheckpointsCondition4000/", "ckpt_99_.pt"), map_location=device)
#         model.load_state_dict(ckpt)
#         model.eval()
        
#         t = torch.arange(3999, -1, -1).to(device)
#         betas = torch.linspace(1e-4, 0.028, 4000).double()
#         alphas = 1. - betas
#         alphas_bar = torch.flip(torch.cumprod(alphas, dim=0), [0]).to(device)
#         snr = (alphas_bar / torch.sqrt(1. - alphas_bar))
        
#         W_gamma, delta_W, z_gamma = simulate_BM(img, snr)
        
#         alphas_bar_reshaped = alphas_bar[:, None, None, None].to(torch.float32)
#         z_gamma = z_gamma.to(torch.float32)
#         x_t = z_gamma * torch.sqrt((1 - alphas_bar_reshaped) / alphas_bar_reshaped)
        
#         labels = torch.full((4000,), label, dtype=torch.int).to(device)
        
#         eps = model(x_t, t, labels)
#         nonEps = model(x_t, t, torch.zeros_like(labels))

#         eps_comb = (1. + w) * eps - w * nonEps
        
#         alphas_bar_reshaped = alphas_bar[:, None, None, None]
#         x_hat_cond = 1 / torch.sqrt(alphas_bar_reshaped) * (x_t - torch.sqrt(1 - alphas_bar_reshaped) * eps_comb)
#         x_hat_uncond = 1 / torch.sqrt(alphas_bar_reshaped) * (x_t - torch.sqrt(1 - alphas_bar_reshaped) * nonEps)
        
#         img_reshaped = img[None, :, :, :]

#         diff_unconditional = img_reshaped - x_hat_uncond
#         diff_tensor = img_reshaped - x_hat_cond

#         squared_l2_unconditional = (diff_unconditional ** 2).sum(dim=(1, 2, 3))
#         squared_l2_tensor = (diff_tensor ** 2).sum(dim=(1, 2, 3))

#         difference = squared_l2_unconditional - squared_l2_tensor

#         snr_diff = snr[1:] - snr[:-1]

#         integral_approximation = torch.sum(difference[:-1] * snr_diff, dim=0)

#         difference = x_hat_cond[:-1, :, :, :] - x_hat_uncond[:-1, :, :, :]
#         ito = (difference * delta_W).sum()

#     return 0.5 * integral_approximation, 0.5 * integral_approximation + ito


def get_pointwise_mutual_info_3(img, label, w=1.0):
    with torch.no_grad():
        device = torch.device("cuda:0")
        model = UNet(T=4000, num_labels=10, ch=128, ch_mult=[1, 2, 2, 2], num_res_blocks=2, dropout=0.15).to(device)
        ckpt = torch.load(os.path.join("./CheckpointsCondition4000/", "ckpt_99_.pt"), map_location=device)
        model.load_state_dict(ckpt)
        model.eval().to(device)
        
        integral_approximation = 0.0
        ito = 0.0

        for i in range(4):
            start_idx = i * 1000
            end_idx = start_idx + 1000
            
            t = torch.arange(end_idx-1, start_idx-1, -1).to(device)
            betas = torch.linspace(1e-4, 0.028, 4000)[start_idx:end_idx].double()
            alphas = 1. - betas
            alphas_bar = torch.flip(torch.cumprod(alphas, dim=0), [0]).to(device)
            snr = (alphas_bar / torch.sqrt(1. - alphas_bar))

            W_gamma, delta_W, z_gamma = simulate_BM(img, snr)

            # The rest of your code with adjustments to slicing based on start_idx and end_idx
            
            alphas_bar_reshaped = alphas_bar[:, None, None, None].to(torch.float32)
            z_gamma = z_gamma.to(torch.float32)
            x_t = z_gamma * torch.sqrt((1 - alphas_bar_reshaped) / alphas_bar_reshaped)
            
            labels = torch.full((1000,), label, dtype=torch.int).to(device)
            
            eps = model(x_t, t, labels)
            nonEps = model(x_t, t, torch.zeros_like(labels))
            
            eps_comb = (1. + w) * eps - w * nonEps

            alphas_bar_reshaped = alphas_bar[:, None, None, None]
            x_hat_cond = 1 / torch.sqrt(alphas_bar_reshaped) * (x_t - torch.sqrt(1 - alphas_bar_reshaped) * eps_comb)
            x_hat_uncond = 1 / torch.sqrt(alphas_bar_reshaped) * (x_t - torch.sqrt(1 - alphas_bar_reshaped) * nonEps)

            img_reshaped = img[None, :, :, :]

            diff_unconditional = img_reshaped - x_hat_uncond
            diff_tensor = img_reshaped - x_hat_cond

            squared_l2_unconditional = (diff_unconditional ** 2).sum(dim=(1, 2, 3))
            squared_l2_tensor = (diff_tensor ** 2).sum(dim=(1, 2, 3))

            difference = squared_l2_unconditional - squared_l2_tensor

            snr_diff = snr[1:] - snr[:-1]

            integral_approximation += torch.sum(difference[:-1] * snr_diff, dim=0)

            difference = x_hat_cond[:-1, :, :, :] - x_hat_uncond[:-1, :, :, :]
            ito += (difference * delta_W).sum()

    return 0.5 * integral_approximation, 0.5 * integral_approximation + ito
