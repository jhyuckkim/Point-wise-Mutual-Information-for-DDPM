import torch
import os
from DiffusionFreeGuidence.ModelCondition import UNet

def simulate_BM(x, gamma_list):
    # Given a data and list of snr, simulates a single Brownian motion path
    # z_gamma = gamma * x + W_gamma, W_gamma ~ N(0, gamma)
    
    gamma_list = gamma_list.clone().detach().to(x.device)
    delta_gamma = torch.diff(gamma_list).to(x.device)
    delta_W = torch.randn((len(delta_gamma),) + x.shape, device=x.device) * torch.sqrt(delta_gamma).view(-1, 1, 1, 1).to(x.device)
    W_initial = torch.randn_like(x) * torch.sqrt(gamma_list[0]).to(x.device)
    W_gamma = torch.cat([W_initial.unsqueeze(0), torch.cumsum(delta_W, dim=0)], dim=0).to(x.device)
    z_gamma = gamma_list.view(-1, 1, 1, 1).to(x.device) * x + W_gamma
    return W_gamma, delta_W, z_gamma


def get_intermediate_pointwise_mutual_info(img, label, model_num, t, w=1.0):
    with torch.no_grad():

        # Load model
        device = torch.device("cuda:0")
        model = UNet(T=4000, num_labels=10, ch=128, ch_mult=[1, 2, 2, 2], num_res_blocks=2, dropout=0.15).to(device)
        ckpt = torch.load(os.path.join("./CheckpointsCondition4000/Model_"+str(model_num), "ckpt_99_.pt"), map_location=device)
        model.load_state_dict(ckpt)
        model.eval().to(device)
        
        # Initialize two terms
        standard_integral = 0.0
        ito_integral = 0.0

        # Conventional notations used in DDPM paper
        betas = torch.linspace(1e-4, 0.028, 4000).double()
        alphas = 1. - betas
        alphas_bar = torch.cumprod(alphas, dim=0).to(device)
        
        # alpha bar calculated from t (intermediate timestep)
        alphas_bar_from_t = torch.cumprod(alphas[t:], dim=0).to(device)
        # snr when we consider input at timestep t
        snrs = (alphas_bar_from_t / (1. - alphas_bar_from_t))

        # Simulate BM from t taking account of new snr values above
        W_gamma, delta_Ws, z_gammas = simulate_BM(img, torch.flip(snrs, [0]))
        W_gamma = torch.flip(W_gamma, [0])
        delta_Ws = torch.flip(delta_Ws, [0])
        delta_Ws = torch.cat((torch.zeros((1, 3, 32, 32)).to(device), delta_Ws), dim=0)
        z_gammas = torch.flip(z_gammas, [0])

        # Scale z_gamma approriately to get x_t
        alphas_bar_from_t_reshaped = alphas_bar_from_t[:, None, None, None].to(torch.float32)
        z_gammas = z_gammas.to(torch.float32)
        x_ts = z_gammas * (1 - alphas_bar_from_t_reshaped) / torch.sqrt(alphas_bar_from_t_reshaped)

        size = 4000 - t

        # Limit maximum batch size to 1000
        for i in range((size-1)//1000+1):
            # timestep t corresponds to index 0
            start_idx = i*1000
            end_idx = min((i+1)*1000-1, size - 1)

            # Truncate appropriately
            snr = snrs[start_idx:end_idx+1]
            x_t = x_ts[start_idx:end_idx+1]
            delta_W = delta_Ws[start_idx:end_idx+1]

            original_t_values = torch.arange(start_idx + t, end_idx + t + 1, 1).to(device)

            labels = torch.full((end_idx - start_idx + 1,), label, dtype=torch.int).to(device)

            # Inference
            eps = model(x_t, original_t_values, labels)
            nonEps = model(x_t, original_t_values, torch.zeros_like(labels))
            eps_comb = (1. + w) * eps - w * nonEps

            # Calculate coefficients
            alpha_bar = alphas_bar[start_idx + t:end_idx + t + 1]
            alpha_bar_reshaped = alpha_bar[:, None, None, None].to(torch.float32)

            alpha_bar_from_t = alphas_bar_from_t[start_idx:end_idx + 1]
            alpha_bar_from_t_reshaped = alpha_bar_from_t[:, None, None, None].to(torch.float32)

            coeff1 = torch.sqrt(1 / alpha_bar_from_t_reshaped)
            coeff2 = (1 - alpha_bar_from_t_reshaped) / torch.sqrt(1 - alpha_bar_reshaped)

            # Calculate mmse estimates of x_t
            x_hat_cond = coeff1 * (x_t - coeff2 * eps_comb)
            x_hat_uncond = coeff1 * (x_t - coeff2 * nonEps)

            img_reshaped = img[None, :, :, :]

            diff_unconditional = img_reshaped - x_hat_uncond
            diff_tensor = img_reshaped - x_hat_cond

            squared_l2_unconditional = (diff_unconditional ** 2).sum(dim=(1, 2, 3))
            squared_l2_tensor = (diff_tensor ** 2).sum(dim=(1, 2, 3))

            difference = squared_l2_unconditional - squared_l2_tensor

            snr_diff = -1 * (snr[1:] - snr[:-1])

            standard_integral += torch.sum(difference[1:] * snr_diff, dim=0)

            difference = x_hat_cond[:, :, :, :] - x_hat_uncond[:, :, :, :]
            
            ito_integral += (difference * delta_W).sum()

    return 0.5 * standard_integral, 0.5 * standard_integral + ito_integral