import torch
from torch import nn
from torch.nn import functional as F
from torch.nn import MSELoss
import gen_ddpm
from tqdm import tqdm
import json
import gym
import os
import d4rl  # Import required to register environments
import argparse

from gen_ddpm import reconstruct, str2bool, MLPDiffusion, p_sample
from ddim import DDIM
from utils import parse_trajectories, denorm_vec, norm_vec, sigmoid_beta_schedule

def generate_ddim(model, x_0, n_steps, recon=True, eta=0.0):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ddim = DDIM(train_timestep=1000, model=model)
    t = torch.ones_like(x_0, dtype=torch.long).to(device) * n_steps
    # print(t)
    a = ddim.alpha_bars[t].sqrt()
    # aml = one_minus_alphas_bar_sqrt[t]
    aml = ddim.noise_params[t]
    e = torch.randn_like(x_0).to(device)
    if recon:
        x_T = x_0 * a + e * aml
    else:
        x_T = e
    if x_T.dtype == torch.float64:
        x_T = x_T.to(torch.float32)
    # print(x_0.shape, x_T.shape)
    x_0 = ddim.ddim_sample(x_T, n_steps, eta=eta, ddim_step=1000)
    return x_0

def generate_ddpm(model, x_0, alphas_bar_sqrt, one_minus_alphas_bar_sqrt, n_steps, betas, recon=True):
    # generate random t for a batch data
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    t = torch.ones_like(x_0, dtype=torch.long).to(device) * n_steps
    # coefficient of x0
    a = alphas_bar_sqrt[t]
    # coefficient of eps
    aml = one_minus_alphas_bar_sqrt[t]
    # generate random noise eps
    e = torch.randn_like(x_0).to(device)
    # model input
    # print("Recon: ", recon)
    if recon:
        # print("Generate from expert data")
        x_T = x_0 * a + e * aml
        # mse = MSELoss()
        # diff = mse(x_T, e)
        # print(a, aml)
        # print("Difference between x_T and noise: ", diff)
    else:
        print("Generate from pure noise")
        x_T = e
    if x_T.dtype == torch.float64:
        x_T = x_T.to(torch.float32)
    # generate[T-1]、x[T-2]|...x[0] from x[T]
    for i in reversed(range(n_steps)):
        # for i in reversed(range(1, n_steps+1)):
        # print(f"step {i}", end='\r')
        x_T = p_sample(model, x_T, i, betas, one_minus_alphas_bar_sqrt)
    x_construct = x_T
    return x_construct

def main(seed_list, steps_list, args, train_steps = 1000, norm=True, etas=None):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    depth = args.ddpm_depth
    env_name = args.env_name
    hidden_dim = args.hidden_dim
    expert_dir = args.expert_data
    ddpm_model_dir = args.ddpm_model_dir
    output_dir = args.output_dir

    output_path_env = os.path.join(output_dir, env_name)
    os.makedirs(output_path_env, exist_ok=True)

    expert_path = os.path.join(expert_dir, f"{env_name}.pt")
    expert_data = torch.load(expert_path)
    expert_obs = expert_data["obs"]
    expert_actions = expert_data["actions"]
    expert_done = expert_data["done"]
    expert_next_obs=expert_data["next_obs"]

    state_dim = expert_obs.shape[1]
    action_dim = expert_actions.shape[1]
    input_dim = state_dim + action_dim
    expert_state_mean = expert_obs.mean(0)
    expert_state_std = expert_obs.std(0)
    expert_action_mean = expert_actions.mean(0)
    expert_action_std = expert_actions.std(0)

    if norm:
        expert_obs = norm_vec(expert_obs, expert_state_mean, expert_state_std)
        expert_actions = norm_vec(expert_actions, expert_action_mean, expert_action_std)

    expert_pairs = torch.cat((expert_obs, expert_actions), dim=1).squeeze().to(device)
    
    model = MLPDiffusion(train_steps, input_dim=input_dim, num_units=hidden_dim, depth=depth)
    model_path = os.path.join(ddpm_model_dir, f"{env_name}_ddpm.pt")
    model.load_state_dict(torch.load(model_path))
    model.to(device)
    model.eval()
    
    betas = sigmoid_beta_schedule(train_steps).to(device)
    betas = torch.clip(betas, 0.0001, 0.9999).to(device)
    alphas = 1 - betas
    alphas_prod = torch.cumprod(alphas, 0)
    one_minus_alphas_bar_sqrt = torch.sqrt(1 - alphas_prod)
    alphas_bar_sqrt = torch.sqrt(alphas_prod)

    all_configs = {}
    for denoise_step in steps_list:
        for seed in seed_list:
            torch.manual_seed(seed)
            with torch.no_grad():
                if args.use_ddim:
                    gen_trajs = generate_ddim(
                        model=model,
                        x_0=expert_pairs, 
                        n_steps=denoise_step-1, 
                        recon=args.reconstruct,
                        eta=0.0
                    )
                else:
                    gen_trajs = generate_ddpm(
                        model,
                        expert_pairs,
                        alphas_bar_sqrt,
                        one_minus_alphas_bar_sqrt,
                        denoise_step-1,
                        betas[:denoise_step],
                        recon=args.reconstruct
                    )

            gen_trajs = gen_trajs.cpu().detach()
            gen_states = gen_trajs[:, :state_dim]
            gen_action = gen_trajs[:, state_dim:]

            mse = MSELoss()
            states_loss = mse(gen_states, expert_obs)
            action_loss = mse(gen_action, expert_actions)
            print(f"MSE loss of {env_name}, step{denoise_step}, seed{seed}: {states_loss.item()}, {action_loss.item()}")

            gen_states = denorm_vec(gen_states, expert_state_mean, expert_state_std)
            gen_action = denorm_vec(gen_action, expert_action_mean, expert_action_std)
            gen_trajs = torch.cat((gen_states, gen_action), dim=1).squeeze()
            gen_trajs_set = parse_trajectories(gen_trajs, state_dim, expert_done)
            gen_trajs_set["next_obs"] = torch.cat((gen_trajs_set["next_obs"], expert_next_obs[-1,:].unsqueeze(0)), dim=0)

            

            # gen_traj_set = parse_trajectories(gen)

