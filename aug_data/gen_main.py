import sys, os
sys.path.insert(0, "./")

from functools import partial

import d4rl
import torch
import torch.nn as nn
import numpy as np
from rlf import run_policy, evaluate_policy
from rlf.algos import BaseAlgo, BehavioralCloning, DBC
from rlf.algos.il.base_il import BaseILAlgo
from rlf.algos.nested_algo import NestedAlgo
from rlf.policies import BasicPolicy
from rlf.args import str2bool
from rlf.rl.loggers.base_logger import BaseLogger
from rlf.rl.loggers.wb_logger import WbLogger
# from rlf.rl.loggers.wb_logger import WbLogger, get_wb_ray_config, get_wb_ray_kwargs
from rlf.rl.model import MLPBase, MLPBasic
from rlf.run_settings import RunSettings

import dbc.envs.ball_in_cup
import dbc.envs.d4rl
import dbc.envs.fetch
import dbc.envs.goal_check
import dbc.envs.gridworld
import dbc.envs.hand
import dbc.gym_minigrid
from dbc.envs.goal_traj_saver import GoalTrajSaver
from dbc.utils import trim_episodes_trans
from dbc.models import GwImgEncoder
from dbc.policies.grid_world_expert import GridWorldExpert
from typing import Dict, Optional, Tuple
from torch.utils.data import Dataset
import time

from gen_ddpm import p_sample, p_sample_loop, norm_vec, sigmoid_beta_schedule, MLPDiffusion
import argparse

def gen_new_data(model, x_0, alphas_bar_sqrt, one_minus_alphas_bar_sqrt, n_steps):
    t = torch.ones_like(x_0, dtype=torch.long).to(device) * n_steps
    # coefficient of x0
    a = alphas_bar_sqrt[t]
    # coefficient of eps
    aml = one_minus_alphas_bar_sqrt[t]
    # generate random noise eps
    e = torch.randn_like(x_0).to(device)
    # model input
    # x_T = x_0 * a + e * aml
    x_T = e

# def inference()

if __name__ == "__main__":
    # print()
    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    parser = argparse.ArgumentParser()
    parser.add_argument('--traj-load-path', type=str, default='expert_datasets/maze.pt')
    parser.add_argument('--hidden-dim', type=int, default=128)
    parser.add_argument('--depth', type=int, default=4)
    parser.add_argument('--norm', type=str2bool, default=True)
    parser.add_argument('--lr', type=float, default=1e-4)
    # parser.add_argument('--scheduler-type', type=str, default='linear')
    # parser.add_argument('--achieve-range', type=float, default=0.1)
    # parser.add_argument('--data', type=int, default=100)
    parser.add_argument('--num-epoch', type=int, default=8000)
    parser.add_argument('--prefix', type=str, default='')
    args = parser.parse_args()
    print(f"Hidden dimension = {args.hidden_dim}")
    print(f"Depth = {args.depth}")

    ########### hyper parameter
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    batch_size = 128
    num_epoch = args.num_epoch
    num_steps = 1000
    betas = sigmoid_beta_schedule(num_steps)

    model_save_path = 'data/dm/trained_models'
    if not os.path.exists(model_save_path):
        os.makedirs(model_save_path, exist_ok=True)
    image_save_path = 'data/dm/trained_imgs'
    if not os.path.exists(image_save_path):
        os.makedirs(image_save_path, exist_ok=True)

    betas = torch.clip(betas, 0.0001, 0.9999).to(device)

    # calculate alpha、alpha_prod、alpha_prod_previous、alpha_bar_sqrt
    alphas = 1 - betas
    alphas_prod = torch.cumprod(alphas, 0)
    alphas_prod_p = torch.cat(
        [torch.tensor([1]).float().to(device), alphas_prod[:-1]], 0
    )
    alphas_bar_sqrt = torch.sqrt(alphas_prod)
    one_minus_alphas_bar_log = torch.log(1 - alphas_prod)
    one_minus_alphas_bar_sqrt = torch.sqrt(1 - alphas_prod)

    assert (
        alphas.shape
        == alphas_prod.shape
        == alphas_prod_p.shape
        == alphas_bar_sqrt.shape
        == one_minus_alphas_bar_log.shape
        == one_minus_alphas_bar_sqrt.shape
    )
    print("all the same shape", betas.shape)

    env = args.traj_load_path.split('/')[-1][:-3]
    data = torch.load(args.traj_load_path)
    # print(data)
    # Demonstration normalization
    obs = data["obs"]
    print(obs.shape)
    if args.norm:
        obs_mean = obs.mean(0)
        obs_std = obs.std(0)
        print(f"obs std: {obs_std}")
        obs = norm_vec(obs, obs_mean, obs_std)

    actions = data["actions"]
    print(actions.shape)
    if args.norm:
        actions_mean = actions.mean(0)
        actions_std = actions.std(0)
        print(f"actions std: {actions_std}")
        actions = norm_vec(actions, actions_mean, actions_std)

    dataset = torch.cat((obs, actions), 1)
    sample_num = dataset.size()[0]
    if sample_num % 2 == 1:
        dataset = dataset[1:sample_num, :]
    print("after", dataset.size())
    print("actions.dtype:", actions.dtype)

    print("Training model...")
    dataloader = torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, shuffle=True
    )
    model = MLPDiffusion(
        num_steps,
        input_dim=dataset.shape[1],
        num_units=args.hidden_dim,
        depth=args.depth,
    )
    model_path = os.path.join(model_save_path, f"{env}_ddpm.pt")
    model.load_state_dict(torch.load(model_path, weights_only=True))
    model = model.to(device)
    model.eval()

    with torch.no_grad():
        out_seq = p_sample_loop(
            model,
            dataset.squeeze().shape,
            n_steps=num_steps-1,
            betas=betas,
            one_minus_alphas_bar_sqrt=one_minus_alphas_bar_sqrt
        )
        out = out_seq[-1]
        out = out.cpu().detach()
        # fig, axs = plt.subplots(1, 2, figsize=(14, 6))
        # axs[0].scatter(dataset[:5000, 0], dataset[:5000, 1], color='blue', edgecolor='white')
        # axs[1].scatter(out[:5000, 0], out[:5000, 1], color='red', edgecolor='white')
        # plt.savefig(f'data/dm/trained_imgs/{env}-pos.png')
        # plt.close()