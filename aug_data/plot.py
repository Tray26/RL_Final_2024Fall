from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import argparse

def norm_vec(x, mean, std):
    obs_x = torch.clamp(
        (x - mean) / (std + 1e-8),
        -10.0,
        10.0,
    )
    return obs_x


def norm(obs, actions):
    
    obs_mean = obs.mean(0)
    obs_std = obs.std(0)
    print(f"obs std: {obs_std}")
    obs = norm_vec(obs, obs_mean, obs_std)

    actions_mean = actions.mean(0)
    actions_std = actions.std(0)
    print(f"actions std: {actions_std}")
    actions = norm_vec(actions, actions_mean, actions_std)

    return obs, actions


data = torch.load('./expert_datasets/maze.pt')
out = torch.load('./gen_datasets/maze/step1000/step1000seed0_denorm.pt')
env = 'maze_step1000'
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
batch_size = 128
num_steps = 1000

# Demonstration normalization
obs, actions = norm(data["obs"], data["actions"])
obs_g, actions_g = norm(out["obs"], out["actions"])

dataset = torch.cat((obs, actions), 1)
out = torch.cat((obs_g, actions_g), 1)

sample_num = dataset.size()[0]
if sample_num % 2 == 1:
    dataset = dataset[1:sample_num, :]
    out = out[1:sample_num, :]

# print("after", dataset.size(), out.size())
# print("actions.dtype:", actions.dtype)


fig, axs = plt.subplots(1, 2, figsize=(14, 6))
axs[0].scatter(dataset[:5000, 0], dataset[:5000, 1], color='blue', edgecolor='white')
axs[1].scatter(out[:5000, 0], out[:5000, 1], color='red', edgecolor='white')
plt.savefig(f'data/trained_imgs/{env}-pos.png') #
plt.close()


