import torch
from torch import nn
from torch.nn import functional as F
from torch.nn import MSELoss
import gen_ddpm
from tqdm import tqdm

import gym
import os
import d4rl  # Import required to register environments
import argparse

from gen_ddpm import reconstruct, str2bool, norm_vec


def p_sample(model, x, t, betas, one_minus_alphas_bar_sqrt):
    t = torch.tensor([t]).to(device)
    coeff = betas[t] / one_minus_alphas_bar_sqrt[t]
    eps_theta = model(x, t)
    mean = (1 / (1 - betas[t]).sqrt()) * (x - (coeff * eps_theta))
    z = torch.randn_like(x)
    sigma_t = betas[t].sqrt()
    sample = mean + sigma_t * z
    return sample

def p_sample_loop(model, shape, n_steps, betas, one_minus_alphas_bar_sqrt):
    cur_x = torch.randn(shape).to(device)
    # x_seq = [cur_x]
    for i in reversed(range(n_steps)):
        cur_x = p_sample(model, cur_x, i, betas, one_minus_alphas_bar_sqrt)
        # x_seq.append(cur_x)
    return cur_x

# Define the beta schedule and other parameters
def sigmoid_beta_schedule(timesteps, start=0.0001, end=0.02):
    betas = torch.linspace(-6, 6, timesteps)
    return torch.sigmoid(betas) * (end - start) + start

def parse_trajectories(trajectories, state_dim, done):
    parsed_data = {'obs': [], 'actions': [], 'next_obs': []}
    parsed_data['obs'] = trajectories[:, :state_dim].clone().detach()
    parsed_data['next_obs'] = trajectories[1:, :state_dim].clone().detach()
    parsed_data['actions'] = trajectories[:, state_dim:].clone().detach()
    parsed_data['done'] = done[:].clone().detach()
    return parsed_data



if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--reconstruct', type=str2bool, default=False)
    # parser.add_argument('--num_step_max', type=int, default=1000)
    parser.add_argument('--num_step', type=int, default=1000)
    parser.add_argument('--ddpm_depth', type=int, default=4)
    parser.add_argument('--env_name', type=str, default='maze')
    parser.add_argument('--hidden_dim', type=int, default=1024)
    parser.add_argument('--expert_data', type=str, default="../expert_datasets")
    parser.add_argument('--ddpm_model_dir', type=str, default='../data/dm/trained_models/')
    parser.add_argument('--output_dir', type=str, default='../gen_datasets')

    args = parser.parse_args()

    print(f"Hidden dimension = {args.hidden_dim}")
    print(f"Depth = {args.ddpm_depth}")
    
    print(args)

    # parameters
    train_steps = 1000
    depth = args.ddpm_depth
    env_name = args.env_name
    hidden_dim = args.hidden_dim
    expert_dir = args.expert_data
    ddpm_model_dir = args.ddpm_model_dir
    num_steps = args.num_step
    output_dir = args.output_dir
    
    os.makedirs(output_dir, exist_ok=True)
    


    
    dataset_path = os.path.join(expert_dir, f"{env_name}.pt")
    # dataset_path = f"../expert_datasets/{env_name}.pt"
    data = torch.load(dataset_path)
    obs = data["obs"]
    actions = data["actions"]
    done = data["done"]
    next_obs=data["next_obs"]
    # print(obs[-1,:])
    # print(next_obs[-1,:])    
    trj_step_num = obs.shape[0]
    state_dim = obs.shape[1]
    action_dim = actions.shape[1]
    input_dim = state_dim + action_dim
    
    # if args.norm:
    obs_mean = obs.mean(0)
    obs_std = obs.std(0)
    print(f"obs std: {obs_std}")
    obs = norm_vec(obs, obs_mean, obs_std)
    # if args.norm:
    actions_mean = actions.mean(0)
    actions_std = actions.std(0)
    print(f"actions std: {actions_std}")
    actions = norm_vec(actions, actions_mean, actions_std)
    
    expert_data = torch.cat((obs, actions), 1)

    
    # Load the pretrained model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = gen_ddpm.MLPDiffusion(train_steps, input_dim=input_dim, num_units=hidden_dim, depth=depth)
    ##TODO fix the path
    model_path = os.path.join(ddpm_model_dir, f"{env_name}_ddpm.pt")
    model.load_state_dict(torch.load(model_path))
    model.to(device)
    # model.load_state_dict(torch.load(f'../data/dm/trained_models/{env_name}_ddpm.pt'))
    model.eval()
    
    betas = sigmoid_beta_schedule(train_steps).to(device)
    betas = torch.clip(betas, 0.0001, 0.9999).to(device)
    alphas = 1 - betas
    alphas_prod = torch.cumprod(alphas, 0)
    print(alphas_prod)
    one_minus_alphas_bar_sqrt = torch.sqrt(1 - alphas_prod)
    print(f"1-alpha_bar_sqrt={one_minus_alphas_bar_sqrt}")
    alphas_bar_sqrt = torch.sqrt(alphas_prod)
    

    # Generate expert trajectories
    # shape = (1, input_dim)  # Shape of the input (batch_size, input_dim)
    # output_dir = '../gen_datasets'
    
    # if args.reconstruct:
    # trajectories = []
    # obs_mean = obs.mean(0)
    # obs_std = obs.std(0)
    # print(f"obs std: {obs_std}")
    # obs = norm_vec(obs, obs_mean, obs_std)
    # # if args.norm:
    # actions_mean = actions.mean(0)
    # actions_std = actions.std(0)
    # print(f"actions std: {actions_std}")
    # actions = norm_vec(actions, actions_mean, actions_std)
    
    # expert_data = torch.cat((obs, actions), 1)
    sample_num = expert_data.size()[0]
    torch.manual_seed(500)
    
    with torch.no_grad():
        out = reconstruct(
            model,
            expert_data.squeeze().to(device),
            alphas_bar_sqrt,
            one_minus_alphas_bar_sqrt,
            num_steps-1,
            betas[:num_steps],
            recon=args.reconstruct
        )
        
        mse = MSELoss()     
        loss = mse(out.cpu().detach(), expert_data)
        
        print(loss.item())
            
            # print(out.shape)
            # out = out.cpu().detach()
            
            # expert_dataset = parse_trajectories(out.cpu().detach(), state_dim, done)
            # # print(expert_dataset['obs'].shape, expert_dataset['next_obs'].shape, expert_dataset['actions'].shape, expert_dataset['done'].shape)
            # expert_dataset["next_obs"] = torch.cat((expert_dataset["next_obs"], next_obs[-1,:].unsqueeze(0)), dim=0)
            # # print(expert_dataset['obs'].shape, expert_dataset['next_obs'].shape, expert_dataset['actions'].shape, expert_dataset['done'].shape)
            
            # torch.save(expert_dataset, os.path.join(output_dir, f'{env_name}_{num_steps}_gen.pt'))
            
        
    # else:
    #     trajectories = []
    #     for i in tqdm(range(trj_step_num)):
    #         trj = p_sample_loop(model, shape, num_steps, betas, one_minus_alphas_bar_sqrt)
    #         trajectories.append(trj.cpu().detach())
    #         if i % 100 == 0:
    #             # trajectories = [traj.cpu().detach().numpy() for traj in trajectories]
    #             print(f"Generated {i} trajectories")

            
    #         expert_dataset = parse_trajectories(trajectories, state_dim)
            
    #         # print(expert_dataset['obs']) 
    #         # expert_dataset = {'trajectories': parsed_trajectories }
    #         # output_dir = '../expert_datasets'
    #         # os.makedirs(output_dir, exist_ok=True)
    #         torch.save(expert_dataset, os.path.join(output_dir, f'{env_name}_gen.pt'))
            
    #         # print(f"Obs: {expert_dataset['obs'][-1]}")
    #         # print(f"Action: {expert_dataset['action'][-1]}")
    #         # print(f"Next Obs: {expert_dataset['next_obs'][-2]}")
    #         # print(f"Next Obs: {expert_dataset['next_obs'][-1]}")

    #     torch.save(expert_dataset, os.path.join(output_dir, f'{env_name}_gen.pt'))

    # Print or visualize the generated trajectories
    # Create the directory if it doesn't exist

    # Convert the generated trajectories to numpy arrays for further processing

    # Print or visualize the generated trajectories
    # Create the directory if it doesn't exist
    
    # parsed_trajectories = parse_trajectories(trajectories, state_dim)
    # expert_dataset = {'trajectories': parsed_trajectories }
    # torch.save(expert_dataset, os.path.join(output_dir, 'Maze_try.pt'))
    # print(parsed_trajectories)

    # Print or visualize the parsed trajectories
    # for i in range(len(parsed_trajectories['obs'])):




