import torch
from torch import nn
from torch.nn import functional as F
import gen_ddpm
from tqdm import tqdm

import gym
import os
import d4rl  # Import required to register environments


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

def parse_trajectories(trajectories, state_dim):
    parsed_data = {'obs': [], 'action': [], 'next_obs': []}
    for i in range(len(trajectories) - 1):
        obs = torch.tensor(trajectories[i][:, :state_dim])
        action = torch.tensor(trajectories[i][:, state_dim:])
        next_obs = torch.tensor(trajectories[i + 1][:, :state_dim])
        if len(parsed_data['obs']) == 0:
            parsed_data['obs'] = obs
            parsed_data['next_obs'] = next_obs
            parsed_data['action'] = action
        else:
            parsed_data['obs'] = torch.cat((parsed_data['obs'], obs), dim=0)
            parsed_data['next_obs'] = torch.cat((parsed_data['next_obs'], next_obs), dim=0)
            parsed_data['action'] = torch.cat((parsed_data['action'], action), dim=0)
    return parsed_data





# Print or visualize the generated trajectories


# state_dim = 6
# def parse_trajectories(trajectories, state_dim):
#     parsed_data = {'obs': [], 'action': [], 'next_obs': []}
#     for i in range(len(trajectories) - 1):
#         obs = torch.tensor(trajectories[i][:, :state_dim])
#         action = torch.tensor(trajectories[i][:, state_dim:])
#         next_obs = torch.tensor(trajectories[i + 1][:, :state_dim])
#         if len(parsed_data['obs']) == 0:
#             parsed_data['obs'] = obs
#             parsed_data['next_obs'] = next_obs
#             parsed_data['action'] = action
#         else:
#             parsed_data['obs'] = torch.cat((parsed_data['obs'], obs), dim=0)
#             parsed_data['next_obs'] = torch.cat((parsed_data['next_obs'], next_obs), dim=0)
#             parsed_data['action'] = torch.cat((parsed_data['action'], action), dim=0)
#     return parsed_data


if __name__=="__main__":
    # Load the pretrained model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # input_dim = 8  # Input dimension (should match the training configuration)
    num_steps = 1000
    depth = 4  # Depth of the model (should match the training configuration)
    # dim = 8

    env_name = "pick"
    hidden_dim = 1024  # Hidden dimension (should match the training configuration)
    
    # num_units=1024
    
    dataset_path = f"../expert_datasets/{env_name}.pt"
    data = torch.load(dataset_path)
    obs = data["obs"]
    actions = data["actions"]
    trj_step_num = obs.shape[0]
    state_dim = obs.shape[1]
    action_dim = actions.shape[1]
    input_dim = state_dim + action_dim
    
    
    # print(obs.shape)
    # if args.norm:
    #     obs_mean = obs.mean(0)
    #     obs_std = obs.std(0)
    #     print(f"obs std: {obs_std}")
    #     obs = norm_vec(obs, obs_mean, obs_std)

    # print(actions.shape)
    # if args.norm:
    #     actions_mean = actions.mean(0)
    #     actions_std = actions.std(0)
    #     print(f"actions std: {actions_std}")
    #     actions = norm_vec(actions, actions_mean, actions_std)
    
    # trj_step_num = 18525
    # state_dim = 6
    # action_dim = input_dim - state_dim

    model = gen_ddpm.MLPDiffusion(num_steps, input_dim=input_dim, num_units=hidden_dim, depth=depth).to(device)
    ##TODO fix the path
    model.load_state_dict(torch.load(f'../data/dm/trained_models/{env_name}_ddpm.pt'))
    model.eval()
    
    betas = sigmoid_beta_schedule(num_steps).to(device)
    betas = torch.clip(betas, 0.0001, 0.9999).to(device)
    alphas = 1 - betas
    alphas_prod = torch.cumprod(alphas, 0)
    one_minus_alphas_bar_sqrt = torch.sqrt(1 - alphas_prod)

    # Generate expert trajectories
    shape = (1, input_dim)  # Shape of the input (batch_size, input_dim)
    output_dir = '../expert_datasets'
    os.makedirs(output_dir, exist_ok=True)
    
    trajectories = []
    for i in tqdm(range(trj_step_num)):
        trj = p_sample_loop(model, shape, num_steps, betas, one_minus_alphas_bar_sqrt)
        trajectories.append(trj.cpu().detach())
        if i % 100 == 0:
            # trajectories = [traj.cpu().detach().numpy() for traj in trajectories]
            print(f"Generated {i} trajectories")

            
            expert_dataset = parse_trajectories(trajectories, state_dim)
            
            print(expert_dataset['obs'])
            # expert_dataset = {'trajectories': parsed_trajectories }
            # output_dir = '../expert_datasets'
            # os.makedirs(output_dir, exist_ok=True)
            torch.save(expert_dataset, os.path.join(output_dir, f'{env_name}_try_{i}.pt'))
            
            # print(f"Obs: {expert_dataset['obs'][-1]}")
            # print(f"Action: {expert_dataset['action'][-1]}")
            # print(f"Next Obs: {expert_dataset['next_obs'][-2]}")
            # print(f"Next Obs: {expert_dataset['next_obs'][-1]}")

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




