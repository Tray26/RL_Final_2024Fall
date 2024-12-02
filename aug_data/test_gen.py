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

from gen_ddpm import reconstruct, str2bool, norm_vec, MLPDiffusion, denorm_vec
from ddim import DDIM

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


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

# def torch_mse(prediction, truth, dim=0):

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

    

def main(seed_list, steps_list, args, train_steps = 1000, norm=True):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    depth = args.ddpm_depth
    env_name = args.env_name
    hidden_dim = args.hidden_dim
    expert_dir = args.expert_data
    ddpm_model_dir = args.ddpm_model_dir
    # num_steps = args.num_step
    output_dir = args.output_dir
    output_path_env = os.path.join(output_dir, env_name)
    os.makedirs(output_path_env, exist_ok=True)
    
    expert_path = os.path.join(expert_dir, f"{env_name}.pt")
    expert_data = torch.load(expert_path)
    expert_obs = expert_data["obs"]
    expert_actions = expert_data["actions"]
    expert_done = expert_data["done"]
    expert_next_obs=expert_data["next_obs"]
    
    trj_step_num = expert_obs.shape[0]
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
    
    all_configs={}
    for denoise_step in steps_list:
        # step_dir_path = os.path.join(output_path_env, f"{denoise_step}")
        for seed in seed_list:
            torch.manual_seed(seed)
            with torch.no_grad():
                if args.use_ddim:
                    out = generate_ddim(
                        model=model,
                        x_0=expert_pairs, 
                        n_steps=denoise_step-1, 
                        recon=args.reconstruct,
                        eta=0.0                    )
                else:
                    out = reconstruct(
                        model,
                        expert_pairs,
                        alphas_bar_sqrt,
                        one_minus_alphas_bar_sqrt,
                        denoise_step-1,
                        betas[:denoise_step],
                        recon=args.reconstruct
                    )

                
                out = out.cpu().detach()
                out_states = out[:, :state_dim]
                out_action = out[:, state_dim:]
                # print("Expert: ", expert_actions.max(), expert_actions.min())
                # print("Generated:", out_action.max(), out_action.min())
                
                mse = MSELoss()     
                # loss = mse(out, expert_pairs)
                states_loss = mse(out_states, expert_obs)
                action_loss = mse(out_action, expert_actions)
                gen_dataset = parse_trajectories(out, state_dim, expert_done)
                gen_dataset["next_obs"] = torch.cat((gen_dataset["next_obs"], expert_next_obs[-1,:].unsqueeze(0)), dim=0)
                
                print(f"MSE loss of {env_name}, step{denoise_step}, seed{seed}: {states_loss.item()}, {action_loss.item()}")
                
                denorm_state = denorm_vec(out_states, expert_state_mean, expert_state_std)
                denorm_action = denorm_vec(out_action, expert_action_mean, expert_action_std)
                denorm_action = torch.clamp(denorm_action, min=-1, max=1)
                
                denorm_out = torch.cat((denorm_state, denorm_action), dim=1).squeeze()
                denorm_gen = parse_trajectories(denorm_out, state_dim, expert_done)
                denorm_gen["next_obs"] = torch.cat((denorm_gen["next_obs"], expert_next_obs[-1,:].unsqueeze(0)), dim=0)
                
                save_config = {
                    "env_name": env_name,
                    "step": denoise_step,
                    "seed": seed,
                    "State MSE": states_loss.item(),
                    "action MSE": action_loss.item()                
                }  
                pt_filename=f"step{denoise_step}seed{seed}.pt"
                denorm_filename = f"step{denoise_step}seed{seed}_denorm.pt"
                all_configs[pt_filename]=save_config
                
                save_dir_path = os.path.join(output_path_env, "data")
                os.makedirs(save_dir_path, exist_ok=True)
                # torch.save(gen_dataset, os.path.join(save_dir_path, pt_filename))
                torch.save(denorm_gen, os.path.join(save_dir_path, denorm_filename))
    with open(os.path.join(output_path_env, "all_configs.json"), "w") as out_json:
        json.dump(all_configs, out_json, indent=4)
        # for key, config in all_configs.items():
        #     json.dump({key: config}, out_json, indent=4)
        #     out_json.write('\n')
                
                
        
            


if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--reconstruct', type=str2bool, default=False)
    # parser.add_argument('--num_step_max', type=int, default=1000)
    # parser.add_argument('--num_step', type=int, default=1000)
    parser.add_argument('--ddpm_depth', type=int, default=4)
    parser.add_argument('--env_name', type=str, default='maze')
    parser.add_argument('--hidden_dim', type=int, default=1024)
    parser.add_argument('--use_ddim', type=str2bool, default=False)
    parser.add_argument('--expert_data', type=str, default="../expert_datasets")
    parser.add_argument('--ddpm_model_dir', type=str, default='../data/dm/trained_models/')
    parser.add_argument('--output_dir', type=str, default='../gen_datasets_denorm')

    args = parser.parse_args()

    print(f"Hidden dimension = {args.hidden_dim}")
    print(f"Depth = {args.ddpm_depth}")
    
    print(args)
    seed_list=range(0, 501, 50)
    # steps_list = range(100, 1001, 100)
    # steps_list = [1, 10, 50]
    # steps_list += range(100, 1001, 100)
    steps_list = [1000]
    print(seed_list)
    print(steps_list)
    
    main(seed_list, steps_list, args)

    # parameters
    # train_steps = 1000
    # depth = args.ddpm_depth
    # env_name = args.env_name
    # hidden_dim = args.hidden_dim
    # expert_dir = args.expert_data
    # ddpm_model_dir = args.ddpm_model_dir
    # num_steps = args.num_step
    # output_dir = args.output_dir
    
    # os.makedirs(output_dir, exist_ok=True)
    
    


    
    # dataset_path = os.path.join(expert_dir, f"{env_name}.pt")
    # # dataset_path = f"../expert_datasets/{env_name}.pt"
    # data = torch.load(dataset_path)
    # obs = data["obs"]
    # actions = data["actions"]
    # done = data["done"]
    # next_obs=data["next_obs"]
    # # print(obs[-1,:])
    # # print(next_obs[-1,:])    
    # trj_step_num = obs.shape[0]
    # state_dim = obs.shape[1]
    # action_dim = actions.shape[1]
    # input_dim = state_dim + action_dim
    
    # # if args.norm:
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

    
    # # Load the pretrained model
    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # model = gen_ddpm.MLPDiffusion(train_steps, input_dim=input_dim, num_units=hidden_dim, depth=depth)
    # ##TODO fix the path
    # model_path = os.path.join(ddpm_model_dir, f"{env_name}_ddpm.pt")
    # model.load_state_dict(torch.load(model_path))
    # model.to(device)
    # # model.load_state_dict(torch.load(f'../data/dm/trained_models/{env_name}_ddpm.pt'))
    # model.eval()
    
    # betas = sigmoid_beta_schedule(train_steps).to(device)
    # betas = torch.clip(betas, 0.0001, 0.9999).to(device)
    # alphas = 1 - betas
    # alphas_prod = torch.cumprod(alphas, 0)
    # print(alphas_prod)
    # one_minus_alphas_bar_sqrt = torch.sqrt(1 - alphas_prod)
    # print(f"1-alpha_bar_sqrt={one_minus_alphas_bar_sqrt}")
    # alphas_bar_sqrt = torch.sqrt(alphas_prod)
    

    # # Generate expert trajectories
    # shape = (1, input_dim)  # Shape of the input (batch_size, input_dim)
    # # output_dir = '../gen_datasets'
    
    # # if args.reconstruct:
    # trajectories = []
    # # obs_mean = obs.mean(0)
    # # obs_std = obs.std(0)
    # # print(f"obs std: {obs_std}")
    # # obs = norm_vec(obs, obs_mean, obs_std)
    # # # if args.norm:
    # # actions_mean = actions.mean(0)
    # # actions_std = actions.std(0)
    # # print(f"actions std: {actions_std}")
    # # actions = norm_vec(actions, actions_mean, actions_std)
    
    # # expert_data = torch.cat((obs, actions), 1)
    # sample_num = expert_data.size()[0]
    # torch.manual_seed(500)
    
    # with torch.no_grad():
    #     out = reconstruct(
    #         model,
    #         expert_data.squeeze().to(device),
    #         alphas_bar_sqrt,
    #         one_minus_alphas_bar_sqrt,
    #         num_steps-1,
    #         betas[:num_steps],
    #         recon=args.reconstruct
    #     )
        
    #     mse = MSELoss()     
    #     loss = mse(out.cpu().detach(), expert_data)
        
    #     print(loss.item())
            
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




