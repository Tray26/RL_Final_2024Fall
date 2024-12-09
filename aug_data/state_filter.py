import torch
from torch.nn import MSELoss
import argparse
import os

from utils import str2bool, norm_vec

def calculate_mse(generation_data, expert_data):
    # generation_data 和 expert_data 都是 (N, D) 張量，N 是數據筆數，D 是每筆數據的維度
    # print(generation_data.shape, expert_data.shape)
    mse_values = torch.cdist(generation_data, expert_data, p=2) ** 2
    # print(mse_values.shape)
    # mse_values = mse_values.mean(dim=1)  # 每一筆 generation_data 和所有 expert_data 的 MSE 計算出來
    return mse_values

def main(args):
    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    env_name = args.env_name
    expert_data_dir = args.expert_data_dir
    gen_data_dir = args.gen_data_dir
    filter_data_dir = args.filter_data_dir
    gen_steps = args.gen_steps
    proportion = args.proportion
    
    expert_path = os.path.join(expert_data_dir, f"{env_name}.pt")
    gen_path = os.path.join(gen_data_dir, env_name, 'data', f"step{gen_steps}seed0_denorm.pt")
    filtered_path = os.path.join(filter_data_dir, env_name, f"step{gen_steps}_{proportion}.pt")
    
    os.makedirs(os.path.join(filter_data_dir, env_name), exist_ok=True)
    expert_data = torch.load(expert_path)
    expert_obs = expert_data["obs"]
    
    gen_data = torch.load(gen_path)
    gen_obs = gen_data["obs"]
    gen_act = gen_data["actions"]
    gen_done = gen_data["done"]
    gen_next_obs=gen_data["next_obs"]
    
    norm_exp_obs = norm_vec(expert_obs, expert_obs.mean(0), expert_obs.std(0))
    norm_gen_obs = norm_vec(gen_obs, gen_obs.mean(0), gen_obs.std(0))
    
    mse_values = calculate_mse(norm_gen_obs, norm_exp_obs)
    # print(mse_values.shape)
    min_mse_values, _ = mse_values.min(dim=1)
    # print(min_mse_values.shape)
    
    threshold = torch.quantile(min_mse_values, (100 - proportion)/100)  # 計算 80 百分位數，選出最大的 20%
    top_indices = (min_mse_values <= threshold).nonzero(as_tuple=True)[0]
    
    top_obs = gen_obs[top_indices]
    top_act = gen_act[top_indices]
    top_done = gen_done[top_indices]
    top_next = gen_next_obs[top_indices]
    
    filtered_data = {
        "obs": top_obs,
        "actions": top_act,
        "done": top_done, 
        "next_obs": top_next
    }
    torch.save(filtered_data, filtered_path)
    
    
    # print(top_obs.shape)
    # top_20_percent_generation_data = generation_data[top_20_percent_indices]

    
    # expert_actions = expert_data["actions"]
    # expert_done = expert_data["done"]
    # expert_next_obs=expert_data["next_obs"]
    
    
    

if __name__=="__main__":
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--env_name', type=str, default='pick')
    parser.add_argument('--expert_data_dir', type=str, default="../expert_datasets")
    parser.add_argument('--gen_data_dir', type=str, default='../gen_datasets_denorm')
    parser.add_argument('--filter_data_dir', type=str, default='../gen_datasets_filter_near')
    parser.add_argument('--gen_steps', type=int, default=100)
    parser.add_argument('--proportion', type=int, default=20)

    args = parser.parse_args()
    
    main(args)

