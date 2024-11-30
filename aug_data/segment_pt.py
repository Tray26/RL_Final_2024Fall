import torch
import numpy as np
import os

def load_data(file_path):
    return torch.load(file_path)

def save_data(data, file_path):
    torch.save(data, file_path)

def partition_data(data, fraction=0.25):
    partitioned_data = {}
    flag = True
    for key, value in data.items():
        if flag:
            num_samples = int(len(value) * fraction)
            indices = np.random.choice(len(value), num_samples, replace=False)
            flag = False
        partitioned_data[key] = value[indices]
    return partitioned_data

def main(env_name, data_dir, partition):
    # Load the datasets
    data = load_data(os.path.join(data_dir, f"{env_name}.pt"))
    # print(data2['trajectories']['obs'])

    # Cut the datasets
    print(partition/100)
    cut_data = partition_data(data, fraction=partition/100)
    
    print(data['obs'].shape)
    print(cut_data['obs'].shape)
    # print(cut_data.shape)

    # Save the combined dataset
    # save_data(cut_data, '../expert_datasets/picks/pick75.pt')
    os.makedirs(os.path.join(data_dir, env_name), exist_ok=True)
    save_data(cut_data, os.path.join(data_dir, env_name, f"{env_name}{partition}.pt"))

    print('CUT')
    # print("Datasets cut and saved to 'expers_datasets/ants/ant25.pt'")

if __name__ == "__main__":
    env_name="hand_gen"
    data_dir='../expert_datasets'
    partition = 75

    main(env_name, data_dir, partition)