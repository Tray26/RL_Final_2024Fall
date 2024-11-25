import torch
import numpy as np

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

def main():
    # Load the datasets
    data = load_data('expert_datasets/pick.pt')
    # print(data2['trajectories']['obs'])

    # Cut the datasets
    cut_data = partition_data(data, fraction=0.75)

    # Save the combined dataset
    save_data(cut_data, 'expert_datasets/picks/pick75.pt')

    print('CUT')
    # print("Datasets cut and saved to 'expers_datasets/ants/ant25.pt'")

if __name__ == "__main__":
    main()