import torch
import numpy as np
import argparse

def load_data(file_path):
    return torch.load(file_path)

def save_data(data, file_path):
    torch.save(data, file_path)

def sample_data(data, fraction, seed=None):
    np.random.seed(seed)
    torch.manual_seed(seed)

    sampled_data = {}
    for key, value in data.items():
        num_samples = int(len(value) * fraction)
        indices = np.random.choice(len(value), num_samples, replace=False)
        sampled_data[key] = value[indices]
    return sampled_data


def compute_split(data, traj_frac, rnd_seed=31):
    """
    Split a dictionary-like dataset into a subset based on a fraction.

    Args:
        data (dict): A dictionary where values are array-like structures.
        traj_frac (float): Fraction of data to include in the subset (0 < traj_frac ≤ 1).
        rnd_seed (int): Random seed for reproducibility.

    Returns:
        dict: A dictionary containing the subset of data for each key.
    """
    # Determine the number of samples to use
    total_length = len(next(iter(data.values())))  # Length of the first array
    # print('total len: ', total_length)
    use_count = int(total_length * traj_frac)
    # print('use count: ', use_count)

    # Generate shuffled indices
    all_idxs = np.arange(total_length)
    rng = np.random.default_rng(rnd_seed)
    rng.shuffle(all_idxs)
    selected_idxs = all_idxs[:use_count]

    # Create subset dictionary
    subset_data = {key: value[selected_idxs] for key, value in data.items()}
    return subset_data

def combine_datasets(data1, data2, frac1, frac2):
    # data1 = sample_data(data1, frac1)
    # data2 = sample_data(data2, frac2)
    data1 = compute_split(data1, frac1)
    data2 = compute_split(data2, frac2)
    # print(data1)
    combined_data = {}
    for key in data1.keys():
        combined_data[key] = torch.cat((data1[key], data2[key]), dim=0)
    return combined_data

def main():
    parser = argparse.ArgumentParser(description='Combine datasets')
    parser.add_argument('--input1', type=str, default='expert_datasets/maze.pt', help='Path to the first input dataset')
    parser.add_argument('--input2', type=str, default='gen_datasets/maze/step1000seed0_noise.pt', help='Path to the second input dataset')
    parser.add_argument('--output', type=str, default='expert_datasets/mazes/maze25_3.pt',  help='Path to save the combined dataset')
    parser.add_argument('--frac1', type=float, default=1.0, help='Fraction of data to sample from dataset1')
    parser.add_argument('--frac2', type=float, default=1.0, help='Fraction of data to sample from dataset2')
    
    args = parser.parse_args()

    # Load the datasets
    data1 = load_data(args.input1)
    data2 = load_data(args.input2)

    # Combine the datasets with the specified proportion and sample fraction
    combined_data = combine_datasets(data1, data2, args.frac1, args.frac2)
    # total_length = len(next(iter(combined_data.values())))  # Length of the first array
    # print('total cb len: ', total_length)

    # Save the combined dataset
    save_data(combined_data, args.output)

if __name__ == "__main__":
    main()