import torch
import numpy as np
import argparse

def load_data(file_path):
    return torch.load(file_path)

def save_data(data, file_path):
    torch.save(data, file_path)

def sample_data(data, fraction):
    sampled_data = {}
    for key, value in data.items():
        num_samples = int(len(value) * fraction)
        indices = np.random.choice(len(value), num_samples, replace=False)
        sampled_data[key] = value[indices]
    return sampled_data

def combine_datasets(data1, data2, frac1, frac2):
    data1 = sample_data(data1, frac1)
    data2 = sample_data(data2, frac2)
    
    combined_data = {}
    for key in data1.keys():
        # num_samples_data1 = int(proportion * data1[key].shape[0])
        # num_samples_data2 = int((1 - proportion) * data2[key].shape[0])
        
        combined_data[key] = torch.cat((data1[key], data2[key]), dim=0)
    return combined_data

def main():
    parser = argparse.ArgumentParser(description='Combine datasets')
    parser.add_argument('--input1', type=str, default='expert_datasets/maze.pt', help='Path to the first input dataset')
    parser.add_argument('--input2', type=str, default='gen_datasets/maze/step1000seed0_noise.pt', help='Path to the second input dataset')
    parser.add_argument('--output', type=str, default='gen_datasets/cb/w_z/maze.pt',  help='Path to save the combined dataset')
    # parser.add_argument('--proportion', type=float, default=0.5, help='Proportion of data to take from the first dataset')
    parser.add_argument('--frac1', type=float, default=1.0, help='Fraction of data to sample from dataset1')
    parser.add_argument('--frac2', type=float, default=1.0, help='Fraction of data to sample from dataset2')
    
    args = parser.parse_args()

    # Load the datasets
    data1 = load_data(args.input1)
    data2 = load_data(args.input2)

    # Combine the datasets with the specified proportion and sample fraction
    combined_data = combine_datasets(data1, data2, args.frac1, args.frac2)

    # Save the combined dataset
    save_data(combined_data, args.output)

if __name__ == "__main__":
    main()