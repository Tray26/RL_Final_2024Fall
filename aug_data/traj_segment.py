import numpy as np
import torch
import torch.nn as nn

def load_data(file_path):
    return torch.load(file_path)

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
    one_ids = [index for index, value in enumerate(data['done']) if value == 1]
    use_count = int(len(one_ids) * traj_frac)
    one_sections = []
    one_sections.append((0, one_ids[0]))
    one_sections[1:len(one_ids)] = [(one_ids[i-1]+1, one_ids[i]) for i in range(1,len(one_ids))]
    rng = np.random.default_rng(rnd_seed)
    rng.shuffle(one_sections)
    selected_ones = one_sections[:use_count]
    selected_idxs = []
    for i, ids in enumerate(selected_ones):
        selected_idxs.extend(np.arange(ids[0], ids[1]+1))

    # Create subset dictionary
    subset_data = {key: value[selected_idxs] for key, value in data.items()}
    return subset_data

def main():
    traj = load_data('expert_datasets/pick.pt')
    # print('len done: ', len(traj['done']))
    data_cut = compute_split(traj, 0.5)
    torch.save(data_cut, 'gen_datasets/maze/traj/maze_traj_50.pt')

    # print(traj.keys())
    # print(len(traj['obs']))
    # print(len(data_cut['done']))


if __name__ == "__main__":
    main()

