import torch
import pandas as pd

class PtLoader:
    def __init__(self, model_path):
        self.model_path = model_path
        self.model = None

    def load_model(self):
        self.model = torch.load(self.model_path)

    def get_param_range(self):
        if self.model is None:
            raise ValueError("Model is not loaded. Call load_model() first.")
        
        min_val = float('inf')
        max_val = float('-inf')

        for param in self.model.parameters():
            min_val = min(min_val, param.min().item())
            max_val = max(max_val, param.max().item())

        return min_val, max_val

def norm_vec(x, mean, std):
    obs_x = torch.clamp(
        (x - mean) / (std + 1e-8),
        -10.0,
        10.0,
    )
    return obs_x

if __name__ == "__main__":


    model_path = 'expert_datasets/halfcheetah.pt'
    loader = PtLoader(model_path)
    loader.load_model()




    min_val = float('inf')
    max_val = float('-inf')

    results = []

    model_files = [ 'hand.pt', 'maze.pt', 'pick.pt', 'walker.pt', 'ant.pt', 'halfcheetah.pt']
    for model_file in model_files:
        loader = PtLoader(f'expert_datasets/{model_file}')
        loader.load_model()

        for key, tensor in loader.model.items():
            if key == 'obs':
                # obs = tensor
                obs_mean = tensor.mean(0)
                obs_std = tensor.std(0)
                print(f"obs std: {obs_std}")
                tensor = norm_vec(tensor, obs_mean, obs_std)

            elif key == 'actions':
                # actions = tensor
                actions_mean = tensor.mean(0)
                actions_std = tensor.std(0)
                print(f"actions std: {actions_std}")
                tensor = norm_vec(tensor, actions_mean, actions_std)
              

            min_val = tensor.min().item()
            max_val = tensor.max().item()
            print(f'Key: {key}, Minimum value: {min_val}, Maximum value: {max_val}')
            results.append({'Dataset': model_file, 'Key': key, 'Min Value': min_val, 'Max Value': max_val})

        df = pd.DataFrame(results)
        print(df)
        df.to_csv('Normed_Dataset_ranges.csv', index=False)