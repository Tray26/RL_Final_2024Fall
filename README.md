# 2024 Fall Reinforcement Learning Final Project
## Title: Behavior Cloning with Diffusion Data Augmentation

This code is based on the paper <a href="https://arxiv.org/abs/2302.13335" title="Diffusion Model-Augmented Behavioral Cloning">Diffusion Model-Augmented Behavioral Cloning </a>(ICML2024).

### Installation
```
conda create -n [your_env_name] python=3.7.2
conda activate [your_env_name]
pip install -r requirements.txt

cd d4rl
pip install -e .
cd ../rl-toolkit
pip install -e .

cd ..
mkdir -p data/trained_models
```
You may also need to install mujoco. Please check the website: <a href="https://blog.csdn.net/qq_47997583/article/details/125400418?ops_request_misc=%257B%2522request%255Fid%2522%253A%2522172447674316800188543607%2522%252C%2522scm%2522%253A%252220140713.130102334..%2522%257D&request_id=172447674316800188543607&biz_id=0&utm_medium=distribute.pc_search_result.none-task-blog-2~all~top_positive~default-1-125400418-null-null.142%5Ev100%5Epc_search_result_base7&utm_term=ubuntu20.04%E5%AE%89%E8%A3%85mujoco&spm=1018.2226.3001.4187" title="Diffusion Model-Augmented Behavioral Cloning">Install mujoco under Ubuntu20.04</a>.

### Train
* For diffusion model pretraining, run `python dbc/ddpm.py`
* For behavior cloning (BC), run 
    * Maze: `bash ./wandb.sh ./configs/maze/bc.yaml`
    * Fetch Pick: `bash ./wandb.sh./configs/fetchPick/bc.yaml`
    * Walker: `bash ./wandb.sh ./configs/walker/bc.yaml`
* For diffusion policy (DP), run 
    * Maze: `bash ./wandb.sh ./configs/maze/dp.yaml`
    * Fetch Pick: `bash ./wandb.sh./configs/fetchPick/dp.yaml`
    * Walker: `bash ./wandb.sh ./configs/walker/dp.yaml`

You can modify the arguments in **configs** folder for BC and DP, including dataset file, checkpoints, evaluation, etc. settings. For instance, 
* To train the policy, you may need to set **traj-load-path** to correspoding path. 
    * 25%, 50%, 75%, 100% expert data are all in `expert_datasets` directory.
    * Generated data with different steps are in `gen_datasets_denorm` directory.
    * Generated data with filter are in `gen_datasets_filter_far_mean` (for least similar to expert)and `gen_datasets_filter_near_mean`(for most similar to expert) directories. 
* To evaluate, you may need to:
    * Set **eval-only** to True.
    * Set **seed** to [1,2,3,...] to evaluate multiple times.
    * Set **load-file** to the path of the desired model checkpoint.
    

### Data generation
* For generation based on timesteps, run `bash aug_data/generate.sh`
* For generation based on filter, run `bash aug_data/filtering.sh`

### Data augmentation
Run `python aug_data/combine_pts.py --input1 [file_name1 you want to combine] --input2 [file_name2] --output [file_name_out]`

### Checkpoints
*    Checkpoints of diffusion model for data generation are in `data/dm/trained_models`.
*    Checkpoints of DP training results are also provided in `data/dm/trained_models`. 
