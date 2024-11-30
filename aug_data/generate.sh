expert_dataset="expert_datasets/maze.pt"
reconstruct="False"
ddpm_depth=4
expert_data="../expert_datasets"
ddpm_model_dir="../data/dm/trained_models/"
num_step=$2

echo $ddpm_model_dir

env_name=$1
if [ "$env_name" == "pick" ]; then
    hidden_dim=1024
elif [ "$env_name" == "hand" ]; then
    hidden_dim=2048
elif [ "$env_name" == "halfcheetah" ]; then
    hidden_dim=1024
elif [ "$env_name" == "walker" ]; then
    hidden_dim=1024
elif [ "$env_name" == "ant" ]; then
    hidden_dim=1024
elif [ "$env_name" == "maze" ]; then
    hidden_dim=128
else
    echo "Unknown environment: $env_name"
    exit 1
fi

python3 data_gen.py --reconstruct $reconstruct --num_step $num_step --ddpm_depth $ddpm_depth --env_name $env_name --hidden_dim $hidden_dim --expert_data $expert_data --ddpm_model_dir $ddpm_model_dir
