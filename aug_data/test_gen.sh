reconstruct="True"
ddpm_depth=4

# exexpert_data="../expert_datasets"
# ddpm_model_dir="../data/dm/trained_models/"
# output_dir="../gen_datasets"

env_list=("pick" "hand" "halfcheetah" "walker" "ant" "maze")
# env_list=("pick")


for env_name in "${env_list[@]}"; do
# env_name=$1
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
    python3 test_gen.py --reconstruct $reconstruct --ddpm_depth $ddpm_depth --env_name $env_name --hidden_dim $hidden_dim 
done

# python3 test_gen.py --reconstruct $reconstruct --ddpm_depth $ddpm_depth --env_name $env_name --hidden_dim $hidden_dim 
