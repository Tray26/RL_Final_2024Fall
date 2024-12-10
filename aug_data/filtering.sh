env_list=("pick" "hand" "halfcheetah" "walker" "ant" "maze")
gen_step_list=(10 100 1000)
proportions=(25 50 75)
filter_mode=$1

for env_name in "${env_list[@]}"; do
    for gen_step in "${gen_step_list[@]}"; do
        for proportion in "${proportions[@]}"; do
            echo "Running with env_name=$env_name, gen_steps=$gen_step, proportion=$proportion"
            python3 state_filter.py  --env_name $env_name --gen_steps $gen_step --proportion $proportion --filter_mode $filter_mode
        done
    done
done