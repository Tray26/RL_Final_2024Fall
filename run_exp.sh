#!/bin/bash

# List of `traj-load-path` values
TRAJ_PATHS=(
    "gen_datasets_noise/step1000seed0_noise.pt"
    "gen_datasets_noise/step1000seed50_noise.pt"
    "gen_datasets_noise/step1000seed100_noise.pt"
    "gen_datasets_noise/step1000seed150_noise.pt"
    "gen_datasets_noise/step1000seed200_noise.pt"
    "gen_datasets_noise/step1000seed250_noise.pt"
    "gen_datasets_noise/step1000seed300_noise.pt"
    "gen_datasets_noise/step1000seed350_noise.pt"
    "gen_datasets_noise/step1000seed400_noise.pt"
    "gen_datasets_noise/step1000seed450_noise.pt"
    "gen_datasets_noise/step1000seed500_noise.pt"
)

# YAML file to run
CONFIG_FILE="./configs/walker/bc.yaml"

for TRAJ_PATH in "${TRAJ_PATHS[@]}"; do
    PREFIX=$(basename "$TRAJ_PATH" .pt)

    echo "Running experiment with traj-load-path=$TRAJ_PATH and prefix=$PREFIX"
    
    # Update YAML file dynamically
    UPDATED_CONFIG="${CONFIG_FILE%.yaml}_temp.yaml"
    sed "s|traj-load-path:.*|traj-load-path:\n    value: $TRAJ_PATH|g" "$CONFIG_FILE" > "$UPDATED_CONFIG"
    sed -i "s|prefix:.*|prefix:\n    value: $PREFIX|g" "$UPDATED_CONFIG"
    
    # Debug: Print the updated YAML
    echo "Updated YAML:"
    cat "$UPDATED_CONFIG"
    
    # Run the experiment
    bash ./wandb.sh "$UPDATED_CONFIG"
    
    # Clean up
    rm "$UPDATED_CONFIG"
done