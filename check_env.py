import gym
import d4rl
from gym.envs.registration import register

# Register custom environments

from gym.envs.registration import register

register(
    id='AntGoal-v0',
    entry_point='dbc.envs.ant:AntGoalEnv',
    max_episode_steps=50,
    )

register(
    id='FetchPick-v1',
    entry_point='dbc.envs.ant :FetchPickEnv',  # Replace with the actual entry point
)

register(
    id='HandManipulateBlockRotateZ-v0',
    entry_point='path.to.your.module:HandManipulateBlockRotateZEnv',  # Replace with the actual entry point
)

# Load Fetch Pick task
fetch_env = gym.make('FetchPick-v1')
print("Fetch Pick task action space range:", fetch_env.action_space)

# Load Hand Rotate task
hand_env = gym.make('HandManipulateBlockRotateZ-v0')
print("Hand Rotate task action space range:", hand_env.action_space)