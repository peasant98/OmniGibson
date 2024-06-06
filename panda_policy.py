"""
Example training code using stable-baselines3 PPO for one BEHAVIOR activity.
Note that due to the sparsity of the reward, this training code will not converge and achieve task success.
This only serves as a starting point that users can further build upon.
"""

import argparse
import os, time, cv2
import numpy as np
import yaml

import omnigibson as og
from omnigibson import example_config_path
from omnigibson.macros import gm
from omnigibson.robots.robot_base import BaseRobot
from omnigibson.utils.python_utils import meets_minimum_version
from gym import spaces

try:
    import gym
    import torch as th
    import torch.nn as nn
    import tensorboard
    from stable_baselines3 import PPO
    from stable_baselines3.common.evaluation import evaluate_policy
    from stable_baselines3.common.preprocessing import maybe_transpose
    from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
    from stable_baselines3.common.utils import set_random_seed
    from stable_baselines3.common.callbacks import CallbackList, CheckpointCallback, EvalCallback

except ModuleNotFoundError:
    og.log.error("torch, stable-baselines3, or tensorboard is not installed. "
                 "See which packages are missing, and then run the following for any missing packages:\n"
                 "pip install stable-baselines3[extra]\n"
                 "pip install tensorboard\n"
                 "pip install shimmy>=0.2.1\n"
                 "Also, please update gym to >=0.26.1 after installing sb3: pip install gym>=0.26.1")
    exit(1)

assert meets_minimum_version(gym.__version__, "0.26.1"), "Please install/update gym to version >= 0.26.1"

# We don't need object states nor transitions rules, so we disable them now, and also enable flatcache for maximum speed
gm.ENABLE_OBJECT_STATES = False
gm.ENABLE_TRANSITION_RULES = False
gm.ENABLE_FLATCACHE = True


class PandaEnv(og.Environment):
    def __init__(self, configs):
        super().__init__(configs)
        self.robot = self.robots[0]
        self.ee_pos = self.robot.get_eef_position()
        
        # target is an offset from the current position
        # for now, task is to move the EE to the target position
        self.target_position = self.ee_pos + np.array([0.1, 0.1, 0.1])
        
    def step(self, action):
        obs, reward, done, info = super().step(action)
        
        # get robot EE position
        self.ee_pos = self.robot.get_eef_position()
        
        depth = np.random.random(2)
        language_embedding = np.random.random(2)
        extended_obs = np.concatenate((depth, language_embedding))
        custom_reward = self.custom_reward(obs, reward, done)

        return obs, custom_reward, done, info
        
    def reset(self):
        return super().reset()
    
    def custom_reward(self, obs, reward, done):
        dist = np.linalg.norm(self.ee_pos - self.target_position)
        reward = -dist
        
        if dist < 0.01:
            reward = 10
            done = True
        return reward
    

def main():
    # Parse args
    parser = argparse.ArgumentParser(description="Train or evaluate a PPO agent in BEHAVIOR for the Franka Panda")

    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="Absolute path to desired PPO checkpoint to load for evaluation",
    )

    parser.add_argument(
        "--eval",
        action="store_true",
        help="If set, will evaluate the PPO agent found from --checkpoint",
    )

    args = parser.parse_args()
    
    # create tensorboard log dir
    tensorboard_log_dir = os.path.join("log_dir", time.strftime("%Y%m%d-%H%M%S"))
    os.makedirs(tensorboard_log_dir, exist_ok=True)
    prefix = ''
    seed = 0
    
    print(f"example_config_path: {example_config_path}")

    # Load config
    with open(f"{example_config_path}/panda_behavior.yaml", "r") as f:
        cfg = yaml.load(f, Loader=yaml.FullLoader)


    # Make sure flattened obs and action space is used
    cfg["env"]["flatten_action_space"] = True
    cfg["env"]["flatten_obs_space"] = True

    # Only use RGB obs
    cfg["robots"][0]["obs_modalities"] = ["rgb"]

    if not args.eval:
        cfg["task"]["visualize_goal"] = False

    env = PandaEnv(configs=cfg)

    # If we're evaluating, hide the ceilings and enable camera teleoperation so the user can easily
    # visualize the rollouts dynamically
    if args.eval:
        ceiling = env.scene.object_registry("name", "ceilings")
        ceiling.visible = False
        og.sim.enable_viewer_camera_teleoperation()

    # Set the seed
    set_random_seed(seed)
    env.reset()
    
    # wait a few seconds
    time.sleep(3)
    
    env.robots[0].reset()
    
    env_robot: BaseRobot = env.robots[0]
    
    # sensor name is first key
    sensor_name = list(env_robot.sensors.keys())[0]
    
    instrinsic_matrix = env_robot.sensors[sensor_name].intrinsic_matrix
    
    os.makedirs(tensorboard_log_dir, exist_ok=True)

    if args.eval:
        pass
    else:
        step = 0
        max_steps = -1
        ee_start_pos = env_robot.get_eef_position()
        target_pos = ee_start_pos + np.array([0.2, 0.2, 0.2])
        action = np.zeros(env_robot.action_dim)
        while step != max_steps:
            obs, reward, done, info = env.step(action=action)
            print("reward: ", reward)
            print("obs: ", obs)
            # get loss from diff in EE position and target position
            ee_pos = env_robot.get_eef_position()
            loss = target_pos - ee_pos
            
            # pid with kp = 1, ki = 0, kd = 0
            # action = loss
            print("loss: ", loss)
            # first 3 values of actions are the EE position
            kp = 0.5
            action[:3] = (kp * loss)
            print(np.linalg.norm(loss))
            if np.linalg.norm(loss) < 0.03:
                print('task complete!')
                break
            print(action)
            step += 1

        # Always shut down the environment cleanly at the end
        env.close()
        
    

if __name__ == "__main__":
    main()
