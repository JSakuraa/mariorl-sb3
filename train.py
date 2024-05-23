import numpy as np
import torch

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import SubprocVecEnv, VecMonitor
from stable_baselines3.common.results_plotter import load_results, ts2xy
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.atari_wrappers import MaxAndSkipEnv

import os
import retro # type: ignore

class SaveOnBestTrainingRewardCallback(BaseCallback):
    """
    Callback for saving a model (the check is done every ``check_freq`` steps)
    based on the training reward (in practice, we recommend using ``EvalCallback``).

    :param check_freq:
    :param log_dir: Path to the folder where the model will be saved.
      It must contains the file created by the ``Monitor`` wrapper.
    :param verbose: Verbosity level.
    """
    def __init__(self, check_freq: int, log_dir: str, verbose: int = 1):
        super(SaveOnBestTrainingRewardCallback, self).__init__(verbose)
        self.check_freq = check_freq
        self.log_dir = log_dir
        self.save_path = os.path.join(log_dir, 'best_model')
        self.best_mean_reward = -np.inf

    def _init_callback(self) -> None:
        # Create folder if needed
        if self.save_path is not None:
            os.makedirs(self.save_path, exist_ok=True)

    def _on_step(self) -> bool:
        if self.n_calls % self.check_freq == 0:

          # Retrieve training reward
          x, y = ts2xy(load_results(self.log_dir), 'timesteps')
          if len(x) > 0:
              # Mean training reward over the last 100 episodes
              mean_reward = np.mean(y[-100:])
              if self.verbose > 0:
                print(f"Num timesteps: {self.num_timesteps}")
                print(f"Best mean reward: {self.best_mean_reward:.2f} - Last mean reward per episode: {mean_reward:.2f}")

              # New best model, you could save the agent here
              if mean_reward > self.best_mean_reward:
                  self.best_mean_reward = mean_reward
                  # Example for saving best model
                  if self.verbose > 0:
                    print(f"Saving new best model to {self.save_path}")
                  self.model.save(self.save_path)

        return True

def make_env(env_id, rank, seed=0):
    """
    Makes retro environment and applies the MaxAndSkipEnv

    :param check_freq:
    :param log_dir: Path to the folder where the model will be saved.
      It must contains the file created by the ``Monitor`` wrapper.
    :param verbose: Verbosity level.
    """
    def _init():
        env = retro.make(game=env_id)
        env = MaxAndSkipEnv(env, 4)
        return env
    
    return _init

if __name__ == '__main__':
    # Create a folder to store the best model
    log_dir = "tmp/"
    os.makedirs(log_dir, exist_ok=True)

    # Set the env id to Super Mario Bros
    env_id = "SuperMarioBros-Nes"
    # The amount of environments to train at once (Depends on PC)
    num_cpu = 2

    # Create a monitor for the multiprocessed environments
    env = VecMonitor(SubprocVecEnv([make_env(env_id, i) for i in range(num_cpu)]), "tmp/monitor")

    # Use PPO model from Stable baselines3
    model = PPO('CnnPolicy', env, verbose=1, tensorboard_log="./board/", learning_rate=0.00003)
    # Use this to train from a pre-existing model
    model = PPO.load("/home/justin/marioml/tmp/best_model.zip",env=env)

    # Train model
    print("------------------Starting Learning---------------------")
    # Create callback
    callback = SaveOnBestTrainingRewardCallback(check_freq=1000, log_dir=log_dir)
    # Set learning parameter
    model.learn(total_timesteps=2000000, callback=callback, tb_log_name="PPO_00003")
    # Save model (likely not the best model)
    model.save(env_id)
    print("------------------Done Learning---------------------")