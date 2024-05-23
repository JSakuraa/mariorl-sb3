import gymnasium as gym
import retro
from stable_baselines3 import PPO
import numpy as np
from stable_baselines3.common.atari_wrappers import MaxAndSkipEnv

model = PPO.load("tmp/best_model.zip")

# Run the model using the trained network
def main():
    env = retro.make(game="SuperMarioBros-Nes")
    env = MaxAndSkipEnv(env, 4)

    obs, info = env.reset()

    while True:
        action, state = model.predict(obs)
        obs, reward, terminated, truncated, info = env.step(action)
        env.render()
        if terminated or truncated:
            env.reset()
    env.close()


if __name__ == "__main__":
    main()