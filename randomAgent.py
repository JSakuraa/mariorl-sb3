import gymnasium as gym
import retro

def main():
    # Create the Stable-retro environment
    env = retro.make(game="SuperMarioBros-Nes")
    # Reset the environment to a default state
    env.reset()

    # Run the environment with random actions
    while True:
        # Set action equal to a random item from the action_space
        action = env.action_space.sample()
        # Take a step in the environment
        observation, reward, terminated, truncated, info = env.step(action)
        # Render the environment
        env.render()
        # Resets the environment when done conditions are met
        if terminated or truncated:
            env.reset()
    env.close()


if __name__ == "__main__":
    main()