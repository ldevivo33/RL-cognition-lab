import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt
from dqn_agent import DQNAgent
import torch

def train_dqn(n_episodes=1000,
              batch_size=64,
              target_update_freq=1000,  # update target net every 1000 steps
              update_every=4):          # do a gradient step every 4 steps
    env = gym.make("CartPole-v1")
    watch_env = gym.make("CartPole-v1", render_mode="human")  # just for occasional watching

    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n
    agent = DQNAgent(state_dim, action_dim, lr=1e-4, epsilon_decay=0.995)

    rewards_per_episode = []
    steps = 0

    for ep in range(n_episodes):
        obs, _ = env.reset()
        state = obs
        terminated = truncated = False
        total_reward = 0

        while not (terminated or truncated):
            # Pick action
            action = agent.select_action(state)

            # Step
            next_obs, reward, terminated, truncated, _ = env.step(action)

            # Store transition
            agent.replay_buffer.push(state, action, reward, next_obs, terminated or truncated)

            # Learn every few steps
            if steps % update_every == 0:
                agent.update(batch_size=batch_size)

            # Update target net occasionally
            if steps % target_update_freq == 0:
                agent.update_target()

            state = next_obs
            total_reward += reward
            steps += 1

        # Decay epsilon after each episode
        agent.decay_epsilon()
        rewards_per_episode.append(total_reward)

        # Logging
        if ep % 50 == 0:
            avg = np.mean(rewards_per_episode[-50:])
            print(f"Episode {ep}, avg reward={avg:.1f}, epsilon={agent.epsilon:.3f}")

        # Smooth “watching” every 200 episodes
        if ep % 200 == 0 and ep > 0:
            obs, _ = watch_env.reset()
            done = False
            while not done:
                act = agent.select_action(obs)
                obs, _, terminated, truncated, _ = watch_env.step(act)
                done = terminated or truncated

        # Early stop if solved
        if len(rewards_per_episode) >= 100:
            avg_last_100 = np.mean(rewards_per_episode[-100:])
            if avg_last_100 >= 500:
                print(f"🎉 Solved CartPole with avg=500 at episode {ep}!")
                break

    env.close()
    watch_env.close()

    # Plot learning curve
    window = 20
    smoothed = [np.mean(rewards_per_episode[i:i+window]) 
                for i in range(len(rewards_per_episode)-window)]
    plt.plot(smoothed)
    plt.xlabel("Episode")
    plt.ylabel("Average Reward")
    plt.title("CartPole with DQN")
    plt.show()

    torch.save(agent.q_net.state_dict(), "dqn_cartpole.pth")
    print("Model saved to dqn_cartpole.pth")

def evaluate(agent, episodes=5):
    """Run the trained agent with epsilon=0 (greedy) and render."""
    eval_env = gym.make("CartPole-v1", render_mode="human")
    old_epsilon = agent.epsilon
    agent.epsilon = 0.0  # disable exploration

    for ep in range(episodes):
        obs, _ = eval_env.reset()
        done = False
        total_reward = 0
        while not done:
            action = agent.select_action(obs)
            obs, reward, terminated, truncated, _ = eval_env.step(action)
            total_reward += reward
            done = terminated or truncated
        print(f"Evaluation Episode {ep}, Reward={total_reward}")

    agent.epsilon = old_epsilon
    eval_env.close()


if __name__ == "__main__":
    agent = train_dqn(n_episodes=2500)

    # CartPole has 4 states, 2 actions
    agent = DQNAgent(state_dim=4, action_dim=2)
    agent.q_net.load_state_dict(torch.load("dqn_cartpole.pth"))

    evaluate(agent, episodes=5)
