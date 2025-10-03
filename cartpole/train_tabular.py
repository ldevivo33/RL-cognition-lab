# Run `pip install "gymnasium[classic-control]" matplotlib` first
import gymnasium as gym
import matplotlib.pyplot as plt
from tabular_agent import CartpoleAgent, discretize

# Two environments: one for fast training, one for watching
train_env = gym.make("CartPole-v1")  # no rendering (fast)
watch_env = gym.make("CartPole-v1", render_mode="human")  # for occasional render

rewards_per_episode = []

# Initialize agent
agent = CartpoleAgent(train_env, learning_rate=0.05,
                       initial_epsilon=1.0,
                       epsilon_decay=0.0001,
                       final_epsilon=0.01,
                       discount_factor=0.99)

n_episodes = 15000
for ep in range(n_episodes):
    obs, info = train_env.reset()
    state = discretize(obs)
    terminated = truncated = False
    total_reward = 0
    
    while not (terminated or truncated):
        action = agent.get_action(state)
        obs, reward, terminated, truncated, info = train_env.step(action)
        next_state = discretize(obs)
        agent.update(state, action, reward, terminated, next_state)
        state = next_state
        total_reward += reward

    agent.decay_epsilon()
    rewards_per_episode.append(total_reward)

    # Print every 100 episodes
    if ep % 100 == 0:
        avg = sum(rewards_per_episode[-100:]) / len(rewards_per_episode[-100:])
        print(f"Episode {ep}, avg reward (last 100): {avg:.1f}, epsilon={agent.epsilon:.3f}")

    # Render every 500 episodes
    if ep % 1000 == 0:
        print(f"🎥 Watching episode {ep}")
        watch_obs, _ = watch_env.reset()
        watch_state = discretize(watch_obs)
        done = False
        while not done:
            watch_action = agent.get_action(watch_state)
            watch_obs, reward, terminated, truncated, _ = watch_env.step(watch_action)
            watch_state = discretize(watch_obs)
            done = terminated or truncated

train_env.close()
watch_env.close()

# Plot learning curve
window = 50
smoothed = [sum(rewards_per_episode[i:i+window])/window 
            for i in range(len(rewards_per_episode)-window)]
plt.plot(smoothed)
plt.xlabel("Episode")
plt.ylabel("Average Reward (window=50)")
plt.title("CartPole Q-learning Performance")
plt.show()
