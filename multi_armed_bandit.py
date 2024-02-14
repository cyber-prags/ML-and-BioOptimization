
import numpy as np

# Set the seed for reproducibility
np.random.seed(0)

# Parameters
n_arms = 2
n_plays = 10
epsilon = 0.1
reward_probabilities = [0.5, 0.6]  # Probabilities of getting a reward from each arm
q_values = np.zeros(n_arms)  # Estimated values of each arm
n_selected = np.zeros(n_arms)  # Number of times each arm was selected
total_reward = 0
rewards = []
optimal_action_count = []

# Run the bandit algorithm
for play in range(n_plays):
    # Exploration vs exploitation
    if np.random.rand() < epsilon:
        # Explore: select a random arm
        chosen_arm = np.random.choice(n_arms)
    else:
        # Exploit: select the arm with the highest estimated value
        chosen_arm = np.argmax(q_values)
    
    # Simulate pulling the arm
    reward = np.random.rand() < reward_probabilities[chosen_arm]
    rewards.append(reward)
    total_reward += reward
    n_selected[chosen_arm] += 1
    
    # Update the estimated value (Q-value) for the chosen arm
    q_values[chosen_arm] += (reward - q_values[chosen_arm]) / n_selected[chosen_arm]

    # Check if the optimal arm is chosen
    optimal_arm = np.argmax(reward_probabilities)
    is_optimal = chosen_arm == optimal_arm
    optimal_action_count.append(is_optimal)

# Calculate percentage of optimal actions
optimal_action_percentage = np.mean(optimal_action_count) * 100

# Results
q_values, total_reward, optimal_action_percentage, rewards

if __name__ == '__main__':
    print("Q-values for the arms:", q_values)
    print("Total reward:", total_reward)
    print("Percentage of optimal action:", optimal_action_percentage)
    print("Rewards:", rewards)
