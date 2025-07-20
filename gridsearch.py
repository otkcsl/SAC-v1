import numpy as np
import gymnasium as gym
import matplotlib.pyplot as plt
import moviepy.editor as mpy
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from collections import deque
import random
import os
import csv
import pandas as pd
import time

task_name = 'Reacher-v4'
buffer_alpha = 0.6
gamma = 0.99
tau = 0.005
buffer_capacity=1000000
batch_size=256
lr = 0.0003
max_steps = 500000
test_interval = 1000
episodes_per_test = 5
use_per = False
    
def hard_update(target, source):
    target.load_state_dict(source.state_dict())

def grad_false(network):
    for param in network.parameters():
        param.requires_grad = False

def save_to_csv(data, filename, headers):
    with open(filename, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(headers)
        writer.writerows(data)
        
def average_over_intervals(data, interval_size):
    return [np.mean(data[i:i+interval_size]) for i in range(0, len(data), interval_size)]

def compare_network_parameters(net1, net2):
    for param1, param2 in zip(net1.parameters(), net2.parameters()):
        if not torch.equal(param1, param2):
            print("The parameters are different!")
            return False
    print("The parameters are identical.")
    return True

class ContinuousQNetwork(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(ContinuousQNetwork, self).__init__()
        self.fc1 = nn.Linear(input_dim, 256)
        self.fc2 = nn.Linear(256, 256)
        self.fc3 = nn.Linear(256, output_dim)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

class ContinuousPolicyNetwork(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(ContinuousPolicyNetwork, self).__init__()
        self.fc1 = nn.Linear(input_dim, 256)
        self.fc2 = nn.Linear(256, 256)
        self.mean = nn.Linear(256, output_dim)
        self.log_std = nn.Linear(256, output_dim)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        mean = self.mean(x)
        log_std = self.log_std(x).clamp(-20, 2)
        std = torch.exp(log_std)
        return mean, std

    def sample(self, state):
        mean, std = self.forward(state)
        normal = torch.distributions.Normal(mean, std)
        z = normal.rsample()
        action = torch.tanh(z)
        log_prob = normal.log_prob(z) - torch.log(1 - action.pow(2) + 1e-6)
        log_prob = log_prob.sum(-1, keepdim=True)
        return action, log_prob

class PrioritizedReplayBuffer:
    def __init__(self, capacity, alpha):
        self.capacity = capacity
        self.buffer = deque(maxlen=capacity)
        self.priorities = np.zeros((capacity,), dtype=np.float32) 
        self.position = 0
        self.alpha = alpha

    def add(self, state, action, reward, next_state, done):
        max_priority = self.priorities.max() if len(self.buffer) > 0 else 1.0
        self.buffer.append((state, action, reward, next_state, done))
        self.priorities[self.position] = max_priority 
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size, beta=0.4):
        if len(self.buffer) == self.capacity:
            priorities = self.priorities
        else:
            priorities = self.priorities[:self.position]

        probs = priorities ** self.alpha
        probs /= probs.sum()

        indices = np.random.choice(len(self.buffer), batch_size, p=probs)
        batch = [self.buffer[idx] for idx in indices]

        total = len(self.buffer)
        weights = (total * probs[indices]) ** (-beta)
        weights /= weights.max()  

        states, actions, rewards, next_states, dones = zip(*batch)
        return (states, actions, rewards, next_states, dones, indices, weights)

    def update_priorities(self, batch_indices, batch_priorities):
        for idx, priority in zip(batch_indices, batch_priorities):
            self.priorities[idx] = priority
            
    def __len__(self):
        return len(self.buffer)

class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)

    def add(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        return states, actions, rewards, next_states, dones

    def __len__(self):
        return len(self.buffer)

class ContinuousSACAgent:
    def __init__(self, state_space, action_space, action_scale, alpha, gamma, tau, buffer_capacity, 
                 batch_size, lr, use_per):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.state_space = state_space
        self.action_space = action_space
        self.action_scale = action_scale
        self.alpha = alpha
        self.lr = lr
        self.gamma = gamma
        self.tau = tau
        self.batch_size = batch_size
        self.q1_losses = []
        self.q2_losses = []
        self.policy_losses = []
        
        self.q_network1 = ContinuousQNetwork(state_space.shape[0] + action_space.shape[0], 1).to(self.device)
        self.q_network2 = ContinuousQNetwork(state_space.shape[0] + action_space.shape[0], 1).to(self.device)
        self.target_q_network1 = ContinuousQNetwork(state_space.shape[0] + action_space.shape[0], 1).to(self.device)
        self.target_q_network2 = ContinuousQNetwork(state_space.shape[0] + action_space.shape[0], 1).to(self.device)
        self.policy_network = ContinuousPolicyNetwork(state_space.shape[0], action_space.shape[0]).to(self.device)

        self.q_optimizer1 = optim.Adam(self.q_network1.parameters(), lr=self.lr)
        self.q_optimizer2 = optim.Adam(self.q_network2.parameters(), lr=self.lr)
        self.policy_optimizer = optim.Adam(self.policy_network.parameters(), lr=self.lr)
        
        if use_per:
            self.replay_buffer = PrioritizedReplayBuffer(buffer_capacity, buffer_alpha)
        else:
            self.replay_buffer = ReplayBuffer(buffer_capacity)
        
        # self.target_entropy = -torch.prod(torch.Tensor(action_space.shape).to(self.device)).item()
        # #self.log_alpha = torch.zeros(1, requires_grad=True, device=self.device)
        # #self.alpha = self.log_alpha.exp()
        # #self.alpha_optimizer = optim.Adam([self.log_alpha], lr=self.lr)

        self.update_target_networks(self.tau)
        
        self.alpha_values = []
        self.action_scale = torch.tensor(self.action_scale)
        
        hard_update(self.target_q_network1, self.q_network1)
        hard_update(self.target_q_network2, self.q_network2)
        grad_false(self.target_q_network1)
        grad_false(self.target_q_network2)
        
    def choose_action(self, state):
        state = torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(self.device)
        action, _ = self.policy_network.sample(state)
        return (action * self.action_scale).cpu().detach().numpy()[0]

    def update(self):
        if len(self.replay_buffer) < self.batch_size:
            return
        
        if isinstance(self.replay_buffer, PrioritizedReplayBuffer):
            states, actions, rewards, next_states, dones, indices, weights = self.replay_buffer.sample(self.batch_size, self.alpha)
            weights = torch.tensor(weights, dtype=torch.float32).to(self.device)
        else:
            states, actions, rewards, next_states, dones = self.replay_buffer.sample(self.batch_size)
        
        states = torch.tensor(states, dtype=torch.float32).to(self.device)
        next_states = torch.tensor(next_states, dtype=torch.float32).to(self.device)
        actions = torch.tensor(actions, dtype=torch.float32).to(self.device)
        rewards = torch.tensor(rewards, dtype=torch.float32).unsqueeze(1).to(self.device)
        dones = torch.tensor(dones, dtype=torch.float32).unsqueeze(1).to(self.device)
        
        self.update_target_networks()

        with torch.no_grad():
            next_actions, next_log_probs = self.policy_network.sample(next_states)
            next_actions = next_actions * self.action_scale 
            target_q1 = self.target_q_network1(torch.cat([next_states, next_actions], dim=-1))
            target_q2 = self.target_q_network2(torch.cat([next_states, next_actions], dim=-1))
            target_q = torch.min(target_q1, target_q2) - self.alpha * next_log_probs
            target_q_value = rewards + self.gamma * (1 - dones) * target_q

        q1_value = self.q_network1(torch.cat([states, actions], dim=-1))
        q2_value = self.q_network2(torch.cat([states, actions], dim=-1))
        q1_loss = F.mse_loss(q1_value, target_q_value)
        q2_loss = F.mse_loss(q2_value, target_q_value)
        
        if isinstance(self.replay_buffer, PrioritizedReplayBuffer):
            td_errors = torch.abs(q1_value - target_q_value).detach().cpu().numpy()
            self.replay_buffer.update_priorities(indices, td_errors)

            q1_loss = (q1_loss * weights).mean()
            q2_loss = (q2_loss * weights).mean()
        else:
            weights = 1.

        self.q_optimizer1.zero_grad()
        q1_loss.backward()
        self.q_optimizer1.step()
        self.q_optimizer2.zero_grad()
        q2_loss.backward()
        self.q_optimizer2.step()

        new_actions, log_probs = self.policy_network.sample(states)
        new_actions = new_actions * self.action_scale
        q1_new = self.q_network1(torch.cat([states, new_actions], dim=-1))
        q2_new = self.q_network2(torch.cat([states, new_actions], dim=-1))
        q_new = torch.min(q1_new, q2_new)
        policy_loss = (self.alpha * log_probs - q_new).mean()
        self.policy_optimizer.zero_grad()
        policy_loss.backward()
        self.policy_optimizer.step()
        
        #alpha_loss = -(self.log_alpha * (log_probs + self.target_entropy).detach()).mean()
        #self.alpha_optimizer.zero_grad()
        #alpha_loss.backward()
        #self.alpha_optimizer.step()
        #self.alpha = self.log_alpha.exp()
        
        self.alpha_values.append(self.alpha) 
        self.q1_losses.append(q1_loss.item())
        self.q2_losses.append(q2_loss.item())
        self.policy_losses.append(policy_loss.item())

    def update_target_networks(self, tau=None):
        tau = tau or self.tau
        for target_param, param in zip(self.target_q_network1.parameters(), self.q_network1.parameters()):
            target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)
        for target_param, param in zip(self.target_q_network2.parameters(), self.q_network2.parameters()):
            target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)

alpha_values = [0.00001, 0.0001, 0.001, 0.01, 0.1, 1.0]
seeds = [2021, 2022, 2023, 2024, 2025, 2026]

num_trials = 1

overall_result_dir = f"{task_name}_gridsearchsac_gamma{gamma}_tau{tau}_lr{lr}_use_per{use_per}_trials"
if not os.path.exists(overall_result_dir):
    os.makedirs(overall_result_dir)

for trial in range(1, num_trials + 1):
    print(f"Trial {trial}/{num_trials}")

    for alpha in alpha_values:
        
        for seed in seeds:
            
            torch.manual_seed(seed)
            np.random.seed(seed)
            random.seed(seed)
            torch.cuda.manual_seed(seed)
            env = gym.make(task_name)
            env.action_space.seed(seed)
            env.observation_space.seed(seed)
            env.reset(seed=seed)
            state_space = env.observation_space
            action_space = env.action_space
            torch.backends.cudnn.deterministic = True 
            torch.backends.cudnn.benchmark = False
            
            print(alpha)
            print(f"Starting training for alpha = {alpha} | Trial {trial}")
            
            agent = ContinuousSACAgent(state_space=env.observation_space, action_space=env.action_space, action_scale=env.action_space.high[0],
                                    alpha=alpha, gamma=gamma, tau=tau, buffer_capacity=buffer_capacity, batch_size=batch_size, lr=lr, use_per=use_per)
            
            train_rewards = []
            test_rewards = []
            test = []
            
            total_steps = 0
            
            start_time = time.time()
            
            while total_steps < max_steps:
                state, _ = env.reset()
                done = False
                episode_reward = 0

                while not done:
                    action = agent.choose_action(state)
                    next_state, reward, terminated, truncated, info = env.step(action)

                    agent.replay_buffer.add(state, action, reward, next_state, terminated or truncated)
                    agent.update()

                    state = next_state
                    episode_reward += reward
                    total_steps += 1
                    done = terminated or truncated

                    if total_steps % test_interval == 0:
                        test_episode_rewards = []
                        for _ in range(episodes_per_test):
                            test_state, _ = env.reset()
                            test_done = False
                            test_reward = 0
                            while not test_done:
                                test_action = agent.choose_action(test_state)
                                test_next_state, test_reward_step, test_terminated, test_truncated, info = env.step(test_action)
                                test_state = test_next_state
                                test_reward += test_reward_step
                                test_done = test_terminated or test_truncated
                            test_episode_rewards.append(test_reward)

                        avg_test_reward = np.mean(test_episode_rewards)
                        test.append((total_steps, test_episode_rewards[0], test_episode_rewards[1], test_episode_rewards[2], test_episode_rewards[3], test_episode_rewards[4]))
                        test_rewards.append((total_steps, avg_test_reward))
                        print(f"Alpha {alpha} | Trial {trial} | Steps: {total_steps}/{max_steps}, Avg Test Reward: {avg_test_reward}")

                train_rewards.append(episode_reward)
            
            end_time = time.time()
            total_time = end_time - start_time
            
            print(f"run関数の実行時間: {total_time:.4f} 秒")
            
            trial_dir = f"{overall_result_dir}/trial_{trial}"
            if not os.path.exists(trial_dir):
                os.makedirs(trial_dir)

            alpha_dir = f"{trial_dir}/alpha_{alpha}"
            if not os.path.exists(alpha_dir):
                os.makedirs(alpha_dir)
                
            seed_dir = f"{alpha_dir}/seed_{seed}"
            if not os.path.exists(seed_dir):
                os.makedirs(seed_dir)

            save_to_csv([[total_time]], f"{seed_dir}/time.csv", headers=["Time"])
            
            save_to_csv(test, f"{seed_dir}/Test_Return.csv", headers=["Steps", "Test_Reward[0]", "Test_Reward[1]", "Test_Reward[2]", "Test_Reward[3]", "Test_Reward[4]"])

            save_to_csv(test_rewards, f"{seed_dir}/Test_Returns_Trial_{trial}.csv", headers=["Steps", "Avg_Test_Reward"])

            train_rewards_data = list(enumerate(train_rewards, start=1))
            save_to_csv(train_rewards_data, f"{seed_dir}/Training_Rewards_Trial_{trial}.csv", headers=["Episode", "Training_Reward"])

            interval_size = 100

            q1_losses_avg = average_over_intervals(agent.q1_losses, interval_size)
            q2_losses_avg = average_over_intervals(agent.q2_losses, interval_size)
            policy_losses_avg = average_over_intervals(agent.policy_losses, interval_size)

            losses_data_avg = list(zip(q1_losses_avg, q2_losses_avg, policy_losses_avg))

            save_to_csv(losses_data_avg, f"{seed_dir}/Losses_Trial_Averaged.csv", headers=["Q1_Loss_Avg", "Q2_Loss_Avg", "Policy_Loss_Avg"])

            torch.save(agent.q_network1.state_dict(), f"{seed_dir}/q_network1.pth")
            torch.save(agent.q_network2.state_dict(), f"{seed_dir}/q_network2.pth")
            torch.save(agent.policy_network.state_dict(), f"{seed_dir}/policy_network.pth")

            steps, avg_test_returns = zip(*test_rewards)
            plt.plot(steps, avg_test_returns)
            plt.xlabel('Time Steps')
            plt.ylabel('Average Test Returns')
            plt.ylim(0, 6000)
            plt.title(f'Average Test Returns (Trial {trial}, alpha = {alpha})')
            plt.savefig(f"{seed_dir}/Test_Returns_Trial_{trial}.png")
            plt.close()

            plt.figure()
            plt.plot(train_rewards)
            plt.xlabel('Episodes')
            plt.ylabel('Training Rewards')
            plt.ylim(0, 6000)
            plt.title(f'Training Rewards (Trial {trial}, alpha = {alpha})')
            plt.savefig(f"{seed_dir}/Training_Rewards_Trial_{trial}.png")
            plt.close()

            plt.figure()
            plt.plot(agent.q1_losses, label='Q1 Loss')
            plt.plot(agent.q2_losses, label='Q2 Loss')
            plt.plot(agent.policy_losses, label='Policy Loss')
            plt.xlabel('Updates')
            plt.ylabel('Loss')
            plt.legend()
            plt.title(f'Losses (Trial {trial}, alpha = {alpha})')
            plt.savefig(f"{seed_dir}/Losses_Trial_{trial}.png")
            plt.close()

            all_parameter = [task_name, alpha, buffer_alpha, gamma, tau, buffer_capacity, batch_size, lr, max_steps, test_interval, episodes_per_test, use_per]
            headers = ["task_name","alpha", "buffer_alpha", "gamma", "tau", "buffer_capacity", "batch_size", "lr", "max_steps", "test_interval", "episodes_per_test", "use_per"]

            df = pd.DataFrame([all_parameter], columns=headers)

            df.to_csv(f"{seed_dir}/all_parameter.csv", index=False)


env.close()

alpha_averages = {}

average_results_file = f"{overall_result_dir}/average_test_rewards.csv"

with open(average_results_file, mode='w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(["Alpha", "Average_Test_Reward"])

    for alpha in alpha_values:
        total_rewards = []
        for trial in range(1, num_trials + 1):
            trial_dir = f"{overall_result_dir}/trial_{trial}/alpha_{alpha}"
            rewards_file = f"{trial_dir}/Test_Returns_Trial_{trial}.csv"
            
            with open(rewards_file, 'r') as file:
                reader = csv.reader(file)
                next(reader)
                rewards = [float(row[1]) for row in reader]
                total_rewards.append(rewards)

        avg_rewards = np.mean(total_rewards, axis=0)
        alpha_averages[alpha] = avg_rewards
        
        writer.writerow(avg_rewards) 

        print(f"Alpha {alpha} | Average Test Rewards: {avg_rewards}")
