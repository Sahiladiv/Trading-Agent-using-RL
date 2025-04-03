import gym
import numpy as np
import pandas as pd
import yfinance as yf
from gym.spaces import Box, Discrete
from gym import Env
import torch.nn as nn
from stock_trading_env import StockTradingEnv
from ppo_agent import PPO
import matplotlib.pyplot as plt
import torch.optim as optim
import torch
from collections import Counter

def train_ppo(env, model, optimizer, epochs=500, gamma=0.99, clip_epsilon=0.2, batch_size=32, lambda_gae=0.95):
    policy_losses = []
    value_losses = []
    entropies = []

    initial_entropy_coef = 0.2

    for episode in range(epochs):
        state = env.reset()
        log_probs, values, rewards, actions, entropies_ep = [], [], [], [], []
        done = False

        while not done:
            state_tensor = torch.tensor(state, dtype=torch.float32)
            action_probs, value = model(state_tensor)
            dist = torch.distributions.Categorical(action_probs)
            action = dist.sample()

            log_prob = dist.log_prob(action)
            entropy = dist.entropy()

            next_state, reward, done, _ = env.step(action.item())

            log_probs.append(log_prob)
            values.append(value.squeeze())
            rewards.append(reward)
            actions.append(action)
            entropies_ep.append(entropy)

            state = next_state

        # Process episode
        values = torch.stack(values)
        log_probs = torch.stack(log_probs)
        entropies_ep = torch.stack(entropies_ep).mean()

        # GAE
        advantages, gae = [], 0
        for t in reversed(range(len(rewards))):
            next_value = values[t + 1] if t < len(rewards) - 1 else 0
            delta = rewards[t] + gamma * next_value - values[t]
            gae = delta + gamma * lambda_gae * gae
            advantages.insert(0, gae)

        advantages = torch.tensor(advantages, dtype=torch.float32)
        returns = advantages + values.detach()
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        # PPO Update
        ratio = torch.exp(log_probs - log_probs.detach())
        surr1 = ratio * advantages
        surr2 = torch.clamp(ratio, 1 - clip_epsilon, 1 + clip_epsilon) * advantages
        policy_loss = -torch.min(surr1, surr2).mean()
        value_loss = (returns - values).pow(2).mean()

        entropy_coef = initial_entropy_coef * (0.99 ** episode)
        loss = policy_loss + 0.5 * value_loss - entropy_coef * entropies_ep

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.5)
        optimizer.step()

        policy_losses.append(policy_loss.item())
        value_losses.append(value_loss.item())
        entropies.append(entropies_ep.item())

        env.render()
        if episode % 10 == 0:
            from collections import Counter
            action_counter = Counter([a.item() for a in actions])
            print(f"Episode {episode}, Loss: {loss.item():.4f}, Policy Loss: {policy_loss.item():.4f}, Value Loss: {value_loss.item():.4f}, Entropy: {entropies_ep.item():.4f}, Actions: {action_counter}")

    return policy_losses, value_losses, entropies


def evaluate_ppo(env, model, episodes=10):
    all_rewards = []
    portfolio_values = []

    for episode in range(episodes):
        state = env.reset()
        done = False
        total_reward = 0
        portfolio_value = [env.initial_balance]

        while not done:
            state_tensor = torch.tensor(state, dtype=torch.float32)
            action_probs, _ = model(state_tensor)
            action = torch.argmax(action_probs).item()

            state, reward, done, _ = env.step(action)
            total_reward += reward
            portfolio_value.append(env.balance + (env.shares_held * env.stock_data[env.current_step][3]))

        all_rewards.append(total_reward)
        portfolio_values.append(portfolio_value)

    return all_rewards, portfolio_values