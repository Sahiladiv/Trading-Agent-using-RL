import gym
import numpy as np
import pandas as pd
import yfinance as yf
from gym.spaces import Box, Discrete
from gym import Env
import torch.nn as nn

class StockTradingEnv(Env):

    def __init__(self, stock_symbol='AAPL', start_date='2020-01-01', end_date='2023-01-01', initial_balance=10000):
        super(StockTradingEnv, self).__init__()
        
        # Load stock data
        self.stock_data = self.get_stock_data(stock_symbol, start_date, end_date)
        self.max_steps = len(self.stock_data) - 1
        
        # Trading parameters
        self.initial_balance = initial_balance
        self.balance = initial_balance
        self.shares_held = 0
        self.current_step = 0
        self.done = False
        
        # Define action space (0: Sell, 1: Hold, 2: Buy)
        self.action_space = Discrete(3)
        
        # Define observation space (OHLCV + Technical Indicators + Portfolio State)
        self.observation_space = Box(low=-np.inf, high=np.inf, shape=(8,), dtype=np.float32)

    def get_stock_data(self, stock_symbol, start_date, end_date):
        """Fetch historical stock data from Yahoo Finance"""
        df = yf.download(stock_symbol, start=start_date, end=end_date)
        df['SMA_10'] = df['Close'].rolling(window=10).mean()
        df['SMA_50'] = df['Close'].rolling(window=50).mean()
        df.fillna(method='bfill', inplace=True)
        return df[['Open', 'High', 'Low', 'Close', 'Volume', 'SMA_10', 'SMA_50']].values
    
    def reset(self):
        """Reset environment state"""
        self.balance = self.initial_balance
        self.shares_held = 0
        self.current_step = 0
        self.done = False
        return self._next_observation()
    
    def _next_observation(self):
        """Get the next state observation"""
        obs = self.stock_data[self.current_step]
        portfolio_state = [self.balance, self.shares_held]
        state = np.concatenate((obs, portfolio_state))
        return state
    
    def step(self, action):
        """Execute trade action"""
        current_price = self.stock_data[self.current_step][3]  # Close price
        
        if action == 0:  # Sell
            self.balance += self.shares_held * current_price
            self.shares_held = 0
        elif action == 2:  # Buy
            shares_to_buy = self.balance // current_price
            self.shares_held += shares_to_buy
            self.balance -= shares_to_buy * current_price
        
        self.current_step += 1
        if self.current_step >= self.max_steps:
            self.done = True
        
        next_obs = self._next_observation()
        reward = (self.balance + (self.shares_held * current_price)) - self.initial_balance
        if self.current_step < self.max_steps - 1:  # Prevent out-of-bounds error
            reward += (self.shares_held * (self.stock_data[self.current_step + 1][3] - current_price))

        print(f"Reward: {reward}, Balance: {self.balance}, Shares: {self.shares_held}, Action:{action}")

        return next_obs, reward, self.done, {}
        
    def render(self):
        """Render the current portfolio state"""
        print(f'Step: {self.current_step}, Balance: {self.balance}, Shares Held: {self.shares_held}')