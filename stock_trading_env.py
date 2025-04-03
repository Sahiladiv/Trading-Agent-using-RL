import gym
import numpy as np
import pandas as pd
import yfinance as yf
from gym.spaces import Box, Discrete
from gym import Env

class StockTradingEnv(Env):

    def __init__(self, stock_symbol='AAPL', start_date='2020-01-01', end_date='2024-01-01', initial_balance=10000):
        super(StockTradingEnv, self).__init__()

        # Load stock data
        self.stock_data = self.get_stock_data(stock_symbol, start_date, end_date)
        self.max_steps = len(self.stock_data) - 1

        # Normalize statistics
        self.mean = np.mean(self.stock_data, axis=0)
        self.std = np.std(self.stock_data, axis=0) + 1e-8

        # Trading parameters
        self.initial_balance = initial_balance
        self.balance = initial_balance
        self.shares_held = 0
        self.current_step = 0
        self.done = False
        self.last_portfolio_value = initial_balance

        # Action and observation space
        self.action_space = Discrete(3)  # 0: Sell, 1: Hold, 2: Buy
        self.observation_space = Box(low=-np.inf, high=np.inf, shape=(9,), dtype=np.float32)

    def get_stock_data(self, stock_symbol, start_date, end_date):
        """Fetch historical stock data from Yahoo Finance"""
        df = yf.download(stock_symbol, start=start_date, end=end_date)
        df['SMA_10'] = df['Close'].rolling(window=10).mean()
        df['SMA_50'] = df['Close'].rolling(window=50).mean()
        df.fillna(method='bfill', inplace=True)
        return df[['Open', 'High', 'Low', 'Close', 'Volume', 'SMA_10', 'SMA_50']].values

    def reset(self):
        self.balance = self.initial_balance
        self.shares_held = 0
        self.current_step = np.random.randint(0, min(20, self.max_steps))
        self.done = False
        self.last_portfolio_value = self.initial_balance
        return self._next_observation()

    def _next_observation(self):
        obs = self.stock_data[self.current_step]
        obs_normalized = (obs - self.mean) / self.std
        portfolio_state = [self.balance / 1e4, self.shares_held, self.last_portfolio_value / 1e4]
        state = np.concatenate((obs_normalized, portfolio_state))
        return state

    def step(self, action):
        current_price = self.stock_data[self.current_step][3]
        prev_portfolio_value = self.balance + self.shares_held * current_price

        reward = 0
        TRANSACTION_FEE_PERCENT = 0.001  # 0.1%

        if action == 0:  # Sell
            if self.shares_held > 0:
                shares_sold = self.shares_held
                proceeds = shares_sold * current_price
                fee = TRANSACTION_FEE_PERCENT * proceeds
                self.balance += proceeds - fee
                self.shares_held = 0
                reward += (self.balance - self.last_portfolio_value) / self.initial_balance

        elif action == 2:  # Buy
            shares_to_buy = self.balance // current_price
            if shares_to_buy > 0:
                cost = shares_to_buy * current_price
                fee = TRANSACTION_FEE_PERCENT * cost
                self.balance -= cost + fee
                self.shares_held += shares_to_buy

        elif action == 1:  # Hold
            reward -= 0.01  # Small penalty for idle holding

        # Move to next step
        self.current_step += 1
        if self.current_step >= self.max_steps:
            self.done = True

        portfolio_value = self.balance + self.shares_held * current_price
        self.last_portfolio_value = portfolio_value

        # Reward is scaled change in portfolio
        reward += (portfolio_value - prev_portfolio_value) / self.initial_balance
        reward = np.clip(reward * 100, -1, 1)  # Scale and clip

        next_obs = self._next_observation()
        return next_obs, reward, self.done, {}


    def render(self):
        portfolio_value = self.balance + self.shares_held * self.stock_data[self.current_step][3]
        print(f'Step: {self.current_step}, Balance: {self.balance:.2f}, Shares Held: {self.shares_held}, Portfolio Value: {portfolio_value:.2f}')
