# 📈 PPO Stock Trading Agent with Streamlit Dashboard:

🚀 Features:
🧠 Trains a PPO agent to buy/sell/hold based on stock indicators
🧾 Custom StockTradingEnv using real stock data (yfinance)
🧩 Neural network with shared actor-critic layers (PyTorch)
📊 Streamlit dashboard for:
  - Real-time training visualization
  - Buy/Sell action markers
  - Sharpe Ratio and Max Drawdown calculation
📅 Customizable training/testing date ranges

💾 Save/load model weights for reuse or evaluation
🧠 Environment Design: StockTradingEnv
This custom OpenAI Gym environment simulates a stock trading experience on historical data using reinforcement learning.

📥 State Space
The observation space is a 10-dimensional vector representing market and portfolio conditions:

### 📥 State Space

| Index | Feature           | Description                            |
|-------|-------------------|----------------------------------------|
| 0     | `Open`            | Opening price of the stock             |
| 1     | `High`            | Highest price of the day               |
| 2     | `Low`             | Lowest price of the day                |
| 3     | `Close`           | Closing price                          |
| 4     | `Volume`          | Daily trading volume                   |
| 5     | `SMA_10`          | 10-day Simple Moving Average           |
| 6     | `SMA_50`          | 50-day Simple Moving Average           |
| 7     | `Balance`         | Current cash balance (normalized)      |
| 8     | `Shares Held`     | Number of shares currently held        |
| 9     | `Portfolio Value` | Total portfolio value (normalized)     |


🎮 Action Space
The action space is a discrete space of 3 possible actions:

Action	Description

| Action | Description                          |
|--------|--------------------------------------|
| `0`    | **Sell All**: Sell all held shares   |
| `1`    | **Hold**: Take no action             |
| `2`    | **Buy All**: Buy max shares with cash|


💸 Transaction Fee
Every buy or sell incurs a transaction fee of 0.1%

🎯 Reward Function
The reward at each step is proportional to the change in portfolio value, scaled and clipped for training stability.
✅ Realized Profit Boost: Reward spikes when a sell action realizes gains
❌ Idle Penalty: Small penalty -0.01 for holding too long without action
