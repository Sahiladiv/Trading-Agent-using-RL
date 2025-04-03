import streamlit as st
import torch
import numpy as np
import matplotlib.pyplot as plt
from train_evaluate_ppo_agent import evaluate_ppo, train_ppo
from ppo_agent import PPO
from stock_trading_env import StockTradingEnv
import torch.optim as optim
import pandas as pd
from datetime import datetime

# Load model and env
st.set_page_config(layout="wide")
st.title("📈 PPO Stock Trading Agent Dashboard")

# Configs
initial_balance = 10000
stock_symbol = st.sidebar.text_input("Stock Symbol", value="AAPL")
start_date = st.sidebar.date_input("Start Date", value=datetime(2022, 1, 1))
end_date = st.sidebar.date_input("End Date", value=datetime(2023, 1, 1))
epochs = st.sidebar.slider("Training Epochs", min_value=10, max_value=500, value=100, step=10)
train_button = st.sidebar.button("Train PPO Agent")
evaluate_button = st.sidebar.button("Evaluate Agent")

# Initialize env and model
env = StockTradingEnv(stock_symbol=stock_symbol, start_date=start_date.strftime('%Y-%m-%d'), end_date=end_date.strftime('%Y-%m-%d'), initial_balance=initial_balance)
model = PPO(input_dim=10, output_dim=3)
optimizer = optim.Adam(model.parameters(), lr=0.001)

if train_button:
    st.subheader("🔁 Training PPO Agent")
    portfolio_progress = []
    step_counter = []

    chart_placeholder = st.empty()

    def custom_render():
        portfolio_value = env.balance + env.shares_held * env.stock_data[env.current_step][3]
        portfolio_progress.append(portfolio_value)
        step_counter.append(len(portfolio_progress))
        chart_placeholder.line_chart(
            data={"Portfolio Value": portfolio_progress},
            use_container_width=True
        )

    # Monkey patch env.render
    env.render = custom_render

    policy_losses, value_losses, entropies = train_ppo(env, model, optimizer, epochs=epochs)

    torch.save(model.state_dict(), "ppo_stock_trader.pt")
    st.success("Model trained and saved as 'ppo_stock_trader.pt'")

    # Final static plot
    st.subheader("📈 Final Portfolio Value Over Training")
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(portfolio_progress)
    ax.set_title("Portfolio Value During Training")
    ax.set_xlabel("Step")
    ax.set_ylabel("Portfolio Value ($)")
    st.pyplot(fig)

if evaluate_button:
    st.subheader("📊 Evaluating Trained Agent")
    model.load_state_dict(torch.load("ppo_stock_trader.pt", map_location=torch.device('cpu')))
    model.eval()
    episodes = st.sidebar.slider("Evaluation Episodes", min_value=1, max_value=50, value=5)

    st.write(f"Evaluating PPO Agent on **{stock_symbol}** for **{episodes} episodes**...")
    all_rewards, all_portfolios = evaluate_ppo(env, model, episodes=episodes)

    # Plot Portfolio Growth with Buy/Sell Markers
    st.subheader("📊 Portfolio Value Over Time with Buy/Sell Markers")
    fig, ax = plt.subplots(figsize=(10, 4))
    for p in all_portfolios:
        ax.plot(p, label="Portfolio")

    prices = env.stock_data[:, 3]
    actions = []
    state = env.reset()
    done = False
    while not done:
        state_tensor = torch.tensor(state, dtype=torch.float32)
        action_probs, _ = model(state_tensor)
        action = torch.argmax(action_probs).item()
        actions.append(action)
        state, _, done, _ = env.step(action)

    buy_points = [i for i, a in enumerate(actions) if a == 2]
    sell_points = [i for i, a in enumerate(actions) if a == 0]

    ax.scatter(buy_points, [prices[i] for i in buy_points], marker="^", color="green", label="Buy")
    ax.scatter(sell_points, [prices[i] for i in sell_points], marker="v", color="red", label="Sell")

    ax.set_xlabel("Timestep")
    ax.set_ylabel("Portfolio Value ($)")
    ax.set_title("Agent Portfolio Growth with Buy/Sell Actions")
    ax.legend()
    st.pyplot(fig)

    # Stats
    final_values = [p[-1] for p in all_portfolios]
    st.markdown(f"**Average Final Portfolio Value:** ${np.mean(final_values):,.2f}")
    st.markdown(f"**Max Portfolio Value:** ${np.max(final_values):,.2f}")
    st.markdown(f"**Min Portfolio Value:** ${np.min(final_values):,.2f}")

    # Sharpe Ratio and Drawdown
    returns = np.diff(all_portfolios[0]) / all_portfolios[0][:-1]
    sharpe_ratio = np.mean(returns) / (np.std(returns) + 1e-8) * np.sqrt(252)
    cumulative = np.maximum.accumulate(all_portfolios[0])
    drawdowns = (cumulative - all_portfolios[0]) / cumulative
    max_drawdown = np.max(drawdowns)

    st.markdown(f"**Sharpe Ratio:** {sharpe_ratio:.4f}")
    st.markdown(f"**Max Drawdown:** {max_drawdown:.2%}")

    # Display reward summary
    st.subheader("💰 Reward Summary")
    st.write("Total Rewards per Episode:", all_rewards)
    st.line_chart(all_rewards)