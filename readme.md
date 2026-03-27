# Multi-Agent Task Environment with LLM-based Credit Assignment

This project implements a multi-agent task allocation environment with pluggable credit settlement mechanisms, including an optional LLM-based dynamic credit mode selector.

---

## 🚀 Features

* Multi-agent grid-based environment (Gym-style)
* Modular credit assignment engine
* Multiple credit strategies:

  * crowd-based
  * wait-based
  * hybrid penalties
  * adaptive strategies
* Optional LLM-driven credit mode selection
* Trajectory tracking & analysis tools
* Designed for multi-agent RL research

---

## 📦 Installation

pip install -r requirements.txt

Recommended environment:

* Python ≥ 3.7
* numpy
* gym
* torch
* matplotlib
* scipy
* requests

---

## ⚙️ Configuration (IMPORTANT)

Before running, you MUST configure the LLM selector.

### 1. Set API Endpoint

Edit file:

llm_credit_selector.py

Set:

GPTSAPI_CHAT_URL = "https://your-api-endpoint/v1/chat/completions"

---

### 2. Set Model Name

Example:

model = "gpt-4o"

---

### 3. Set API Key (Environment Variable)

Linux / Mac:

export API_KEY=your_api_key_here

Windows (PowerShell):

setx API_KEY "your_api_key_here"

⚠️ If API_KEY is missing, the system will fallback to default credit mode.

---

## 🧠 Core Components

### Credit Settlement Engine

credit_settlement.py

### LLM Credit Mode Selector

llm_credit_selector.py

### Environment

MultiAgentTaskEnv

---

## ▶️ Usage

env = MultiAgentTaskEnv(params)

state = env.reset()
done = False

while not done:
actions = your_policy(state)
state, reward, done, info = env.step(actions)

---

## 🔧 Credit Modes

env.set_credit_mode("crowd_wait_penalty_v1")

Disable LLM:

variant = "no_llm"

---

## 🧩 Training & Agent Implementation

The current training and agent code are temporarily coupled with project-specific components and are being cleaned.

You are encouraged to implement your own agent and training loop.

Minimal example:

env = MultiAgentTaskEnv(params)

state = env.reset()
done = False

while not done:
actions = your_policy(state)
state, reward, done, info = env.step(actions)

---

## ⚠️ Notes

* LLM calls are rate-limited and cached
* Missing API config will NOT crash the system

---

## 📚 License

MIT

---

## ✨ Acknowledgement

Research on:

* Multi-agent coordination
* Credit assignment
* LLM-assisted decision making
