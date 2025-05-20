# Autonomous Driving using Reinforcement Learning


[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
![Python Version](https://img.shields.io/badge/Python-3.8+-blue.svg)
![PyTorch Version](https://img.shields.io/badge/PyTorch-1.x-red.svg)
![Stable-Baselines3 Version](https://img.shields.io/badge/Stable--Baselines3-1.x-green.svg)
![Gymnasium Version](https://img.shields.io/badge/Gymnasium-0.26+-brightgreen.svg)

This repository contains code for implementing reinforcement learning algorithms to train an autonomous agent for driving tasks in simulated environments. The goal is to develop intelligent agents that can perceive their surroundings, make decisions, and navigate autonomously.

## Table of Contents

- [Introduction](#introduction)
- [Key Features](#key-features)
- [Environments](#environments)
- [Algorithms Implemented](#algorithms-implemented)
- [Getting Started](#getting-started)
  - [Prerequisites](#prerequisites)
  - [Installation](#installation)
- [Usage](#usage)
- [Project Structure](#project-structure)
- [Contributing](#contributing)
- [License](#license)
- [Acknowledgments](#acknowledgments)

## Introduction

Autonomous driving is a rapidly evolving field with the potential to revolutionize transportation. Reinforcement learning (RL) offers a promising approach to tackle the complex decision-making challenges involved in autonomous navigation. This project explores the application of various RL techniques to train virtual vehicles in simulated environments, aiming to achieve intelligent and safe autonomous behavior.

## Key Features

* **Modular Design:** The codebase is structured to be easily extensible and adaptable to different environments and algorithms.
* **Integration with Popular Frameworks:** Leverages powerful libraries like PyTorch, Stable-Baselines3, and Gymnasium.
* **Implementation of Key RL Algorithms:** Includes implementations of fundamental and advanced reinforcement learning algorithms (see [Algorithms Implemented](#algorithms-implemented)).
* **Support for Multiple Environments:** Designed to be compatible with various driving simulation environments (see [Environments](#environments)).
* **Clear Documentation:** Well-commented code and a comprehensive README to facilitate understanding and usage.
* **Experiment Tracking:** Includes tools to track and visualize the training progress of the RL agents.

## Environments

This project currently supports the following driving simulation environments:

* **Custom Highway Environment:** A custom environment (`Highway_env_continuos.py`) designed for continuous control tasks in highway scenarios.
* **HighwayEnv:** A modified version of the popular HighwayEnv library for autonomous driving tasks.

## Algorithms Implemented

The following reinforcement learning algorithms are implemented:

* **Deep Q-Network (DQN):** A value-based method for discrete action spaces.
* **Proximal Policy Optimization (PPO):** A policy gradient method suitable for both discrete and continuous action spaces.
* **Soft Actor-Critic (SAC):** An off-policy algorithm for continuous control tasks.
* **Twin Delayed Deep Deterministic Policy Gradient (TD3):** An improved version of DDPG for continuous action spaces.
* **Advantage Actor-Critic (A2C):** A synchronous, deterministic variant of A3C.

These algorithms are implemented using the Stable-Baselines3 library, which provides a robust and efficient framework for reinforcement learning.

## Getting Started

Follow these steps to set up and run the project on your local machine.

### Prerequisites

Ensure you have the following software installed:

* **Python:** Version 3.8 or higher.
* **pip:** Python package installer.
* **PyTorch:** Version 1.x.
    ```bash
    pip install torch
    ```
* **Stable-Baselines3:**
    ```bash
    pip install stable-baselines3
    ```
* **Gymnasium:**
    ```bash
    pip install gymnasium
    ```
* **NumPy and Matplotlib:**
    ```bash
    pip install numpy matplotlib
    ```

### Installation

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/DevMewada1299/Autonomous-Driving-Reinforcement-Learning.git
    cd Autonomous-Driving-Reinforcement-Learning
    ```
2.  **Install required Python packages:**
    ```bash
    pip install -r requirements.txt
    ```

## Usage

To train an agent, navigate to the `train/` directory and run the desired training script. For example:

```bash
python PPO_cont_train.py
```

To evaluate a trained agent, use the corresponding test script in the `test/` directory. For example:

```bash
python PPO_cont_test.py
```

## Project Structure

```
CMPE260/
├── agents/                # RL agent implementations
├── Custom_Env/           # Custom environments for training
├── models/               # Pre-trained models
├── test/                 # Scripts for testing trained agents
├── train/                # Scripts for training agents
├── logs/                 # Training logs
├── Video/                # Scripts for video generation and evaluation
└── README.md             # Project documentation
```

## Contributing

Contributions are welcome! Please fork the repository and submit a pull request with your changes.

## License

This project is licensed under the MIT License. See the LICENSE file for details.


