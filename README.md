# Multi-Objective Trajectory Design for UAV-Assisted Dual-Functional Radar-Communication Network: A Reinforcement Learning Approach

This repository contains the code and core results for a research project exploring the use of reinforcement learning (RL) to design optimal UAV trajectories in a dual-functional radar-communication (DFRC) network. The project aims to simultaneously maximize communication throughput and minimize localization error under UAV energy constraints.

## Overview

This project is part of my PhD research.
- **Dual-Functional Radar-Communication (DFRC):** A single UAV is used to serve communication users and perform ground target localization simultaneously.
- **Multi-Objective Optimization:** A reinforcement learning framework balances conflicting objectives: communication efficiency and localization precision.
- **Energy-Constrained Trajectory Design:** UAV motion is constrained by its limited energy capacity, affecting flight time and sensing opportunities.
- **Reinforcement Learning Approach:** A Double Deep Q-Network (DDQN) agent learns optimal trajectories in complex, partially observable environments.

## Highlights

- Formulation of a multi-objective optimization problem for UAV trajectory.
- Custom reward function integrating localization error and communication rate.
- Use of RSS-based trilateration with variable shadowing for localization.
- Simulation of the UAV's behavior under various reward weights and speeds.

## Repository Structure

```
Figures/                # Project Figures
training/               # Training script example for RL model
    train_dqn.py        # Deep Q-Network training script for the EUAV environment.

requirements.txt        # Python dependencies
README.md               # Project documentation
config.py               # Configuration file for the EUAV environment, defining various parameters and settings.
euav_env.py             # Gym Environment using modular configuration and utilities.
Utils/                  # Directory containing utility modules.
    energy.py           # Energy consumption and battery update.
    geometry.py         # Geometry utilities for the EUAV environment.
    propagation.py      # Propagation model utilities for the EUAV environment.
    trilateration.py    # Trilateration utilities for the EUAV environment.
```

## Setup

### 1. Clone the Repository

```bash
git clone https://github.com/yourusername/DFRC-RL-UAV.git
cd DFRC-RL-UAV
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

### 3. Train the RL Agent

```bash
python training/train_dqn.py
```

or

```bash
python Double_DQN.py
```

## Key Results

Below are key results from our simulations, illustrating the trade-offs between throughput, localization error, energy consumption, and UAV speed.

<p align="center">
  <img src="Figures/fig1.png" alt="System Architecture" width="400"/>
  <br/>
  <em>Figure 1: The UAV communicates with users and localizes targets in a DFRC setup.</em>
</p>

<p align="center">
  <img src="Figures/fig2.png" alt="Trilateration" width="400"/>
  <br/>
  <em>Figure 2: Concentric RSSI circles are used to estimate user positions via trilateration.</em>
</p>

<p align="center">
  <img src="Figures/speed_fig.png" alt="Power Consumption" width="400"/>
  <br/>
  <em>Figure 3: Relationship between UAV speed and power consumption components.</em>
</p>

<p align="center">
  <img src="Figures/fig_4.png" alt="RSSI Ring Intersection" width="400"/>
  <br/>
  <em>Figure 4: Intersecting RSSI rings help localize ground targets.</em>
</p>

<p align="center">
  <img src="Figures/my_figure_1.png" alt="Transmitted Bits" width="400"/>
  <br/>
  <em>Figure 5a: Transmitted bits over episodes during training.</em>
</p>

<p align="center">
  <img src="Figures/my_figure_2.png" alt="Localization Error" width="400"/>
  <br/>
  <em>Figure 5b: Convergence of localization error during training.</em>
</p>

<p align="center">
  <img src="Figures/my_figure_4.png" alt="Flight Time" width="400"/>
  <br/>
  <em>Figure 5c: UAV flight time evolution during DDQN training.</em>
</p>

<p align="center">
  <img src="Figures/new_fig_1.png" alt="Throughput vs Localization" width="400"/>
  <br/>
  <em>Figure 6a: Trade-off between throughput and localization error at different speeds.</em>
</p>

<p align="center">
  <img src="Figures/new_fig_2.png" alt="Flight Time vs Speed" width="400"/>
  <br/>
  <em>Figure 6b: UAV flight time changes with varying speeds.</em>
</p>

<p align="center">
  <img src="Figures/10_w_1.png" alt="Reward Weight 1" width="400"/>
  <br/>
  <em>Figure 7a: Performance under reward setting w₁.</em>
</p>

<p align="center">
  <img src="Figures/10_w_2.png" alt="Reward Weight 2" width="400"/>
  <br/>
  <em>Figure 7b: Performance under reward setting w₂.</em>
</p>

<p align="center">
  <img src="Figures/10_w_3.png" alt="Reward Weight 3" width="400"/>
  <br/>
  <em>Figure 7c: Performance under reward setting w₃.</em>
</p>



## Citation

If you use this code, please cite the associated thesis [*Machine Learning Techniques for UAV-assisted Networks* (PDF)](https://theses.hal.science/tel-03889218/file/118554_SHAHBAZI_2022_archivage.pdf).


## License

This project is licensed under the MIT License. See the `LICENSE` file for details.

## Acknowledgements

This research was supported by the European Commission through the H2020 PAINLESS Project under Grant 812991.
