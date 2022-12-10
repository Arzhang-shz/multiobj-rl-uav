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

The following figures summarize simulation results and provide insights into the trade-offs between throughput, localization error, and UAV speed.

### **Fig. 1: System Architecture**
<!-- Fig. 1: System Architecture -->
<img src="Figures/fig1.png" alt="System Architecture" width="500"/>

---

### **Fig. 2: Trilateration-Based Localization**
<!-- Fig. 2: Trilateration-Based Localization -->
<img src="Figures/fig2.png" alt="Trilateration" width="500"/>

---

### **Fig. 3: Power Consumption vs UAV Speed**
<!-- Fig. 3: Power Consumption vs UAV Speed -->
<img src="Figures/speed_fig.png" alt="Power Consumption" width="500"/>

---

### **Fig. 4: RSSI Ring Intersections**
<!-- Fig. 4: RSSI Ring Intersections -->
<img src="Figures/fig_4.png" alt="Ring Intersection" width="500"/>

---

### **Fig. 5a: Transmitted Bits**
<!-- Fig. 5a: Transmitted Bits -->
<img src="Figures/my_figure_1.png" alt="Transmitted Bits" width="500"/>

---

### **Fig. 5b: Localization Error**
<!-- Fig. 5b: Localization Error -->
<img src="Figures/my_figure_2.png" alt="Localization Error" width="500"/>

---

### **Fig. 5c: UAV Flight Time**
<!-- Fig. 5c: UAV Flight Time -->
<img src="Figures/my_figure_4.png" alt="UAV Flight Time" width="500"/>

---

### **Fig. 6a: Throughput vs. Localization Error**
<!-- Fig. 6a: Throughput vs. Localization Error -->
<img src="Figures/new_fig_1.png" alt="Throughput vs. Localization Error" width="500"/>

---

### **Fig. 6b: UAV Flight Time (speed)**
<!-- Fig. 6b: UAV Flight Time (speed) -->
<img src="Figures/new_fig_2.png" alt="Flight Time at Different Speeds" width="500"/>

---

### **Fig. 7: Performance Under Different Reward Weights**
<!-- Fig. 7: Performance Under Different Reward Weights -->
<img src="Figures/10_w_1.png" alt="Reward Weight 1" width="500"/>
<img src="Figures/10_w_2.png" alt="Reward Weight 2" width="500"/>
<img src="Figures/10_w_3.png" alt="Reward Weight 3" width="500"/>


## Citation

If you use this code, please cite the associated thesis [*Machine Learning Techniques for UAV-assisted Networks* (PDF)](https://theses.hal.science/tel-03889218/file/118554_SHAHBAZI_2022_archivage.pdf).


## License

This project is licensed under the MIT License. See the `LICENSE` file for details.

## Acknowledgements

This research was supported by the European Commission through the H2020 PAINLESS Project under Grant 812991.
