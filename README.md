# Multi-Objective Trajectory Design for UAV-Assisted Dual-Functional Radar-Communication Network: A Reinforcement Learning Approach

This project explores the optimal trajectory for maximizing communication throughput and minimizing localization error in a Dual-Functional Radar Communication (DFRC) network using an unmanned aerial vehicle (UAV). A single UAV serves a group of communication users and localizes ground targets simultaneously.

## Key Features

*   **Dual-Functional Radar-Communication (DFRC):** A single UAV performs both communication and radar sensing tasks, reducing resource consumption and interference.
*   **Reinforcement Learning (RL):** A novel framework based on RL enables the UAV to autonomously find its trajectory to improve localization accuracy and maximize the number of transmitted bits.
*   **Multi-Objective Optimization:** A multi-objective optimization problem is formulated to jointly optimize communication throughput and localization error, considering UAV energy constraints.
*   **Double Deep Q-Network (DDQN):** The DDQN algorithm is used to train the UAV agent to make optimal decisions in a complex and dynamic environment.

## Methodology

The project utilizes a Double Deep Q-Network (DDQN) agent to control the UAV's trajectory. The agent learns to optimize its path based on the following factors:

*   UAV location
*   Estimated distances and localization error of ground targets
*   Communication rate of users
*   UAV energy consumption

The reward function is designed to balance communication throughput and localization accuracy, with weights assigned to each objective.

## Results

The simulation results demonstrate the trade-off between communication throughput and localization error.

*   **UAV Speed:** Higher UAV speeds lead to better localization accuracy but lower communication throughput, and vice versa.
*   **Reward Weights:** Adjusting the weights in the reward function allows prioritizing either communication or localization performance.

The following figures illustrate the key results:

*   **System Architecture:** ![System Architecture](paper/Figures/fig1.png)
*   **Trilateration for Localization:** ![Trilateration for Localization](paper/Figures/fig2.png)
*   **Power Consumption vs. UAV Speed:** ![Power Consumption vs. UAV Speed](paper/Figures/speed_fig.png)
*   **Training Curves:**
    *   Number of Transmitted Bits: ![Number of Transmitted Bits](paper/Figures/my_figure_1.png)
    *   Localization Error: ![Localization Error](paper/Figures/my_figure_2.png)
    *   UAV Flight Time: ![UAV Flight Time](paper/Figures/my_figure_4.png)
    *   Accumulated Reward: ![Accumulated Reward](paper/Figures/my_figure_5.png)
*   **Impact of UAV Speed:**
    *   Number of Transmitted Bits and Localization Error: ![Number of Transmitted Bits and Localization Error](paper/Figures/new_fig_1.png)
    *   UAV Flight Time: ![UAV Flight Time](paper/Figures/new_fig_2.png)
*   **Performance Comparison for Different Weights and UAV Speeds:**
    *   ![Performance Comparison](paper/Figures/10_w_1.png)
    *   ![Performance Comparison](paper/Figures/10_w_2.png)
    *   ![Performance Comparison](paper/Figures/10_w_3.png)
    *   ![Performance Comparison](paper/Figures/20_w_1.png)
    *   ![Performance Comparison](paper/Figures/20_w_2.png)
    *   ![Performance Comparison](paper/Figures/20_w_3.png)
    *   ![Performance Comparison](paper/Figures/30_w_1.png)
    *   ![Performance Comparison](paper/Figures/30_w_2.png)
    *   ![Performance Comparison](paper/Figures/30_w_3.png)

## Getting Started

1.  **Clone the repository:**

    ```bash
    git clone <repository_url>
    cd <repository_name>
    ```

2.  **Install the dependencies:**

    ```bash
    pip install -r requirements.txt
    ```

3.  **Run the training script:**

    ```bash
    python training/train_dqn.py
    ```

    or

    ```bash
    python Double_DQN.py
    ```

## Contributing

Contributions are welcome! Please feel free to submit pull requests or open issues to suggest improvements or report bugs.

## License

[License]
