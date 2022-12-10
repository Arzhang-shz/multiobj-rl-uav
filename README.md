# EUAV Gym Environment

This repository provides a modular, reusable implementation of an Unmanned Aerial Vehicle (UAV) localization and communication environment built on OpenAI Gym. It includes:

* **`config.py`**: Centralized configuration of physical constants, UAV parameters, battery limits, and action definitions.
* **`euav_env.py`**: `EUAVEnv` Gym environment subclass that implements:

  * Distance estimation via a shadowing and path-loss model
  * Two-stage trilateration for user localization
  * Communication rate computation for six real users
  * Energy consumption modeling and battery cutoff logic
  * Reward based on throughput and localization accuracy
* **`utils/`**: Helper modules:

  * `propagation.py` – distance estimation functions
  * `geometry.py` – grid-point generation for trilateration
  * `trilateration.py` – two trilateration routines
  * `energy.py` – UAV energy and speed modeling
* **`training/`**: Training scripts for DQN-based agents (e.g. `train_dqn.py`)
* **`models/`**: Output directory for trained weights and plots
* **`tests/`**: Unit tests for each utility and a smoke test for the environment

---

## Installation

1. Clone this repository:

   ```bash
   git clone https://github.com/your-username/euav-gym.git
   cd euav-gym
   ```

2. Create a virtual environment and install dependencies:

   ```bash
   python3 -m venv venv
   source venv/bin/activate
   pip install -r requirements.txt
   ```

## Usage

### Running the environment

```python
from euav_env import EUAVEnv

env = EUAVEnv()
state = env.reset()
for _ in range(100):
    action = env.action_space.sample()
    state, reward, done, info = env.step(action)
    if done:
        break
```

### Training a DQN agent

```bash
python training/train_dqn.py \
    --episodes 2000 \
    --batch_size 32 \
    --epsilon 0.9 \
    --epsilon_decay 0.99 \
    --epsilon_min 0.01 \
    --discount_factor 0.9 \
    --log_interval 50 \
    --model_path models/dqn_weights.h5 \
    --plot_path models/reward_plot.png
```

## Project Structure

```
euav-gym/             # Repository root
+-- config.py
+-- euav_env.py
+-- utils/            # Utility modules
+-- training/         # Training scripts
¦   +-- train_dqn.py
+-- models/           # Outputs (weights, plots)
+-- tests/            # Unit and integration tests
+-- README.md         # Project overview
+-- requirements.txt  # Python dependencies
+-- .gitignore
```

## Testing

```bash
pytest tests/
```

## License

This project is released under the MIT License. See [LICENSE](LICENSE) for details.
