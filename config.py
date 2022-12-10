"""
EUAV Gym Environment using modular configuration and utilities.
"""

import logging
from gym import Env, spaces
import numpy as np

from config import EnvConfig, COMB, USER_EST_POSITIONS, USER_REAL_POSITIONS
from utils.propagation import distance_estimate
from utils.geometry import generate_points
from utils.trilateration import trilateration1, trilateration2
from utils.energy import compute_energy

# Initialize logger for debug output
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

class EUAVEnv(Env):
    def __init__(self, config: EnvConfig = EnvConfig()):
        self.cfg = config

        # Observation and action spaces
        low = -np.inf * np.ones(self._state_dim(), dtype=np.float32)
        high = np.inf * np.ones(self._state_dim(), dtype=np.float32)
        self.observation_space = spaces.Box(low, high, dtype=np.float32)
        self.action_space = spaces.Discrete(len(COMB))

        # Internal state
        self.seed()
        self.state = None
        self.steps_beyond_done = None
        self.reset()

    def _state_dim(self) -> int:
        # Total number of state variables
        return 34  # update if state structure changes

    def reset(self):
        # Reset to initial config values
        self.state = np.zeros(self._state_dim(), dtype=np.float64)
        self.state[0] = self.cfg.initial_theta  # theta
        self.state[1] = self.cfg.initial_t      # time step
        self.state[2] = self.cfg.initial_x      # UAV X
        self.state[3] = self.cfg.initial_y      # UAV Y
        # distance estimates and other variables remain zero
        self.steps_beyond_done = None
        return self.state

    def step(self, action: int):
        # Validate action
        if not self.action_space.contains(action):
            raise ValueError(f"Invalid action: {action}")
        mov = COMB[action]
        logger.debug(f"Action taken (dx,dy): {mov}")

        # 1) Update time and position
        new_t = int(self.state[1] + 1)
        new_x = self.state[2] + mov[0]
        new_y = self.state[3] + mov[1]

        # 2) Estimate distances for each user
        # distance_estimate returns (d_est1, d_est2, horizontal_dist)
        estimates = [distance_estimate(x, y, new_x, new_y, self.cfg.alpha)
                     for x, y in USER_EST_POSITIONS]
        # Flatten first two distance estimates for each user into state indices 4-11
        d_flat = []
        for d1, d2, _ in estimates:
            d_flat.extend([d1, d2])
        self.state[4:12] = np.array(d_flat, dtype=np.float64)

        # Update UAV state variables
        self.state[1] = new_t
        self.state[2] = new_x
        self.state[3] = new_y

        # TODO: implement 3) trilateration, 4) communication rates, 5) energy, 6) reward, 7) done

        next_state = self.state.copy()
        reward = 0.0
        done = False
        info = {}
        return next_state, reward, done, info

    def render(self, mode='human'):
        # Optional: add visualization of UAV and users
        pass

    def close(self):
        pass
