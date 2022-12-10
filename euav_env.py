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
        return 34  # number of state variables

    def reset(self):
        # Initialize state array
        self.state = np.zeros(self._state_dim(), dtype=np.float64)
        self.state[0] = self.cfg.initial_theta  # theta
        self.state[1] = self.cfg.initial_t      # time step
        self.state[2] = self.cfg.initial_x      # UAV X
        self.state[3] = self.cfg.initial_y      # UAV Y
        # battery, rates, speeds default to 0
        self.steps_beyond_done = None
        return self.state

    def step(self, action: int):
        # Validate action
        if not self.action_space.contains(action):
            raise ValueError(f"Invalid action: {action}")
        mov = COMB[action]
        logger.debug(f"Action taken (dx,dy): {mov}")

        # Snapshot old state
        state_old = self.state.copy()

        # 1) Time and position update
        new_t = int(state_old[1] + 1)
        new_x, new_y = state_old[2] + mov[0], state_old[3] + mov[1]

        # 2) Distance estimates for 4 users
        estimates = [distance_estimate(x, y, new_x, new_y, self.cfg.alpha)
                     for x, y in USER_EST_POSITIONS]
        d_flat = []
        for d1, d2, _ in estimates:
            d_flat.extend([d1, d2])

        # 3) Trilateration for 4 users
        mid_x = (state_old[2] + new_x) / 2
        mid_y = (state_old[3] + new_y) / 2
        total_stop = 0
        loc_err_sum = 0
        radii = []
        est_positions = []

        # compute radii
        for i, (_, _, _) in enumerate(estimates):
            max_d = max(estimates[i][0], estimates[i][1], state_old[4+2*i], state_old[5+2*i])
            radii.append(int(2 * np.sqrt(max_d**2)))

        for i in range(4):
            M = generate_points(mid_x, mid_y, radii[i])
            if new_t == 1:
                _, pos, err, stop_i = trilateration1(
                    M,
                    state_old[2], state_old[3], state_old[5+2*i], state_old[4+2*i],
                    new_x, new_y, estimates[i][1], estimates[i][0],
                    state_old[18+2*i], state_old[19+2*i]
                )
            else:
                M_flat = M.reshape(2, -1).T
                _, pos, err, stop_i = trilateration2(
                    M_flat,
                    new_x, new_y,
                    estimates[i][1], estimates[i][0],
                    state_old[18+2*i], state_old[19+2*i]
                )
            est_positions.extend(pos)
            total_stop += stop_i
            loc_err_sum += err
        loc_err_avg = loc_err_sum / 4

        # 4) Communication rates for 6 real users
        rates = []
        for xr, yr in USER_REAL_POSITIONS:
            d3d = np.sqrt((xr - new_x)**2 + (yr - new_y)**2 + self.cfg.H**2)
            pathloss = self.cfg.klos * (d3d**self.cfg.Beta_los)
            snr = (self.cfg.P / pathloss) / self.cfg.noise_power
            rates.append(np.log2(1 + snr))
        new_sum_real = float(np.mean(rates))
        # cumulative average rate
        if new_t == 1:
            avg_rate = new_sum_real
        else:
            avg_rate = state_old[29] + new_sum_real

        # 5) Energy consumption and battery update
        # compute_energy returns (E_total, cut_off_flag, speed, avg_speed)
        E_total, cut_off_flag, speed, avg_speed = compute_energy(
            self.cfg, mov, state_old[30], state_old[33]
        )

        # 6) Reward calculation
        if not cut_off_flag:
            # normalized metrics
            reward_loc = self.cfg.min_loc_error / loc_err_avg
            reward_com = new_sum_real / self.cfg.max_avg_rate
            # weights
            lambda_com, lambda_loc = 5.0, 1.0
            reward = lambda_com * reward_com + lambda_loc * reward_loc
        else:
            reward = 0.0

        # 7) Termination condition
        done = bool(cut_off_flag)

        # Update state vector
        self.state[1] = new_t
        self.state[2], self.state[3] = new_x, new_y
        self.state[4:12] = np.array(d_flat, dtype=np.float64)
        self.state[12], self.state[13] = mid_x, mid_y
        self.state[14:18] = np.array(radii, dtype=np.float64)
        self.state[18:26] = np.array(est_positions, dtype=np.float64)
        self.state[26] = loc_err_avg
        self.state[27] = total_stop
        self.state[28] = new_sum_real
        self.state[29] = avg_rate
        self.state[30] = E_total
        self.state[31] = cut_off_flag
        self.state[32] = speed
        self.state[33] = avg_speed

        info = {}
        return self.state.copy(), reward, done, info

    def render(self, mode='human'):
        # Optional: add visualization of UAV and users
        pass

    def close(self):
        pass
