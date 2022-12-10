"""
Propagation model utilities for the EUAV environment.
"""
import numpy as np
from config import EnvConfig

# Load configuration constants
cfg = EnvConfig()

def distance_estimate(x, y, x_uav, y_uav, alpha):
    """
    Estimate the user-to-UAV distance range based on received power and reflection.

    Args:
        x, y: User true coordinates
        x_uav, y_uav: UAV coordinates
        alpha: Path-loss exponent parameter (unused here, kept for API compatibility)

    Returns:
        d_est1: Lower-bound distance estimate (m)
        d_est2: Upper-bound distance estimate (m)
        r: Horizontal distance (m)
    """
    # 3D and horizontal distances
    d = np.sqrt((x - x_uav)**2 + (y - y_uav)**2 + cfg.H**2)
    r = np.sqrt((x - x_uav)**2 + (y - y_uav)**2)

    # LOS probability and shadowing parameters
    theta = np.tanh(cfg.H / (r + 1e-6))
    P_los = 1 / (1 + (cfg.a_0 * np.exp(-cfg.b_0 * theta)))
    sigma_los = cfg.a_los * np.exp(-cfg.b_los * theta)
    sigma_nlos = cfg.a_nlos * np.exp(-cfg.b_nlos * theta)

    # For simplicity, use a fixed sigma for log-normal perturbations
    sigma = 4.0

    # Path-loss calculations (dB)
    PL = 20 * np.log10(d) + 20 * np.log10((4 * np.pi * cfg.f) / cfg.c) + np.random.normal(0, sigma)
    p_r = cfg.p_t_dbm - PL

    # Echo/reflection
    p_echo = p_r * cfg.reflection_coefficient
    PL_echo = 20 * np.log10(d) + 20 * np.log10((4 * np.pi * cfg.f) / cfg.c) + np.random.normal(0, sigma)
    p_r_uav = p_echo - PL_echo

    # Inverse path-loss to distance estimate bounds
    log1, log2 = 14.0, -14.0
    d_est1 = 10 ** ((cfg.p_t_dbm - p_r_uav - (40 * np.log10((4 * np.pi * cfg.f) / cfg.c)) - log2) / 40)
    d_est2 = 10 ** ((cfg.p_t_dbm - p_r_uav - (40 * np.log10((4 * np.pi * cfg.f) / cfg.c)) - log1) / 40)

    # Enforce minimum based on UAV altitude
    d_est1 = max(d_est1, cfg.H)
    d_est2 = max(d_est2, cfg.H)

    # Convert 3D to horizontal distances
    d_est1 = np.sqrt(d_est1**2 - cfg.H**2)
    d_est2 = np.sqrt(d_est2**2 - cfg.H**2)

    return d_est1, d_est2, r
