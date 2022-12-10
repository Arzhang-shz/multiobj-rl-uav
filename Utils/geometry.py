"""
Geometry utilities for the EUAV environment.
"""
import numpy as np

def generate_points(x, y, r):
    """
    Generate a circular grid of candidate points around (x,y).

    Args:
        x, y: Center coordinates
        r:    Radius (m)

    Returns:
        M: Array of shape [2, len(theta_vec), len(a_vec)] with X and Y coords
    """
    theta_vec = np.arange(0, 360, 10)
    size_matrix = 100
    step = 1.0 / (size_matrix / (r + 1e-6))
    a_vec = np.arange(0, r, step)

    x_co = np.zeros((len(theta_vec), len(a_vec)))
    y_co = np.zeros((len(theta_vec), len(a_vec)))

    for i, a in enumerate(a_vec):
        for j, theta in enumerate(theta_vec):
            x_co[j, i] = np.round(x + a * np.cos(np.deg2rad(theta)))
            y_co[j, i] = np.round(y + a * np.sin(np.deg2rad(theta)))

    return np.array([x_co, y_co])
