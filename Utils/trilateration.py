"""
Trilateration utilities for the EUAV environment.
"""
import numpy as np

def trilateration1(M, x1, y1, r1_low, r1_high, x2, y2, r2_low, r2_high, x_old, y_old):
    """
    Two-stage trilateration using point cloud M for initial estimate.

    Args:
        M:       Array [2, N, M] of candidate points
        x1, y1:  First reference point
        r1_low, r1_high: Lower/upper radius from first point
        x2, y2:  Second reference point
        r2_low, r2_high: Lower/upper radius from second point
        x_old, y_old: Previous estimate

    Returns:
        common_point: Filtered points array
        estimate_position: Selected estimate
        local_error: Max error among common points
        stop: Always zero here
    """
    common = []
    stop = 0
    # iterate over grid
    for i in range(M.shape[1]):
        for j in range(M.shape[2]):
            px, py = M[0, i, j], M[1, i, j]
            d1 = (px - x1)**2 + (py - y1)**2
            d2 = (px - x2)**2 + (py - y2)**2
            if r1_low**2 < d1 < r1_high**2 and r2_low**2 < d2 < r2_high**2:
                common.append([px, py])
    if len(common) == 0:
        # fallback to previous estimate
        common = np.vstack((M[0].ravel(), M[1].ravel())).T
        estimate_position = (x_old, y_old)
        dists = np.linalg.norm(common - np.array([x_old, y_old]), axis=1)
        local_error = int(dists.max())
    else:
        common = np.array(common)
        centroid = common.mean(axis=0)
        dists = np.linalg.norm(common - centroid, axis=1)
        idx = np.argmin(dists)
        estimate_position = tuple(common[idx])
        local_error = int(np.max(np.linalg.norm(common - estimate_position, axis=1)))
    return common, estimate_position, local_error, stop


def trilateration2(points, x1, y1, r_low, r_high, x_old, y_old):
    """
    Second-stage trilateration with flattened point list.

    Args:
        points:   Array [K,2] of candidate points
        x1, y1:   Reference point
        r_low, r_high: Lower/upper radius bounds
        x_old, y_old: Previous estimate

    Returns:
        common: Filtered points array
        estimate_position: Selected estimate
        local_error: Max error among common points
        stop: Always zero here
    """
    common = []
    stop = 0
    for px, py in points:
        d = (px - x1)**2 + (py - y1)**2
        if r_low**2 < d < r_high**2:
            common.append([px, py])
    if len(common) == 0:
        common = np.array(points)
        estimate_position = (x_old, y_old)
        dists = np.linalg.norm(common - np.array([x_old, y_old]), axis=1)
        local_error = int(dists.max())
    else:
        common = np.array(common)
        centroid = common.mean(axis=0)
        dists_centroid = np.linalg.norm(common - centroid, axis=1)
        idx = np.argmin(dists_centroid)
        estimate_position = tuple(common[idx])
        local_error = int(np.max(np.linalg.norm(common - estimate_position, axis=1)))
    return common, estimate_position, local_error, stop
