import numpy as np
from scipy import ndimage

def generate_field2D(ground_truth_map: np.ndarray, cell_size: float) -> np.ndarray:
    """
    Compute a 2D signed distance field from a binary occupancy map.
    
    Parameters
    ----------
    ground_truth_map : np.ndarray
        2D array with values in [0,1], where 0 = free space, 1 = obstacles.
    cell_size : float
        Size of each grid cell (metric units per cell).
    
    Returns
    -------
    field : np.ndarray
        2D array of same shape as ground_truth_map, where each entry is the
        signed distance to the nearest obstacle (positive in free space,
        negative inside obstacles), scaled by cell_size.
        Infinite distances (if map is empty) are clamped to ±1000.
    """
    # Binarize map: treat values >0.75 as obstacles
    obstacle_map = (ground_truth_map > 0.75).astype(int)
    
    # If there are no obstacles at all, return a large constant field
    if np.max(obstacle_map) == 0:
        return np.ones_like(ground_truth_map, dtype=float) * 1000.0
    
    # Invert map: 1 where free, 0 where obstacle
    free_map = 1 - obstacle_map
    
    # Distance from each free cell to nearest obstacle
    dist_to_obstacle = ndimage.distance_transform_edt(free_map)
    # Distance from each obstacle cell to nearest free cell
    dist_to_free     = ndimage.distance_transform_edt(obstacle_map)
    
    # Signed distance: positive outside, negative inside
    field = dist_to_obstacle - dist_to_free
    
    # Scale to metric units
    field = field * cell_size
    
    # Convert to float and handle infinite values
    field = field.astype(float)
    if np.isinf(field[0, 0]):
        field = np.ones_like(field) * 1000.0
    
    return field
