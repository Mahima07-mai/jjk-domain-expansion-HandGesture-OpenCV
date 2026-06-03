import cv2
import numpy as np

def build_radial_remap(shape, center, strength):
    h, w = shape[:2]
    cx, cy = center
    y, x = np.mgrid[0:h, 0:w].astype(np.float32)
    
    dx = x - cx
    dy = y - cy
    r = np.sqrt(dx**2 + dy**2)
    
    # Warp factor
    warp = strength * np.exp(-r / (h * 0.4))
    map_x = x + dx * warp
    map_y = y + dy * warp
    
    return map_x, map_y

def build_wave_remap(shape, amplitude, frequency, t):
    h, w = shape[:2]
    y, x = np.mgrid[0:h, 0:w].astype(np.float32)
    
    map_x = x + amplitude * np.sin(y * frequency + t * 4)
    map_y = y # Vertical is stable for simple shimmer
    
    return map_x, map_y

def build_mesh_remap(shape, grid_size, offsets):
    """offsets: (rows, cols, 2) array of displacements"""
    h, w = shape[:2]
    rows, cols = grid_size
    
    # Target grid points
    src_pts = np.mgrid[0:h:complex(rows), 0:w:complex(cols)].reshape(2, -1).T
    src_pts = src_pts[:, [1, 0]].astype(np.float32) # (x, y)
    
    dst_pts = src_pts + offsets.reshape(-1, 2)
    
    # This is a bit slow to do per-frame if grid is large, but fine for 5x5
    # For speed, we"d use cv2.remap with pre-interpolated maps
    # But here we provide the logic to build the maps
    # In practice we"ll use a faster vectorized oscillate in effects
    return src_pts, dst_pts
