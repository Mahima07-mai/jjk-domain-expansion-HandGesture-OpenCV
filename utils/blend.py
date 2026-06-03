import cv2
import numpy as np

def alpha_blend(base, overlay, alpha):
    """Blends two images with a constant alpha or alpha map."""
    if isinstance(alpha, (float, int)):
        return cv2.addWeighted(base, 1 - alpha, overlay, alpha, 0)
    else:
        # alpha is a mask (0-1)
        if len(alpha.shape) == 2:
            alpha = cv2.merge([alpha, alpha, alpha])
        return (base * (1 - alpha) + overlay * alpha).astype(np.uint8)

def overlay_mask(base, color_bgr, mask, alpha=1.0):
    """Overlays a solid color using a mask."""
    overlay = np.full(base.shape, color_bgr, dtype=np.uint8)
    return alpha_blend(base, overlay, mask * alpha)

def radial_gradient(shape, center, r_inner, r_outer, color_inner, color_outer):
    """Creates a radial gradient mask/image."""
    h, w = shape[:2]
    y, x = np.ogrid[:h, :w]
    dist = np.sqrt((x - center[0])**2 + (y - center[1])**2)
    
    mask = np.clip((dist - r_inner) / (r_outer - r_inner + 1e-6), 0, 1)
    mask = 1 - mask # 1 at center, 0 at edge
    
    # Map colors
    c_inner = np.array(color_inner)
    c_outer = np.array(color_outer)
    
    res = np.zeros((h, w, 3), dtype=np.uint8)
    for i in range(3):
        res[..., i] = c_outer[i] + (c_inner[i] - c_outer[i]) * mask
        
    return res, mask
