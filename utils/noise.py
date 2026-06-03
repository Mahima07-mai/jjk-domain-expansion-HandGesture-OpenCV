import numpy as np

def perlin2d(width, height, scale=10.0, octaves=1, seed=None):
    if seed is not None:
        np.random.seed(seed)

    def f(t):
        return t * t * t * (t * (t * 6 - 15) + 10)

    def lerp(a, b, x):
        return a + x * (b - a)

    def gradient(h, x, y):
        vectors = np.array([[0, 1], [0, -1], [1, 0], [-1, 0]])
        g = vectors[h % 4]
        return g[..., 0] * x + g[..., 1] * y

    lin_x = np.linspace(0, scale, width, endpoint=False)
    lin_y = np.linspace(0, scale, height, endpoint=False)
    x, y = np.meshgrid(lin_x, lin_y)

    p = np.arange(256, dtype=int)
    np.random.shuffle(p)
    p = np.stack([p, p]).flatten()

    xi = x.astype(int)
    yi = y.astype(int)
    xf = x - xi
    yf = y - yi

    u = f(xf)
    v = f(yf)

    n00 = p[p[xi % 256] + yi % 256]
    n01 = p[p[xi % 256] + (yi + 1) % 256]
    n11 = p[p[(xi + 1) % 256] + (yi + 1) % 256]
    n10 = p[p[(xi + 1) % 256] + yi % 256]

    x1 = lerp(gradient(n00, xf, yf), gradient(n10, xf - 1, yf), u)
    x2 = lerp(gradient(n01, xf, yf - 1), gradient(n11, xf - 1, yf - 1), u)
    
    res = lerp(x1, x2, v)
    return (res - res.min()) / (res.max() - res.min() + 1e-6)

def simplex2d(width, height, offset_x=0.0, offset_y=0.0):
    # Simplified noise for high-speed remapping
    y, x = np.mgrid[0:height, 0:width]
    x = (x.astype(float) + offset_x) * 0.05
    y = (y.astype(float) + offset_y) * 0.05
    
    # Use a combination of sines for a simplex-like look
    noise = np.sin(x) * np.cos(y) + np.sin(x * 0.5 + y * 1.2) * 0.5
    return (noise - noise.min()) / (noise.max() - noise.min() + 1e-6)
