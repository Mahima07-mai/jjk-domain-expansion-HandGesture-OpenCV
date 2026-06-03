import cv2
import numpy as np
from pathlib import Path


def add_bloom(img, threshold=80, blur_size=51, strength=0.9):
    """Lens bloom: only bright pixels bleed light outward."""
    f = img.astype(np.float32)
    bright = np.clip(f - threshold, 0, 255) * (255 / (255 - threshold + 1e-6))
    blurred = cv2.GaussianBlur(bright.astype(np.uint8), (blur_size, blur_size), 0)
    result = np.clip(f + blurred.astype(np.float32) * strength, 0, 255)
    return result.astype(np.uint8)


class HollowPurple:
    MERGE_T    = 1.20
    FLASH_DUR  = 0.22
    WAVE_START = 1.42
    WAVE_DUR   = 1.80
    FADE_START = 3.10
    END_T      = 3.70

    def __init__(self, frame_shape, bg_image=None):
        self.h, self.w = frame_shape[:2]
        self.active = False
        self.t = 0.0
        self.cx, self.cy = self.w // 2, self.h // 2
        self.bg = bg_image
        rng = np.random.default_rng(7)

        # ── PARTICLE STREAMS (trail behind each orb) ─────────────────────
        self.N_stream = 12000
        self.sr_frac  = rng.power(1.5, self.N_stream).astype(np.float32)
        self.sr_offy  = rng.normal(0, 5, self.N_stream).astype(np.float32)
        self.sr_bri   = rng.uniform(100, 255, self.N_stream).astype(np.float32)
        self.sr_phase = rng.uniform(0, 1, self.N_stream).astype(np.float32)

        self.sb_frac  = rng.power(1.5, self.N_stream).astype(np.float32)
        self.sb_offy  = rng.normal(0, 5, self.N_stream).astype(np.float32)
        self.sb_bri   = rng.uniform(100, 255, self.N_stream).astype(np.float32)
        self.sb_phase = rng.uniform(0, 1, self.N_stream).astype(np.float32)

        # ── ENERGY TENDRILS ───────────────────────────────────────────────
        self.N_tend = 80
        self.tend_a = rng.uniform(0, 2*np.pi, self.N_tend).astype(np.float32)
        self.tend_len = rng.uniform(0.5, 1.0, self.N_tend).astype(np.float32)
        self.tend_wb = rng.uniform(0, 2*np.pi, self.N_tend).astype(np.float32)
        self.tend_fr = rng.uniform(10, 25, self.N_tend).astype(np.float32)

        # ── DEBRIS: 30k crisp 1px round dots ────────────────────────────
        self.N_db  = 30000
        db_a       = rng.uniform(0, 2*np.pi, self.N_db)
        db_r_norm  = np.clip(rng.exponential(0.20, self.N_db), 0.005, 1.2)
        db_r       = db_r_norm * min(self.w, self.h) * 0.62
        self.db_tx    = (self.cx + db_r * np.cos(db_a)).astype(np.float32)
        self.db_ty    = (self.cy + db_r * np.sin(db_a)).astype(np.float32)
        self.db_speed = rng.uniform(0.10, 1.0, self.N_db).astype(np.float32)
        self.db_bri   = rng.uniform(80, 255, self.N_db).astype(np.float32)
        roll = rng.random(self.N_db)
        self.db_col = np.where(roll < 0.28, 0,
                      np.where(roll < 0.56, 1,
                      np.where(roll < 0.80, 2, 3))).astype(np.int32)

        # ── STARS ────────────────────────────────────────────────────────
        self.N_st = 350
        self.st_x = rng.uniform(0, self.w, self.N_st).astype(np.float32)
        self.st_y = rng.uniform(0, self.h, self.N_st).astype(np.float32)
        self.st_b = rng.uniform(15, 80, self.N_st).astype(np.float32)
        self.st_p = rng.uniform(0, 2*np.pi, self.N_st).astype(np.float32)
        self.st_f = rng.uniform(0.3, 1.2, self.N_st).astype(np.float32)

    def activate(self):
        self.active = True
        self.t = 0.0

    def _orb_positions(self):
        START = 0.10
        prog = float(np.clip((self.t - START) / (self.MERGE_T - START), 0.0, 1.0))
        ease = prog ** 2.2
        rx = self.w * 0.87 + (self.cx - self.w * 0.87) * ease
        bx = self.w * 0.13 + (self.cx - self.w * 0.13) * ease
        return float(rx), float(self.cy), float(bx), float(self.cy), prog

    def _draw_stars(self, canvas):
        tw = 0.5 + 0.5 * np.sin(self.t * self.st_f + self.st_p)
        b  = np.clip(self.st_b * tw, 0, 255).astype(np.uint8)
        xs = self.st_x.astype(np.int32)
        ys = self.st_y.astype(np.int32)
        # vectorized: stack into Nx3 and use np.maximum on indexed pixels
        bv = b[:, np.newaxis].repeat(3, axis=1)  # (N, 3) grey
        np.maximum.at(canvas, (ys, xs), bv)

    def _draw_orb(self, canvas, cx, cy, color, radius, alpha):
        """Distinct red OR blue orb — no purple bleed between them."""
        if alpha <= 0:
            return
        cx, cy = int(cx), int(cy)
        pulse = 0.92 + 0.08 * np.sin(self.t * 22.0)
        r = max(1, int(radius * pulse))

        layer = np.zeros((self.h, self.w, 3), dtype=np.float32)

        if color == 'red':
            # pure red (BGR = 0,0,255), no blue component at all
            layers_def = [
                (r * 6,  (  0,  0, 60)),   # very faint dark red halo
                (r * 3,  (  0,  0,140)),   # mid red glow
                (r * 1,  (  0, 15,230)),   # bright red
                (r // 2, (  0, 40,255)),   # core
                (r // 5, (140,160,255)),   # white-hot center (neutral white)
            ]
            blur_col = (0, 0, 120)
        else:
            # pure blue (BGR = 255,0,0), no red component
            layers_def = [
                (r * 6,  ( 60,  0,  0)),
                (r * 3,  (140,  0,  0)),
                (r * 1,  (230, 15,  0)),
                (r // 2, (255, 40,  0)),
                (r // 5, (255,160,140)),
            ]
            blur_col = (120, 0, 0)

        for rad, col in layers_def:
            cv2.circle(layer, (cx, cy), max(1, rad),
                       [float(c) for c in col], -1)

        # Halo blur
        halo = layer.copy()
        halo = cv2.GaussianBlur(halo, (61, 61), 0)

        # Sharp core
        core = np.zeros_like(layer)
        cv2.circle(core, (cx, cy), max(1, r), [float(c) for c in layers_def[2][1]], -1)
        core = cv2.GaussianBlur(core, (21, 21), 0)

        combined = np.clip(halo * 0.8 + core * 1.2, 0, 255).astype(np.uint8)
        canvas[:] = np.clip(
            canvas.astype(np.float32) + combined.astype(np.float32) * alpha,
            0, 255).astype(np.uint8)

    def _draw_stream(self, canvas, ox, oy, prog, color):
        """1px particle trail — sharp, not blurry."""
        if prog <= 0:
            return
        trail_len = self.w * 0.44 * prog
        sign = 1.0 if color == 'red' else -1.0
        frac = self.sr_frac if color == 'red' else self.sb_frac
        offy = self.sr_offy if color == 'red' else self.sb_offy
        bri  = self.sr_bri  if color == 'red' else self.sb_bri
        phase = self.sr_phase if color == 'red' else self.sb_phase

        # Animate flow
        anim = (phase + self.t * 1.2) % 1.0

        # --- vectorized (was per-particle for loop) ---
        dist = frac * trail_len * (0.85 + 0.15 * anim)
        tx = (ox + sign * dist).astype(np.int32)
        ty = (oy + offy).astype(np.int32)

        # bounds mask
        mask = (tx >= 0) & (tx < self.w) & (ty >= 0) & (ty < self.h)

        fade = (1.0 - frac ** 0.6) * prog
        b_f  = bri * fade                         # float brightness
        mask &= (b_f >= 15)

        tx, ty, b_f = tx[mask], ty[mask], b_f[mask]
        b_u8 = np.clip(b_f, 0, 255).astype(np.uint8)

        if color == 'red':
            cols = np.stack([
                np.zeros(len(b_u8), dtype=np.uint8),
                np.clip((b_f * 0.05), 0, 255).astype(np.uint8),
                b_u8,
            ], axis=1)
        else:
            cols = np.stack([
                b_u8,
                np.clip((b_f * 0.05), 0, 255).astype(np.uint8),
                np.zeros(len(b_u8), dtype=np.uint8),
            ], axis=1)

        np.maximum.at(canvas, (ty, tx), cols)

    def _draw_tendrils(self, canvas, cx, cy, color, prog, orb_r):
        if prog <= 0.05:
            return
        cx, cy = int(cx), int(cy)
        for i in range(self.N_tend):
            wb = np.sin(self.t * self.tend_fr[i] + self.tend_wb[i]) * 18
            angle = self.tend_a[i] + wb * 0.008
            length = orb_r * (2.0 + prog * 4.0) * self.tend_len[i]
            x0 = int(cx + np.cos(angle) * orb_r * 0.9)
            y0 = int(cy + np.sin(angle) * orb_r * 0.9)
            x1 = int(cx + np.cos(angle) * (orb_r + length))
            y1 = int(cy + np.sin(angle) * (orb_r + length))
            fade = self.tend_len[i] * prog * 0.8
            b = int(200 * fade)
            if b < 12:
                continue
            col = (0, int(b*0.08), b) if color == 'red' else (b, int(b*0.08), 0)
            if (0 <= x0 < self.w and 0 <= y0 < self.h and
                    0 <= x1 < self.w and 0 <= y1 < self.h):
                cv2.line(canvas, (x0, y0), (x1, y1), col, 1)

    def _draw_merge_flash(self, canvas, fp):
        if fp <= 0:
            return
        fade = 1.0 - fp
        layer = np.zeros((self.h, self.w, 3), dtype=np.float32)
        for r, col in [
            (int(self.w * 0.72 * fp), (int(50*fade), 0, int(50*fade))),
            (int(self.w * 0.42 * fp), (int(110*fade), 0, int(110*fade))),
            (int(self.w * 0.20 * fp), (int(190*fade), 30, int(190*fade))),
            (int(self.w * 0.08),      (255, 60, 255)),
            (int(self.w * 0.035),     (255, 180, 255)),
            (int(self.w * 0.012),     (255, 255, 255)),
        ]:
            if r > 0:
                cv2.circle(layer, (self.cx, self.cy), r, [float(c) for c in col], -1)
        blurred = cv2.GaussianBlur(layer.astype(np.uint8), (91, 91), 0)
        canvas[:] = np.clip(
            canvas.astype(np.float32) + blurred.astype(np.float32) * min(2.0, fade * 3.5),
            0, 255).astype(np.uint8)

    def _draw_shockwave(self, canvas, wp, alpha):
        if alpha <= 0 or wp <= 0:
            return

        # Sharp glowing ring
        ring_r = int(min(self.w, self.h) * 0.74 * wp)
        if ring_r > 2:
            rl = np.zeros((self.h, self.w, 3), dtype=np.uint8)
            fade = (1.0 - wp * 0.88) * alpha
            thick_outer = max(2, int(38 * (1 - wp * 0.65)))
            thick_inner = max(1, int(7  * (1 - wp * 0.50)))
            # Outer soft glow
            cv2.circle(rl, (self.cx, self.cy), ring_r,
                       (int(80*fade), 0, int(80*fade)), thick_outer)
            # Bright sharp edge
            cv2.circle(rl, (self.cx, self.cy), ring_r,
                       (int(255*fade), 8, int(255*fade)), thick_inner)
            rl = cv2.GaussianBlur(rl, (13, 13), 0)
            rl = add_bloom(rl, threshold=50, blur_size=29, strength=1.0)
            canvas[:] = np.clip(canvas.astype(np.float32) + rl.astype(np.float32), 0, 255).astype(np.uint8)

        # Central purple glow
        cf = max(0.0, 1.0 - wp * 1.15) * alpha
        if cf > 0.02:
            gl = np.zeros((self.h, self.w, 3), dtype=np.uint8)
            cv2.circle(gl, (self.cx, self.cy),
                       max(1, int(min(self.w, self.h) * 0.24)),
                       (int(150*cf), 0, int(150*cf)), -1)
            gl = cv2.GaussianBlur(gl, (75, 75), 0)
            canvas[:] = np.clip(canvas.astype(np.float32) + gl.astype(np.float32), 0, 255).astype(np.uint8)

        # ── DEBRIS: crisp 1px round particles ──────────────────────────
        ease   = np.clip(wp * self.db_speed, 0.0, 1.0)
        fade_d = alpha * (1.0 - ease * 0.80)
        px = (self.cx + (self.db_tx - self.cx) * ease).astype(np.int32)
        py = (self.cy + (self.db_ty - self.cy) * ease).astype(np.int32)
        bris = np.clip(self.db_bri * fade_d, 0, 255).astype(np.uint8)

        # --- vectorized (was per-particle for loop) ---
        mask = (px >= 0) & (px < self.w) & (py >= 0) & (py < self.h)
        mask &= (bris >= 12)

        px_m   = px[mask]
        py_m   = py[mask]
        b_m    = bris[mask].astype(np.int32)
        col_m  = self.db_col[mask]

        # Build BGR colour array for all surviving particles at once
        bgr = np.zeros((len(b_m), 3), dtype=np.uint8)
        b_f = b_m.astype(np.float32)

        c0 = col_m == 0  # red:    (0, b*0.04, b)
        c1 = col_m == 1  # blue:   (b, b*0.04, 0)
        c2 = col_m == 2  # purple: (b, 0, b)
        c3 = col_m == 3  # white:  (b, b, b)

        bgr[c0, 0] = 0
        bgr[c0, 1] = np.clip(b_f[c0] * 0.04, 0, 255).astype(np.uint8)
        bgr[c0, 2] = np.clip(b_f[c0],         0, 255).astype(np.uint8)

        bgr[c1, 0] = np.clip(b_f[c1],         0, 255).astype(np.uint8)
        bgr[c1, 1] = np.clip(b_f[c1] * 0.04, 0, 255).astype(np.uint8)
        bgr[c1, 2] = 0

        bgr[c2, 0] = np.clip(b_f[c2], 0, 255).astype(np.uint8)
        bgr[c2, 1] = 0
        bgr[c2, 2] = np.clip(b_f[c2], 0, 255).astype(np.uint8)

        bgr[c3, 0] = np.clip(b_f[c3], 0, 255).astype(np.uint8)
        bgr[c3, 1] = np.clip(b_f[c3], 0, 255).astype(np.uint8)
        bgr[c3, 2] = np.clip(b_f[c3], 0, 255).astype(np.uint8)

        # All 1px — round dot (same as before, just vectorized)
        np.maximum.at(canvas, (py_m, px_m), bgr)

    def update(self, frame, dt):
        self.t += dt
        if self.t > self.END_T:
            self.active = False
            return frame

        # Background nebula
        if self.bg is not None:
            bg_a = float(np.clip(self.t / 0.12, 0, 1))
            out = (self.bg.astype(np.float32) * bg_a * 0.50).astype(np.uint8)
        else:
            out = np.zeros((self.h, self.w, 3), dtype=np.uint8)

        self._draw_stars(out)

        rx, ry, bx, by, orb_prog = self._orb_positions()
        orb_r = int(min(self.w, self.h) * 0.048)
        orb_alpha = float(np.clip((self.t - 0.10) / 0.15, 0, 1))

        if self.t < self.MERGE_T + self.FLASH_DUR:
            self._draw_stream(out, rx, ry, orb_prog, 'red')
            self._draw_stream(out, bx, by, orb_prog, 'blue')
            self._draw_tendrils(out, rx, ry, 'red',  orb_prog, orb_r)
            self._draw_tendrils(out, bx, by, 'blue', orb_prog, orb_r)
            self._draw_orb(out, rx, ry, 'red',  orb_r, orb_alpha)
            self._draw_orb(out, bx, by, 'blue', orb_r, orb_alpha)

        if self.t >= self.MERGE_T:
            fp = float(np.clip((self.t - self.MERGE_T) / self.FLASH_DUR, 0, 1))
            self._draw_merge_flash(out, fp)

        if self.t >= self.WAVE_START:
            wp    = float(np.clip((self.t - self.WAVE_START) / self.WAVE_DUR, 0, 1))
            walph = float(1.0 - np.clip((self.t - self.FADE_START) / 0.6, 0, 1))
            self._draw_shockwave(out, wp, walph)

        # Global bloom — makes energy look lit
        if self.t < self.FADE_START:
            out = add_bloom(out, threshold=85, blur_size=39, strength=0.55)

        # Fade to black
        fade = float(np.clip((self.t - self.FADE_START) / 0.6, 0, 1))
        if fade > 0:
            out = np.clip(out.astype(np.float32) * (1 - fade), 0, 255).astype(np.uint8)

        return out

    @property
    def is_active(self):
        return self.active


def generate_space_bg(W, H, seed=99):
    """Generate a deep space nebula background — dark blue-indigo with stars."""
    rng = np.random.default_rng(seed)
    bg = np.zeros((H, W, 3), dtype=np.float32)

    # Dark nebula clouds: subtle blue-indigo ellipses
    for _ in range(22):
        cx = int(rng.integers(-100, W + 100))
        cy = int(rng.integers(-100, H + 100))
        rx = int(rng.integers(60, 380))
        ry = int(rng.integers(50, 250))
        bri = rng.uniform(3, 14)
        angle = int(rng.uniform(0, 180))
        col = (bri * rng.uniform(0.5, 1.0),
               bri * rng.uniform(0.0, 0.08),
               bri * rng.uniform(0.05, 0.35))
        layer = np.zeros((H, W, 3), dtype=np.float32)
        cv2.ellipse(layer, (cx, cy), (rx, ry), angle, 0, 360, col, -1)
        ksize = int(rng.integers(80, 180)) | 1
        layer = cv2.GaussianBlur(layer, (ksize, ksize), 0)
        bg += layer

    # Dense star cluster patch (right-center)
    cluster_cx, cluster_cy = int(W * 0.65), int(H * 0.50)
    for _ in range(12):
        cx = cluster_cx + int(rng.normal(0, 80))
        cy = cluster_cy + int(rng.normal(0, 60))
        r  = int(rng.integers(15, 70))
        bri = rng.uniform(4, 16)
        layer = np.zeros((H, W, 3), dtype=np.float32)
        cv2.circle(layer, (cx, cy), r, (bri, bri * 0.05, bri * 0.1), -1)
        ksize = int(rng.integers(31, 91)) | 1
        layer = cv2.GaussianBlur(layer, (ksize, ksize), 0)
        bg += layer

    bg = np.clip(bg, 0, 60).astype(np.uint8)

    # 1px stars: mostly dim, exponential brightness distribution
    N = 1200
    sx = rng.integers(0, W, N)
    sy = rng.integers(0, H, N)
    sb = np.clip(rng.exponential(12, N), 2, 255)
    for i in range(N):
        x, y = int(sx[i]), int(sy[i])
        b = float(sb[i])
        col = (min(255, int(b)), min(255, int(b * 0.88)), min(255, int(b)))
        bg[y, x] = np.maximum(bg[y, x], col)

    # Bright stars: white core + cross flare + soft glow
    for _ in range(25):
        bx = int(rng.integers(20, W - 20))
        by = int(rng.integers(20, H - 20))
        b  = int(rng.integers(120, 255))
        bg[by, bx] = (min(255, b), min(255, b), 255)
        if b > 160:
            for d in range(1, 4):
                fade = int(b * (0.5 ** d))
                for dx, dy in [(d,0),(-d,0),(0,d),(0,-d)]:
                    nx, ny = bx+dx, by+dy
                    if 0 <= nx < W and 0 <= ny < H:
                        bg[ny, nx] = np.maximum(bg[ny, nx], (fade, fade, fade))
        if b > 140:
            glow = np.zeros((H, W, 3), dtype=np.uint8)
            cv2.circle(glow, (bx, by), 6, (b // 8, b // 8, b // 6), -1)
            glow = cv2.GaussianBlur(glow, (17, 17), 0)
            bg = np.clip(bg.astype(np.float32) + glow.astype(np.float32), 0, 255).astype(np.uint8)

    return bg


if __name__ == "__main__":
    import os
    W, H, FPS = 1280, 720, 60

    print("[BG] Generating deep space nebula...")
    bg = generate_space_bg(W, H)

    hp = HollowPurple((H, W, 3), bg_image=bg)
    hp.activate()

    os.makedirs("/mnt/user-data/outputs", exist_ok=True)
    out_path = "/mnt/user-data/outputs/hollow_purple_UPDATED.mp4"
    writer = cv2.VideoWriter(out_path, cv2.VideoWriter_fourcc(*'mp4v'), FPS, (W, H))

    i = 0
    dt = 1.0 / FPS
    while hp.is_active:
        rendered = hp.update(np.zeros((H, W, 3), dtype=np.uint8), dt)
        writer.write(rendered)
        if i in [55, 78, 100, 145, 175]:
            cv2.imwrite(f'/home/claude/final_frame_{i:03d}.png', rendered)
        i += 1
        if i % 60 == 0:
            print(f"  t={hp.t:.2f}s  frame={i}")

    writer.release()
    print(f"\n[DONE] → {out_path}  ({i} frames @ {FPS}fps)")