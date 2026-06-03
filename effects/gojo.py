import cv2
import numpy as np
import random

class UnlimitedVoid:
    def __init__(self, frame_shape):
        self.h, self.w = frame_shape[:2]
        self.active = False
        self.t = 0.0
        self.cx, self.cy = self.w // 2, self.h // 2
        self._base_ring_r = int(min(self.w, self.h) * 0.36)
        self.ring_r = self._base_ring_r

        rng = np.random.default_rng(42)

        # ── BACKGROUND: sparse cluster in upper-right, few scattered elsewhere ─
        # Main cluster — upper right quadrant
        cluster_cx = int(self.w * 0.72)
        cluster_cy = int(self.h * 0.28)
        self.num_cluster = 180
        cluster_r  = rng.exponential(40, self.num_cluster).astype(np.float32)
        cluster_a  = rng.uniform(0, 2*np.pi, self.num_cluster)
        self.bg_cx = np.clip(cluster_cx + cluster_r * np.cos(cluster_a), 0, self.w-1).astype(np.float32)
        self.bg_cy = np.clip(cluster_cy + cluster_r * np.sin(cluster_a), 0, self.h-1).astype(np.float32)
        self.bg_cr = rng.choice([1,1,1,2,2,3], size=self.num_cluster)
        self.bg_cb = rng.uniform(30, 160, self.num_cluster).astype(np.float32)
        self.bg_cp = rng.uniform(0, 2*np.pi, self.num_cluster).astype(np.float32)
        self.bg_ct = rng.uniform(0.3, 1.2,  self.num_cluster).astype(np.float32)
        # Blue tint fraction
        self.bg_cblue = (rng.random(self.num_cluster) < 0.6).astype(np.float32)

        # Secondary small cluster — upper left area
        cluster2_cx = int(self.w * 0.20)
        cluster2_cy = int(self.h * 0.18)
        self.num_cluster2 = 60
        c2r = rng.exponential(25, self.num_cluster2).astype(np.float32)
        c2a = rng.uniform(0, 2*np.pi, self.num_cluster2)
        self.bg2_cx = np.clip(cluster2_cx + c2r * np.cos(c2a), 0, self.w-1).astype(np.float32)
        self.bg2_cy = np.clip(cluster2_cy + c2r * np.sin(c2a), 0, self.h-1).astype(np.float32)
        self.bg2_cr = rng.choice([1,1,2], size=self.num_cluster2)
        self.bg2_cb = rng.uniform(20, 100, self.num_cluster2).astype(np.float32)
        self.bg2_cp = rng.uniform(0, 2*np.pi, self.num_cluster2).astype(np.float32)
        self.bg2_ct = rng.uniform(0.3, 1.0,  self.num_cluster2).astype(np.float32)

        # Very few completely random isolated stars elsewhere
        self.num_iso = 40
        self.iso_x = rng.uniform(0, self.w, self.num_iso).astype(np.float32)
        self.iso_y = rng.uniform(0, self.h, self.num_iso).astype(np.float32)
        self.iso_b = rng.uniform(20, 80, self.num_iso).astype(np.float32)
        self.iso_p = rng.uniform(0, 2*np.pi, self.num_iso).astype(np.float32)
        self.iso_t = rng.uniform(0.2, 0.8,  self.num_iso).astype(np.float32)

        # ── PARTICLES ─────────────────────────────────────────────────────────
        self.N = 4000
        p = np.random.default_rng(7)

        # Each particle's fixed angle slot on the ring
        self.ring_angle  = p.uniform(0, 2*np.pi, self.N).astype(np.float32)
        self.ring_jitter = p.normal(0, 2.5, self.N).astype(np.float32)

        # Blast target: random scatter across screen
        blast_a = p.uniform(0, 2*np.pi, self.N)
        blast_r = p.uniform(0.05, 1.3, self.N) * min(self.w, self.h) * 0.5
        self.blast_x = (self.cx + blast_r * np.cos(blast_a)).astype(np.float32)
        self.blast_y = (self.cy + blast_r * np.sin(blast_a)).astype(np.float32)

        self.p_bright = p.uniform(160, 255, self.N).astype(np.float32)
        self.p_size   = p.choice([1,1,1,2,2,3], size=self.N)
        self.p_blue   = (p.random(self.N) < 0.3).astype(np.float32)

        # 3D tilt of the ring plane (~28 degrees)
        self.tilt = np.deg2rad(28)
        self.rev_offset = 0.0

        # ── INNER NEBULA ORBS ─────────────────────────────────────────────────
        orb_rng = np.random.default_rng(99)
        self.num_orbs        = 5
        self.orb_orbit_r     = orb_rng.uniform(0.0, 0.5, self.num_orbs) * self._base_ring_r * 0.55
        self.orb_orbit_speed = orb_rng.uniform(0.08, 0.3, self.num_orbs)
        self.orb_orbit_phase = orb_rng.uniform(0, 2*np.pi, self.num_orbs)
        self.orb_size        = orb_rng.uniform(12, 45, self.num_orbs).astype(int)
        self.orb_pulse_phase = orb_rng.uniform(0, 2*np.pi, self.num_orbs)

    # ─────────────────────────────────────────────────────────────────────────
    def activate(self):
        self.active = True
        self.t = 0.0
        self.rev_offset = 0.0
        self.ring_r = self._base_ring_r

    # ─────────────────────────────────────────────────────────────────────────
    def _draw_background(self, canvas):
        """Sparse deep-space background — clusters, not uniform spread."""
        twinkle1 = 0.5 + 0.5 * np.sin(self.t * self.bg_ct + self.bg_cp)
        b1 = np.clip(self.bg_cb * twinkle1, 0, 255).astype(np.uint8)
        for i in range(self.num_cluster):
            x, y = int(self.bg_cx[i]), int(self.bg_cy[i])
            bv = int(b1[i])
            if self.bg_cblue[i]:
                col = (bv, int(bv*0.7), int(bv*0.3))
            else:
                col = (bv, bv, bv)
            cv2.circle(canvas, (x, y), int(self.bg_cr[i]), col, -1)

        twinkle2 = 0.5 + 0.5 * np.sin(self.t * self.bg2_ct + self.bg2_cp)
        b2 = np.clip(self.bg2_cb * twinkle2, 0, 255).astype(np.uint8)
        for i in range(self.num_cluster2):
            x, y = int(self.bg2_cx[i]), int(self.bg2_cy[i])
            bv = int(b2[i])
            col = (bv, int(bv*0.8), int(bv*0.4))
            cv2.circle(canvas, (x, y), int(self.bg2_cr[i]), col, -1)

        twinkle3 = 0.5 + 0.5 * np.sin(self.t * self.iso_t + self.iso_p)
        b3 = np.clip(self.iso_b * twinkle3, 0, 255).astype(np.uint8)
        for i in range(self.num_iso):
            x, y = int(self.iso_x[i]), int(self.iso_y[i])
            bv = int(b3[i])
            cv2.circle(canvas, (x, y), 1, (bv, bv, bv), -1)

    # ─────────────────────────────────────────────────────────────────────────
    def _draw_nebula(self, canvas, intensity):
        if intensity <= 0:
            return
        glow = np.zeros((self.h, self.w, 3), dtype=np.uint8)
        pulse = 0.85 + 0.15 * np.sin(self.t * 5.0)
        for radius, frac in [(180, 0.08), (90, 0.22), (45, 0.55), (18, 1.0)]:
            r = max(1, int(radius * pulse))
            b = int(180 * intensity * frac)
            g = int(90  * intensity * frac)
            cv2.circle(glow, (self.cx, self.cy), r, (b, g, 0), -1)
        glow = cv2.GaussianBlur(glow, (61, 61), 0)
        cv2.addWeighted(canvas, 1.0, glow, 1.0, 0, dst=canvas)

    # ─────────────────────────────────────────────────────────────────────────
    def _ring_pos(self, angle, r):
        """Convert ring angle to 2D screen position with 3D tilt."""
        px = self.cx + r * np.cos(angle)
        py = self.cy + r * np.sin(angle) * np.sin(self.tilt)
        depth = 0.4 + 0.6 * ((np.sin(angle) + 1.0) * 0.5)  # 0.4 (far) → 1.0 (near)
        return px, py, depth

    # ─────────────────────────────────────────────────────────────────────────
    def _draw_particles(self, canvas, phase, p_alpha):
        """
        phase 0→1 : BLAST  — center → scatter
        phase 1→2 : ARC    — scatter → ring positions
        phase 2+  : REVOLVE on ring (ring_r may shrink during collapse)
        """
        if p_alpha <= 0:
            return

        angles_cur = self.ring_angle + self.rev_offset

        if phase < 1.0:
            ease = 1.0 - (1.0 - phase) ** 3      # ease-out
            px = self.cx + (self.blast_x - self.cx) * ease
            py = self.cy + (self.blast_y - self.cy) * ease
            depth = np.ones(self.N, dtype=np.float32)

        elif phase < 2.0:
            t = (phase - 1.0)                      # 0→1
            ease = t * t * (3.0 - 2.0 * t)         # smoothstep — nice arc feel

            rx, ry, depth = self._ring_pos(
                angles_cur,
                self.ring_r + self.ring_jitter
            )
            px = self.blast_x + (rx - self.blast_x) * ease
            py = self.blast_y + (ry - self.blast_y) * ease

        else:
            # Fully on ring, revolving
            px, py, depth = self._ring_pos(
                angles_cur,
                self.ring_r + self.ring_jitter
            )

        for i in range(self.N):
            x, y = int(px[i]), int(py[i])
            if not (0 <= x < self.w and 0 <= y < self.h):
                continue
            d = float(depth[i]) if hasattr(depth, '__len__') else float(depth)
            bright = int(self.p_bright[i] * d * p_alpha)
            bright = max(0, min(255, bright))
            sz = max(1, int(self.p_size[i] * (0.5 + 0.5 * d)))
            if self.p_blue[i]:
                col = (bright, int(bright*0.85), int(bright*0.3))
            else:
                col = (bright, bright, bright)
            cv2.circle(canvas, (x, y), sz, col, -1)

    # ─────────────────────────────────────────────────────────────────────────
    def _draw_inner_orbs(self, canvas, alpha):
        if alpha <= 0:
            return
        for i in range(self.num_orbs):
            angle = self.t * self.orb_orbit_speed[i] + self.orb_orbit_phase[i]
            ox = int(self.cx + self.orb_orbit_r[i] * np.cos(angle))
            oy = int(self.cy + self.orb_orbit_r[i] * np.sin(angle) * np.sin(self.tilt))
            pulse = 0.8 + 0.2 * np.sin(self.t * 2.5 + self.orb_pulse_phase[i])
            size = max(1, int(self.orb_size[i] * pulse))
            layer = np.zeros((self.h, self.w, 3), dtype=np.uint8)
            cv2.circle(layer, (ox, oy), size,      (int(200*alpha), int(100*alpha), 0), -1)
            cv2.circle(layer, (ox, oy), size//2,   (int(255*alpha), int(160*alpha), 20), -1)
            cv2.circle(layer, (ox, oy), size//4,   (255, 240, 200), -1)
            layer = cv2.GaussianBlur(layer, (21, 21), 0)
            cv2.addWeighted(canvas, 1.0, layer, alpha, 0, dst=canvas)

    # ─────────────────────────────────────────────────────────────────────────
    def update(self, frame, dt):
        self.t += dt
        if self.t > 3.5:
            self.active = False
            return frame

        # Timings
        BLAST_START    = 0.1
        BLAST_DUR      = 0.5    # 0.1 → 0.6s
        ARC_DUR        = 0.8    # 0.6 → 1.4s  (ring fully formed at 1.4s)
        RING_START     = BLAST_START + BLAST_DUR + ARC_DUR   # 3.2s
        REVOLVE_DUR    = 0.8    # 1.4 → 2.2s
        COLLAPSE_START = RING_START + REVOLVE_DUR             # 8.2s
        COLLAPSE_DUR   = 0.8    # 2.2 → 3.0s

        # Revolution — begins at blast, accelerates
        if self.t > BLAST_START:
            rev_speed = 0.9
            self.rev_offset = (self.rev_offset + rev_speed * dt) % (2 * np.pi)

        # Phase for particle positions
        if self.t < BLAST_START:
            phase = 0.0
        elif self.t < BLAST_START + BLAST_DUR:
            phase = (self.t - BLAST_START) / BLAST_DUR              # 0→1
        elif self.t < RING_START:
            phase = 1.0 + (self.t - BLAST_START - BLAST_DUR) / ARC_DUR  # 1→2
        else:
            phase = 2.0

        # Collapse: shrink ring radius toward 0
        collapse_prog = 0.0
        if self.t > COLLAPSE_START:
            collapse_prog = float(np.clip(
                (self.t - COLLAPSE_START) / COLLAPSE_DUR, 0.0, 1.0))
            ease_collapse = collapse_prog ** 2
            self.ring_r = max(1, int(self._base_ring_r * (1.0 - ease_collapse)))

        # ── DRAW ──────────────────────────────────────────────────────────────
        out = np.zeros((self.h, self.w, 3), dtype=np.uint8)

        # 1. Sparse background
        self._draw_background(out)

        # 2. Nebula glow at center (fades once ring forms)
        neb_in  = float(np.clip(self.t / 0.4, 0.0, 1.0))
        neb_out = float(np.clip((self.t - 0.8) / 0.5, 0.0, 0.85))
        self._draw_nebula(out, neb_in * (1.0 - neb_out))

        # 3. Particles
        p_alpha = float(np.clip((self.t - BLAST_START) / 0.4, 0.0, 1.0))
        p_alpha *= float(1.0 - np.clip((self.t - 3.0) / 0.4, 0.0, 1.0))
        self._draw_particles(out, phase, p_alpha)

        # 4. Inner orbs — visible once ring is fully formed
        orb_alpha = float(np.clip((self.t - RING_START) / 0.8, 0.0, 1.0))
        orb_alpha *= (1.0 - collapse_prog)
        self._draw_inner_orbs(out, orb_alpha)

        # 5. Final fade to black
        fade_end = float(np.clip((self.t - 3.0) / 0.5, 0.0, 1.0))
        if fade_end > 0:
            out = np.clip(out.astype(np.float32) * (1.0 - fade_end), 0, 255).astype(np.uint8)

        return out

    @property
    def is_active(self):
        return self.active