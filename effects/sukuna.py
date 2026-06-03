import cv2
import numpy as np


class MalevolentShrine:
    def __init__(self, frame_shape):
        self.h, self.w = frame_shape[:2]
        self.active = False
        self.t = 0.0
        cx = self.w // 2
        cy = self.h // 2

        # ── TIMINGS ───────────────────────────────────────────────────────────
        self.T_SCATTER_START = 0.13
        self.T_SCATTER_DUR   = 0.47
        self.T_GROUND_START  = 0.47
        self.T_GROUND_DUR    = 0.67
        self.T_PILLAR_START  = 1.00
        self.T_PILLAR_DUR    = 0.73
        self.T_KASAGI_START  = 1.40
        self.T_KASAGI_DUR    = 0.60
        self.T_FADE_START    = 3.47
        self.T_END           = 4.00

        # ── LAYOUT ────────────────────────────────────────────────────────────
        self.horizon_y = int(self.h * 0.45)
        self.ground_y  = int(self.h * 0.68)

        depth_at_base = (self.ground_y - self.horizon_y) / max(1, self.h - self.horizon_y)
        gate_hw = int(self.w * 0.10 * (depth_at_base ** 0.7))

        self.fp_lx  = cx - gate_hw
        self.fp_rx  = cx + gate_hw
        self.fp_top = int(self.h * 0.40)
        self.fp_bot = self.ground_y
        self.fp_hw  = max(2, int(self.w * 0.007))

        self.rp_lx  = cx - int(gate_hw * 0.65)
        self.rp_rx  = cx + int(gate_hw * 0.65)
        self.rp_top = int(self.h * 0.44)
        self.rp_bot = self.ground_y
        self.rp_hw  = max(2, int(self.w * 0.005))

        self.kasagi_cx            = cx
        self.kasagi_top_y         = int(self.h * 0.32)
        self.kasagi_lx            = self.fp_lx - int(self.w * 0.05)
        self.kasagi_rx_coord      = self.fp_rx + int(self.w * 0.05)
        self.kasagi_end_y         = self.fp_top
        self.kasagi_beam_thickness= max(2, int(self.h * 0.016))

        # ── SCATTER PARTICLES ─────────────────────────────────────────────────
        rng0 = np.random.default_rng(3)
        self.N_sc = 5000
        sc_a = rng0.uniform(0, 2*np.pi, self.N_sc).astype(np.float32)
        sc_r_norm = np.clip(rng0.exponential(0.30, self.N_sc).astype(np.float32), 0.02, 1.0)
        sc_r = (sc_r_norm * min(self.w, self.h) * 0.50).astype(np.float32)
        self.sc_tx    = (cx + sc_r * np.cos(sc_a)).astype(np.float32)
        self.sc_ty    = (cy + sc_r * np.sin(sc_a)).astype(np.float32)
        self.sc_speed = rng0.uniform(0.2, 1.0, self.N_sc).astype(np.float32)
        self.sc_red   = (rng0.random(self.N_sc) < 0.70).astype(np.float32)
        self.sc_bright= rng0.uniform(140, 255, self.N_sc).astype(np.float32)
        self.sc_size  = np.ones(self.N_sc, dtype=np.int32)

        # ── GROUND PARTICLES ──────────────────────────────────────────────────
        rng1 = np.random.default_rng(42)
        N_raw = 150000
        road_span = self.h - self.horizon_y
        gy_raw = rng1.uniform(self.horizon_y, self.h - 1, N_raw).astype(np.float32)
        depth_frac = (gy_raw - self.horizon_y) / max(1, road_span)
        half_w = self.w * 0.03 + (depth_frac ** 0.7) * (self.w * 0.49)
        gx_off = rng1.uniform(-1.0, 1.0, N_raw).astype(np.float32)
        gx_raw = cx + gx_off * half_w

        mask = (gx_raw >= 0) & (gx_raw < self.w) & (gy_raw >= 0) & (gy_raw < self.h)
        gx = gx_raw[mask].astype(np.float32)
        gy = gy_raw[mask].astype(np.float32)
        depth_frac_kept = depth_frac[mask]

        self.N_gp   = len(gx)
        self.gp_x   = gx
        self.gp_y   = gy
        base_bright = 70 + depth_frac_kept * 140
        self.gp_bright = np.clip(
            base_bright + rng1.uniform(-25, 25, self.N_gp), 40, 230
        ).astype(np.float32)
        self.gp_phase = rng1.uniform(0, 2*np.pi, self.N_gp).astype(np.float32)
        self.gp_freq  = rng1.uniform(1.0, 3.0,   self.N_gp).astype(np.float32)
        self.gp_size  = np.ones(self.N_gp, dtype=np.int32)
        self.gp_vy    = np.where(depth_frac_kept < 0.25,
                                  rng1.uniform(-0.12, 0.0, self.N_gp),
                                  0.0).astype(np.float32)
        self.gp_y_off = np.zeros(self.N_gp, dtype=np.float32)

        # ── PILLAR PARTICLES ──────────────────────────────────────────────────
        rng2 = np.random.default_rng(7)
        self.N_pp     = 4000
        self.pp_norm  = rng2.uniform(0, 1, self.N_pp).astype(np.float32)
        self.pp_xoff  = rng2.normal(0, 0.12, self.N_pp).astype(np.float32)
        self.pp_speed = rng2.uniform(0.3, 1.0, self.N_pp).astype(np.float32)
        self.pp_bright= rng2.uniform(160, 255,  self.N_pp).astype(np.float32)
        self.pp_size  = rng2.choice([1, 1, 1, 2], size=self.N_pp)

        # ── KASAGI PARTICLES ──────────────────────────────────────────────────
        rng3 = np.random.default_rng(13)
        self.N_kp = 5000
        kp_t = rng3.uniform(0.0, 1.0, self.N_kp).astype(np.float32)
        P0x = float(self.kasagi_lx);       P0y = float(self.kasagi_end_y)
        P1x = float(self.kasagi_cx);       P1y = float(self.kasagi_top_y)
        P2x = float(self.kasagi_rx_coord); P2y = float(self.kasagi_end_y)
        mt = 1.0 - kp_t
        bx = mt*mt*P0x + 2*mt*kp_t*P1x + kp_t*kp_t*P2x
        by = mt*mt*P0y + 2*mt*kp_t*P1y + kp_t*kp_t*P2y
        dx = 2*(1-kp_t)*(P1x-P0x) + 2*kp_t*(P2x-P1x)
        dy = 2*(1-kp_t)*(P1y-P0y) + 2*kp_t*(P2y-P1y)
        dlen = np.sqrt(dx*dx + dy*dy) + 1e-6
        nx = -dy / dlen
        ny =  dx / dlen
        scatter = rng3.normal(0, self.kasagi_beam_thickness * 0.5, self.N_kp).astype(np.float32)
        self.kp_ax      = (bx + nx * scatter).astype(np.float32)
        self.kp_ay      = (by + ny * scatter).astype(np.float32)
        self.kp_t_param = np.abs(kp_t - 0.5) * 2.0
        self.kp_bright  = rng3.uniform(140, 230, self.N_kp).astype(np.float32)
        self.kp_phase   = rng3.uniform(0, 2*np.pi, self.N_kp).astype(np.float32)
        self.kp_freq    = rng3.uniform(2.0, 6.0,   self.N_kp).astype(np.float32)
        self.kp_size    = np.ones(self.N_kp, dtype=np.int32)

    # ─────────────────────────────────────────────────────────────────────────
    def activate(self):
        self.active = True
        self.t = 0.0
        self.gp_y_off[:] = 0.0
        self.pp_norm[:] = np.random.default_rng(99).uniform(0, 1, self.N_pp)

    # ─────────────────────────────────────────────────────────────────────────
    def _draw_scatter(self, canvas, phase, alpha):
        if alpha <= 0:
            return
        cx, cy = self.w // 2, self.h // 2
        ease = np.clip(phase * self.sc_speed, 0.0, 1.0).astype(np.float32)
        fade = alpha * (1.0 - ease * 0.7)
        px = np.clip((cx + (self.sc_tx - cx) * ease).astype(np.int32), 0, self.w-1)
        py = np.clip((cy + (self.sc_ty - cy) * ease).astype(np.int32), 0, self.h-1)
        brights = np.clip(self.sc_bright * fade, 0, 255).astype(np.uint8)

        valid = brights >= 8
        px, py, brights = px[valid], py[valid], brights[valid]
        red = self.sc_red[valid].astype(bool)

        canvas[py[red],  px[red],  2] = np.maximum(canvas[py[red],  px[red],  2], brights[red])
        canvas[py[red],  px[red],  1] = np.maximum(canvas[py[red],  px[red],  1],
                                                     (brights[red].astype(np.float32)*0.05).astype(np.uint8))
        canvas[py[~red], px[~red], 0] = np.maximum(canvas[py[~red], px[~red], 0], brights[~red])
        canvas[py[~red], px[~red], 1] = np.maximum(canvas[py[~red], px[~red], 1], brights[~red])
        canvas[py[~red], px[~red], 2] = np.maximum(canvas[py[~red], px[~red], 2], brights[~red])

    # ─────────────────────────────────────────────────────────────────────────
    def _draw_ground(self, canvas, progress, alpha):
        if alpha <= 0 or progress <= 0:
            return
        cx = self.w / 2
        road_span = float(self.h - self.horizon_y)
        twinkle = 0.80 + 0.20 * np.sin(self.t * self.gp_freq + self.gp_phase)
        bright  = np.clip(self.gp_bright * twinkle * alpha, 0, 255).astype(np.uint8)
        y_act   = self.gp_y + self.gp_y_off

        depth_frac = np.clip((self.gp_y - self.horizon_y) / max(1, road_span), 0, 1)
        allowed_hw = (self.w * 0.03 + (depth_frac ** 0.7) * (self.w * 0.49)) * progress
        mask = (np.abs(self.gp_x - cx) <= allowed_hw) & (bright >= 6)

        xs = np.clip(self.gp_x[mask].astype(np.int32), 0, self.w-1)
        ys = np.clip(y_act[mask].astype(np.int32),      0, self.h-1)
        bs = bright[mask]

        canvas[ys, xs, 2] = np.maximum(canvas[ys, xs, 2], bs)

        if progress > 0.4:
            glow = np.zeros((self.h, self.w, 3), dtype=np.uint8)
            gc = (int(cx), self.ground_y)
            rx = max(1, int(self.w * 0.10 * progress))
            ry = max(1, int(self.h * 0.04 * progress))
            cv2.ellipse(glow, gc, (rx, ry), 0, 0, 360, (0, 0, 55), -1)
            glow = cv2.GaussianBlur(glow, (41, 41), 0)
            cv2.addWeighted(canvas, 1.0, glow, float(alpha) * 0.5, 0, dst=canvas)

    # ─────────────────────────────────────────────────────────────────────────
    def _draw_pillar(self, canvas, cx_pil, hw, top_y, bot_y,
                     progress, alpha, blur=False):
        if alpha <= 0 or progress <= 0:
            return
        current_top = int(bot_y - (bot_y - top_y) * progress)
        span = bot_y - current_top
        if span <= 0:
            return

        ys = (current_top + self.pp_norm * span).astype(np.int32)
        xs = np.clip((cx_pil + self.pp_xoff * hw).astype(np.int32), 0, self.w-1)
        valid = (ys >= current_top) & (ys < bot_y) & (ys >= 0) & (ys < self.h) & \
                (xs >= 0) & (xs < self.w)
        ys = ys[valid]; xs = xs[valid]

        edge   = np.sin(self.pp_norm[valid] * np.pi)
        bright = np.clip(self.pp_bright[valid] * edge * alpha, 0, 255).astype(np.uint8)
        sz     = self.pp_size[valid]

        layer = np.zeros((self.h, self.w, 3), dtype=np.uint8)

        m1 = sz == 1
        layer[ys[m1], xs[m1], 0] = np.maximum(layer[ys[m1], xs[m1], 0], bright[m1])
        layer[ys[m1], xs[m1], 1] = np.maximum(layer[ys[m1], xs[m1], 1], bright[m1])
        layer[ys[m1], xs[m1], 2] = np.maximum(layer[ys[m1], xs[m1], 2], bright[m1])

        for i in np.where(~m1)[0]:
            b = int(bright[i])
            cv2.circle(layer, (int(xs[i]), int(ys[i])), 2, (b, b, b), -1)

        glow = cv2.GaussianBlur(layer, (7, 7), 0)
        cv2.addWeighted(canvas, 1.0, glow, 0.4 * alpha, 0, dst=canvas)
        if blur:
            layer = cv2.GaussianBlur(layer, (5, 5), 0)
            cv2.addWeighted(canvas, 1.0, layer, 0.40 * alpha, 0, dst=canvas)
        else:
            cv2.addWeighted(canvas, 1.0, layer, alpha, 0, dst=canvas)

    # ─────────────────────────────────────────────────────────────────────────
    def _draw_kasagi(self, canvas, progress, alpha):
        if alpha <= 0 or progress <= 0:
            return
        visible = self.kp_t_param <= progress
        if not np.any(visible):
            return

        twinkle = 0.80 + 0.20 * np.sin(self.t * self.kp_freq + self.kp_phase)
        brights = np.clip(self.kp_bright * twinkle * alpha, 0, 255).astype(np.uint8)

        layer = np.zeros((self.h, self.w, 3), dtype=np.uint8)

        vis_mask = visible & (brights >= 8)
        xs = np.clip(self.kp_ax[vis_mask].astype(np.int32), 0, self.w-1)
        ys = np.clip(self.kp_ay[vis_mask].astype(np.int32), 0, self.h-1)
        bs = brights[vis_mask]
        layer[ys, xs, 0] = np.maximum(layer[ys, xs, 0], bs)
        layer[ys, xs, 1] = np.maximum(layer[ys, xs, 1], bs)
        layer[ys, xs, 2] = np.maximum(layer[ys, xs, 2], bs)

        glow = cv2.GaussianBlur(layer, (5, 5), 0)
        cv2.addWeighted(canvas, 1.0, glow, 0.5 * alpha, 0, dst=canvas)
        cv2.addWeighted(canvas, 1.0, layer, alpha, 0, dst=canvas)

        cx = self.w // 2
        glow2 = np.zeros((self.h, self.w, 3), dtype=np.uint8)
        orb_r = max(1, int(self.w * 0.05 * progress))
        cv2.circle(glow2, (cx, self.kasagi_top_y), orb_r, (0, 15, 180), -1)
        glow2 = cv2.GaussianBlur(glow2, (35, 35), 0)
        cv2.addWeighted(canvas, 1.0, glow2, float(alpha) * 0.8, 0, dst=canvas)

        orb_alpha = float(np.clip((progress - 0.5) / 0.5, 0.0, 1.0)) * alpha
        if orb_alpha > 0:
            pulse = 0.88 + 0.12 * np.sin(self.t * 9.0)
            orb_y = int(self.fp_top + (self.fp_bot - self.fp_top) * 0.35)
            orb_layer = np.zeros((self.h, self.w, 3), dtype=np.uint8)
            for r, col in [
                (int(22 * pulse), (0, 8, 160)),
                (int(12 * pulse), (0, 30, 210)),
                (int(5  * pulse), (50, 80, 255)),
                (int(2  * pulse), (200, 220, 255)),
            ]:
                cv2.circle(orb_layer, (cx, orb_y), max(1, r), col, -1)
            orb_layer = cv2.GaussianBlur(orb_layer, (19, 19), 0)
            cv2.addWeighted(canvas, 1.0, orb_layer, orb_alpha * 1.0, 0, dst=canvas)

    # ─────────────────────────────────────────────────────────────────────────
    def update(self, frame, dt):
        self.t += dt
        if self.t > self.T_END:
            self.active = False
            return frame

        self.pp_norm   = (self.pp_norm + self.pp_speed * dt) % 1.0
        self.gp_y_off += self.gp_vy * dt
        too_high = (self.gp_y + self.gp_y_off) < self.ground_y - self.h * 0.08
        self.gp_y_off[too_high] = 0.0

        out = np.zeros((self.h, self.w, 3), dtype=np.uint8)

        if frame is not None and frame.shape == out.shape:
            cv2.addWeighted(out, 1.0, frame, 0.85, 0, dst=out)

        if self.t >= self.T_SCATTER_START:
            sc_t     = self.t - self.T_SCATTER_START
            sc_phase = sc_t / self.T_SCATTER_DUR
            sc_alpha = float(np.clip(sc_t / 0.06, 0.0, 1.0))
            sc_alpha *= float(1.0 - np.clip((sc_t - 0.20) / 0.18, 0.0, 1.0))
            self._draw_scatter(out, sc_phase, sc_alpha)

        if self.t >= self.T_GROUND_START:
            g_t    = self.t - self.T_GROUND_START
            g_prog = float(np.clip(g_t / self.T_GROUND_DUR, 0.0, 1.0))
            g_prog = g_prog * g_prog * (3.0 - 2.0 * g_prog)
            g_alpha= float(np.clip(g_t / 0.12, 0.0, 1.0))
            self._draw_ground(out, g_prog, g_alpha)

        if self.t >= self.T_PILLAR_START:
            p_t    = self.t - self.T_PILLAR_START
            p_prog = float(np.clip(p_t / self.T_PILLAR_DUR, 0.0, 1.0)) ** 0.55
            p_alpha= float(np.clip(p_t / 0.15, 0.0, 0.45))
            self._draw_pillar(out, self.rp_lx, self.rp_hw,
                              self.rp_top, self.rp_bot, p_prog, p_alpha, blur=True)
            self._draw_pillar(out, self.rp_rx, self.rp_hw,
                              self.rp_top, self.rp_bot, p_prog, p_alpha, blur=True)

        if self.t >= self.T_PILLAR_START:
            p_t    = self.t - self.T_PILLAR_START
            p_prog = float(np.clip(p_t / self.T_PILLAR_DUR, 0.0, 1.0)) ** 0.55
            p_alpha= float(np.clip(p_t / 0.15, 0.0, 1.0))
            self._draw_pillar(out, self.fp_lx, self.fp_hw,
                              self.fp_top, self.fp_bot, p_prog, p_alpha, blur=False)
            self._draw_pillar(out, self.fp_rx, self.fp_hw,
                              self.fp_top, self.fp_bot, p_prog, p_alpha, blur=False)

        if self.t >= self.T_KASAGI_START:
            k_t    = self.t - self.T_KASAGI_START
            k_prog = float(np.clip(k_t / self.T_KASAGI_DUR, 0.0, 1.0)) ** 0.5
            k_alpha= float(np.clip(k_t / 0.12, 0.0, 1.0))
            self._draw_kasagi(out, k_prog, k_alpha)

        if self.t >= self.T_FADE_START:
            fade = float(np.clip(
                (self.t - self.T_FADE_START) / (self.T_END - self.T_FADE_START),
                0.0, 1.0))
            out = np.clip(out.astype(np.float32) * (1.0 - fade), 0, 255).astype(np.uint8)

        return out

    @property
    def is_active(self):
        return self.active