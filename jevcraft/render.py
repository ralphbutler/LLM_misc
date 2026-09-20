"""Frame composition: game view + side panel, and the MP4 / live outputs.

Legibility first (Ralph, phase 0: "not clear what is going on"). The camera
follows the player, so the panel supplies the context the camera can't:
  - intent in words ("Heading to the nearest tree - have 1, want 4")
  - a yellow outline on the tile being worked toward
  - a minimap with the player, a fading trail, the target and the view box
  - the decision: every option that was on the menu, the one taken, and why

The game frame is always the `obs` returned by env.step(). Never call
env.render() - at night Crafter draws its darkness noise from the world's own
random generator, so extra renders change what the zombies do.
"""

from __future__ import annotations

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from state import VIEW

MAT_COLORS = {
    None: (22, 22, 26), "water": (58, 108, 204), "grass": (74, 150, 62),
    "stone": (128, 128, 128), "path": (186, 166, 118), "sand": (222, 204, 142),
    "tree": (28, 86, 30), "lava": (236, 86, 24), "coal": (36, 36, 36),
    "iron": (206, 150, 110), "diamond": (120, 232, 232), "table": (160, 104, 48),
    "furnace": (96, 60, 52),
}
BG = (24, 26, 32)
FG = (236, 236, 240)
DIM = (150, 154, 166)
ACCENT = (255, 214, 64)
GOOD = (120, 220, 130)
BAD = (240, 110, 100)
BAR = (90, 140, 230)
WARN = (255, 170, 60)
INFO = (130, 170, 255)
ALERT_COLORS = {"red": BAD, "amber": WARN, "green": GOOD, "blue": INFO}
NIGHT_LIFT = 0.8          # extra brightness at full dark (1.8x); tune on real night frames
MINIMAP_WINDOW = 40       # tiles shown in explored mode (see _minimap)
MINIMAP_MARGIN = 6        # slide mode: how close to the edge before the window moves
INV_SHOW = ["wood", "stone", "coal", "iron", "sapling",
            "wood_pickaxe", "stone_pickaxe", "iron_pickaxe",
            "wood_sword", "stone_sword", "iron_sword"]


def _font(size):
    try:
        return ImageFont.load_default(size=size)
    except TypeError:                       # very old Pillow
        return ImageFont.load_default()


class Renderer:
    def __init__(self, env, game_size, panel_w, agent_label, seed, goal_label, minimap="slide"):
        self.textures = env._textures
        self.game = game_size
        self.unit = game_size // VIEW[0]
        self.panel_w = panel_w
        self.W, self.H = game_size + panel_w, game_size
        self.agent_label = agent_label
        self.seed = seed
        self.goal_label = goal_label
        self._mm = None               # explored-mode minimap window (x0, y0, side)
        self.minimap = minimap        # "slide" or "widen" - see _minimap
        self.f = {k: _font(v) for k, v in
                  dict(tiny=16, small=19, body=23, big=31, title=44, huge=64).items()}

    # --- helpers -------------------------------------------------------------

    def _icon(self, name, size):
        try:
            tex = self.textures.get(name, (size, size))
        except Exception:
            return None
        mode = "RGBA" if tex.shape[-1] == 4 else "RGB"
        return Image.fromarray(tex.astype(np.uint8), mode)

    def _wrap(self, draw, text, font, width):
        words, lines, cur = text.split(), [], ""
        for w in words:
            trial = (cur + " " + w).strip()
            if draw.textlength(trial, font=font) <= width:
                cur = trial
            else:
                if cur:
                    lines.append(cur)
                cur = w
        if cur:
            lines.append(cur)
        return lines

    # --- main composition ---------------------------------------------------

    def compose(self, obs, st, hud):
        canvas = Image.new("RGB", (self.W, self.H), BG)
        game = Image.fromarray(obs.astype(np.uint8))
        if st.daylight < 0.5:
            # Brightness floor (decision 7): keep night visibly dark but readable.
            # Applied to our copy of the frame only - the game still sees true
            # night, and we never call env.render() (it would change the zombies).
            from PIL import ImageEnhance
            game = ImageEnhance.Brightness(game).enhance(1.0 + NIGHT_LIFT * (0.5 - st.daylight) / 0.5)
        g = ImageDraw.Draw(game)
        target = hud.get("target")
        if target is not None:
            cx = target[0] - st.pos[0] + VIEW[0] // 2
            cy = target[1] - st.pos[1] + VIEW[1] // 2
            if 0 <= cx < VIEW[0] and 0 <= cy < VIEW[1]:
                u = self.unit
                g.rectangle([cx * u + 2, cy * u + 2, (cx + 1) * u - 3, (cy + 1) * u - 3],
                            outline=ACCENT, width=5)
        self._alerts(game, hud.get("alerts", []))
        canvas.paste(game, (0, 0))
        self._panel(canvas, st, hud)
        return np.asarray(canvas)

    def _alerts(self, game, alerts):
        """Attention banners across the top of the game view - where the eye is."""
        if not alerts:
            return
        overlay = Image.new("RGBA", game.size, (0, 0, 0, 0))
        od = ImageDraw.Draw(overlay)
        y = 14
        for text, color in alerts[:3]:
            col = ALERT_COLORS.get(color, FG)
            tw = od.textlength(text, font=self.f["big"])
            x = (game.size[0] - tw) / 2
            od.rounded_rectangle([x - 18, y - 6, x + tw + 18, y + 40], radius=10,
                                 fill=(12, 12, 16, 215), outline=col + (255,), width=3)
            od.text((x, y - 1), text, font=self.f["big"], fill=col + (255,))
            y += 56
        game.paste(Image.alpha_composite(game.convert("RGBA"), overlay).convert("RGB"))

    def _panel(self, canvas, st, hud):
        d = ImageDraw.Draw(canvas)
        x0, pad = self.game, 22
        x, w = x0 + pad, self.panel_w - 2 * pad
        f = self.f

        # header
        d.text((x, 14), "jevcraft", font=f["big"], fill=FG)
        d.text((x + 150, 22), "%s agent  ·  seed %d  ·  step %d"
               % (self.agent_label, self.seed, st.step), font=f["small"], fill=DIM)
        y = 60
        d.text((x, y), "GOAL", font=f["small"], fill=DIM)
        d.text((x + 70, y - 3), self.goal_label, font=f["body"], fill=ACCENT)
        if st.daylight < 0.45:
            d.rounded_rectangle([x + w - 110, y - 6, x + w, y + 24], radius=6, fill=(40, 50, 110))
            d.text((x + w - 96, y - 2), "NIGHT", font=f["small"], fill=FG)
        y += 40
        d.line([x, y, x + w, y], fill=(60, 64, 76), width=2)
        y += 12

        # what it's doing, in words
        d.text((x, y), "NOW", font=f["small"], fill=DIM)
        y += 24
        for line in self._wrap(d, hud.get("intent", ""), f["big"], w)[:2]:
            d.text((x, y), line, font=f["big"], fill=FG)
            y += 38
        reason = hud.get("reason", "")
        if reason:
            for line in self._wrap(d, reason, f["body"], w)[:2]:
                d.text((x, y), line, font=f["body"], fill=DIM)
                y += 28
        y += 10
        d.line([x, y, x + w, y], fill=(60, 64, 76), width=2)
        y += 12

        # the last decision: full menu, what was taken, why
        dec = hud.get("decision")
        if dec:
            extra = dec.get("extra") or {}
            is_opus = dec["source"] == "opus"
            head = "DECISION %d  ·  %s" % (dec["n"], "OPUS (Jev was unsure)" if is_opus else dec["source"])
            if extra.get("latency_s"):
                head += "  ·  Jev %d ms" % round(1000 * extra["latency_s"])
            if is_opus and extra.get("opus_seconds") is not None:
                head += "  ·  Opus %.1f s" % extra["opus_seconds"]
            d.text((x, y), head, font=f["small"],
                   fill=INFO if is_opus else (ACCENT if hud.get("deciding") else DIM))
            y += 26
            probs = dec.get("probs")
            shown_opts = dec["options"]
            if len(shown_opts) > 8:
                # Iron menus can exceed 8: show the 8 most likely (by Jev's
                # probabilities when there are any), always including the chosen one.
                ranked = sorted(shown_opts, key=lambda kl: -(probs or {}).get(kl[0], 0.0)) if probs else shown_opts
                keep = ranked[:8]
                if dec["chosen"] not in [k for k, _ in keep]:
                    keep = keep[:7] + [kl for kl in shown_opts if kl[0] == dec["chosen"]]
                keep_keys = {k for k, _ in keep}
                shown_opts = [kl for kl in shown_opts if kl[0] in keep_keys]
                d.text((x + w - 150, y - 26), "top 8 of %d" % len(dec["options"]),
                       font=f["tiny"], fill=DIM)
            for key, label in shown_opts:
                chosen = key == dec["chosen"]
                col = FG if chosen else DIM
                mark = ">" if chosen else " "
                d.text((x, y), mark, font=f["body"], fill=ACCENT)
                text_w = w - 24 - (90 if probs else 0)
                shown = label
                while d.textlength(shown, font=f["small"]) > text_w and len(shown) > 4:
                    shown = shown[:-2]
                if shown != label:
                    shown = shown.rstrip() + "…"
                d.text((x + 22, y + 3), shown, font=f["small"], fill=col)
                if probs:
                    p = probs.get(key, 0.0)
                    bx = x + w - 84
                    d.rectangle([bx, y + 6, bx + 60, y + 18], outline=(70, 74, 88))
                    d.rectangle([bx, y + 6, bx + int(60 * p), y + 18], fill=BAR)
                    d.text((bx + 64, y + 3), "%d" % round(100 * p), font=f["tiny"], fill=col)
                y += 27
            if dec.get("note"):
                prefix = "Opus: " if is_opus else "why: "
                for line in self._wrap(d, prefix + dec["note"], f["small"], w)[:3 if is_opus else 2]:
                    d.text((x, y + 2), line, font=f["small"], fill=INFO if is_opus else GOOD)
                    y += 24
            if is_opus and extra.get("jev_choice"):
                jl = dict(dec["options"]).get(extra["jev_choice"], extra["jev_choice"])
                verdict = "Opus overrode it" if extra.get("overrode") else "Opus agreed"
                for line in self._wrap(d, "Jev leaned to: %s (%d%%) - %s"
                                       % (jl, round(100 * (extra.get("jev_confidence") or 0)), verdict),
                                       f["small"], w)[:2]:
                    d.text((x, y + 2), line, font=f["small"], fill=WARN)
                    y += 24
            # the two side questions, colored when they fire
            if extra.get("danger") is not None:
                dg, rc = extra["danger"], extra["recover"]
                d.text((x, y + 4), "danger %d%%" % round(100 * dg), font=f["small"],
                       fill=BAD if dg >= 0.5 else DIM)
                d.text((x + 150, y + 4), "recover %d%%" % round(100 * rc), font=f["small"],
                       fill=WARN if rc >= 0.5 else DIM)
                conf = extra.get("confidence")
                if conf is not None:
                    d.text((x + 320, y + 4), "confidence %d%%" % round(100 * conf), font=f["small"],
                           fill=WARN if conf < 0.55 else DIM)
                y += 26
            sc = extra.get("script_choice")
            if sc and dec["source"] in ("jev", "opus") and sc != dec["chosen"]:
                label = dict(dec["options"]).get(sc, sc)
                for line in self._wrap(d, "differs from the script, which picks: " + label,
                                       f["small"], w)[:2]:
                    d.text((x, y + 2), line, font=f["small"], fill=(230, 130, 230))
                    y += 24
        y = max(y + 8, 470)

        # inventory, as the game's own icons
        d.text((x, y), "INVENTORY", font=f["small"], fill=DIM)
        y += 26
        ix = x
        items = [(k, st.inv.get(k, 0)) for k in INV_SHOW if st.inv.get(k, 0) > 0]
        if not items:
            d.text((ix, y + 6), "(empty)", font=f["small"], fill=DIM)
        for name, n in items:
            icon = self._icon(name, 40)
            if icon is not None:
                canvas.paste(icon, (ix, y), icon if icon.mode == "RGBA" else None)
            d.text((ix + 42, y + 12), "%d" % n, font=f["body"], fill=FG)
            ix += 80
            if ix > x + w - 70:
                ix, y = x, y + 46
        y += 56

        # minimap + achievements
        mm = hud.get("minimap_px", 256)
        self._minimap(canvas, st, hud, x, self.H - mm - pad, mm)
        ax = x + mm + 20
        ay = self.H - mm - pad
        d.text((ax, ay), "ACHIEVEMENTS", font=f["small"], fill=DIM)
        ay += 26
        for a in sorted(st.achievements):
            d.text((ax, ay), "+ " + a.replace("_", " "), font=f["small"], fill=GOOD)
            ay += 23

    def _minimap(self, canvas, st, hud, x, y, px):
        known = st.known
        ax, ay = known.area
        rgb = np.zeros((ay, ax, 3), np.uint8)
        for mat, col in MAT_COLORS.items():
            mask = (known.mat == mat)
            rgb[mask.T] = col
        # Explored mode: a 40x40-tile window, fixed scale. Two behaviours
        # (--minimap), kept side by side so either can be chosen:
        #   slide (default) - game-camera deadzone: the window stays still while
        #       the player is more than MINIMAP_MARGIN tiles from its edge, then
        #       slides one tile at a time to keep them in view. Scale never
        #       changes and nothing jumps.
        #   widen - the window stays still until the player nears its edge, then
        #       switches to the whole world once, permanently. Ralph saw that as
        #       a jump when Jev explored far on seed 21.
        if known.mode == "explored":
            side = min(MINIMAP_WINDOW, ax)
            if self._mm is None:
                ox = int(min(max(st.pos[0] - side // 2, 0), ax - side))
                oy = int(min(max(st.pos[1] - side // 2, 0), ay - side))
                self._mm = (ox, oy, side)
            ox, oy, side = self._mm
            if self.minimap == "slide":
                m = MINIMAP_MARGIN
                px_, py_ = st.pos
                if px_ < ox + m:
                    ox = px_ - m
                elif px_ >= ox + side - m:
                    ox = px_ - side + m + 1
                if py_ < oy + m:
                    oy = py_ - m
                elif py_ >= oy + side - m:
                    oy = py_ - side + m + 1
                ox = int(min(max(ox, 0), ax - side))
                oy = int(min(max(oy, 0), ay - side))
                self._mm = (ox, oy, side)
            elif side < ax and not (ox + 2 <= st.pos[0] < ox + side - 2 and oy + 2 <= st.pos[1] < oy + side - 2):
                self._mm = (0, 0, ax)
                ox, oy, side = self._mm
            x0, y0 = ox, oy
            rgb = rgb[y0:y0 + side, x0:x0 + side]
        else:
            x0 = y0 = 0
            side = ax
        img = Image.fromarray(rgb).resize((px, px), Image.NEAREST)
        dd = ImageDraw.Draw(img)
        s = px / side
        st_pos = (st.pos[0] - x0, st.pos[1] - y0)
        trail = [(tx - x0, ty - y0) for tx, ty in hud.get("trail", [])]
        n = len(trail)
        for i, (tx, ty) in enumerate(trail):
            a = 0.25 + 0.75 * (i + 1) / max(n, 1)
            c = tuple(int(v * a) for v in (255, 255, 255))
            dd.rectangle([tx * s + 1, ty * s + 1, (tx + 1) * s - 2, (ty + 1) * s - 2], fill=c)
        vx0 = (st_pos[0] - VIEW[0] // 2) * s
        vy0 = (st_pos[1] - VIEW[1] // 2) * s
        dd.rectangle([vx0, vy0, vx0 + VIEW[0] * s, vy0 + VIEW[1] * s], outline=(255, 255, 255))
        t = hud.get("target")
        if t is not None:
            t = (t[0] - x0, t[1] - y0)
            dd.rectangle([t[0] * s - 2, t[1] * s - 2, (t[0] + 1) * s + 1, (t[1] + 1) * s + 1],
                         outline=ACCENT, width=2)
        dd.rectangle([st_pos[0] * s - 1, st_pos[1] * s - 1, (st_pos[0] + 1) * s, (st_pos[1] + 1) * s],
                     fill=BAD, outline=(255, 255, 255))
        canvas.paste(img, (x, y))
        ImageDraw.Draw(canvas).rectangle([x - 1, y - 1, x + px, y + px], outline=(70, 74, 88))

    # --- title / end cards ---------------------------------------------------

    def card(self, title, lines, sub=None):
        img = Image.new("RGB", (self.W, self.H), BG)
        d = ImageDraw.Draw(img)
        wrapped = []
        for line in lines:                  # wrap anything wider than the card
            wrapped.extend(self._wrap(d, line, self.f["body"], self.W - 160) or [""])
        tw = d.textlength(title, font=self.f["huge"])
        y = max(40, self.H // 2 - 60 - 18 * len(wrapped))
        d.text(((self.W - tw) / 2, y), title, font=self.f["huge"], fill=FG)
        y += 90
        if sub:
            sw = d.textlength(sub, font=self.f["big"])
            d.text(((self.W - sw) / 2, y), sub, font=self.f["big"], fill=ACCENT)
            y += 56
        for line in wrapped:
            lw = d.textlength(line, font=self.f["body"])
            d.text(((self.W - lw) / 2, y), line, font=self.f["body"], fill=DIM)
            y += 34
        return np.asarray(img)


class VideoOut:
    """MP4 writer with frame holding, so decisions stay on screen long enough to read."""

    def __init__(self, path, fps):
        import imageio.v2 as imageio
        self.w = imageio.get_writer(path, fps=fps, codec="libx264",
                                    pixelformat="yuv420p", macro_block_size=1)
        self.fps = fps
        self.frames = 0

    def add(self, frame, seconds=None):
        k = 1 if seconds is None else max(1, int(round(seconds * self.fps)))
        for _ in range(k):
            self.w.append_data(frame)
        self.frames += k

    def close(self):
        self.w.close()


class LiveWindow:
    """Debug viewer (--live). Shows the same composed frames as the MP4."""

    def __init__(self, w, h, fps, title, scale=1.0):
        import pygame
        self.pg = pygame
        pygame.init()
        self.scale = scale
        self.size = (int(w * scale), int(h * scale))
        self.screen = pygame.display.set_mode(self.size)
        pygame.display.set_caption(title)
        self.clock = pygame.time.Clock()
        self.fps = fps

    def show(self, frame, seconds=None):
        pg = self.pg
        ticks = 1 if seconds is None else max(1, int(round(seconds * self.fps)))
        surf = pg.surfarray.make_surface(frame.swapaxes(0, 1))
        if self.scale != 1.0:
            surf = pg.transform.smoothscale(surf, self.size)
        for _ in range(ticks):
            for ev in pg.event.get():
                if ev.type == pg.QUIT:
                    return False
                if ev.type == pg.KEYDOWN and ev.key in (pg.K_ESCAPE, pg.K_q):
                    return False
            self.screen.blit(surf, (0, 0))
            pg.display.flip()
            self.clock.tick(self.fps)
        return True

    def close(self):
        self.pg.quit()
