"""
main.py - FRAUD PATROL: learn data cleaning and z-scores by protecting a bank.
Run with:   python main.py          (needs: pip install pygame pandas numpy)

Screens: title -> intro -> play -> result -> learned -> (next day) ... -> end
Days 1-2 show a ledger TABLE (data cleaning). Day 3 shows the IQR rule. Days 4-6 show z-scores.
"""
import csv
import math
import os
import sys
from datetime import datetime

import pandas as pd
import pygame

import fraud_logic as F

W, H = 1280, 740
FPS = 60
HERE = os.path.dirname(os.path.abspath(__file__))

BG = (18, 22, 32)
PANEL = (30, 36, 50)
PANEL2 = (40, 47, 64)
LINE = (62, 70, 90)
WHITE = (236, 239, 245)
MUTED = (145, 153, 172)
ACCENT = (255, 183, 77)
GREEN = (76, 190, 100)
RED = (235, 80, 80)
BLUE = (80, 150, 255)
PURPLE = (150, 95, 220)
YELLOW_ROW = (110, 95, 30)
RED_ROW = (110, 40, 45)
CARD = (246, 241, 230)
INK = (30, 30, 36)
CODE_BG = (24, 28, 38)
CODE_FG = (152, 230, 160)


# =================================================================== small helpers
def wrap(text, font, width):
    out = []
    for para in str(text).split("\n"):
        line = ""
        for w in para.split(" "):
            t = (line + " " + w) if line else w
            if font.size(t)[0] <= width:
                line = t
            else:
                if line:
                    out.append(line)
                line = w
        out.append(line)
    return out


def text(surf, s, font, col, x, y, width=None, gap=3, limit=None):
    lines = wrap(s, font, width) if width else str(s).split("\n")
    used = 0
    for ln in lines:
        if limit is not None:
            if used >= limit:
                break
            ln = ln[: max(0, limit - used)]
            used += len(ln) + 1
        surf.blit(font.render(ln, True, col), (x, y))
        y += font.get_height() + gap
    return y


def rs(v):
    return f"Rs {v:,.0f}"


class Button:
    def __init__(self, rect, label, font, col=BLUE, enabled=True):
        self.rect, self.label, self.font, self.col, self.enabled = pygame.Rect(rect), label, font, col, enabled

    def draw(self, s):
        hov = self.enabled and self.rect.collidepoint(pygame.mouse.get_pos())
        c = self.col if self.enabled else (70, 74, 86)
        if hov:
            c = tuple(min(255, v + 28) for v in c)
        pygame.draw.rect(s, c, self.rect, border_radius=10)
        img = self.font.render(self.label, True, WHITE if self.enabled else MUTED)
        s.blit(img, img.get_rect(center=self.rect.center))

    def hit(self, e):
        return self.enabled and e.type == pygame.MOUSEBUTTONDOWN and e.button == 1 and self.rect.collidepoint(e.pos)


class Slider:
    """Draggable slider: continuous (lo..hi, step) or a list of options."""

    def __init__(self, key, label, lo=0.0, hi=1.0, step=0.1, value=0.0, options=None, values=None, fmt="{:.1f}"):
        self.key, self.label = key, label
        self.options, self.values = options, values
        self.lo, self.hi, self.step, self.fmt = lo, hi, step, fmt
        if options:
            self.lo, self.hi, self.step = 0, len(options) - 1, 1
        self.value = value
        self.rect = pygame.Rect(0, 0, 0, 0)
        self.drag = False

    def get(self):
        return self.values[int(self.value)] if self.options else round(self.value, 2)

    def set(self, v):
        self.value = self.values.index(v) if self.options else v

    def display(self):
        if self.key == "limit" and self.value >= self.hi:
            return "no filter"
        return self.options[int(self.value)] if self.options else self.fmt.format(self.value)

    def _x(self):
        return self.rect.x + int((self.value - self.lo) / (self.hi - self.lo) * self.rect.w)

    def draw(self, s, x, y, w, f_lab, f_val, f_tick):
        s.blit(f_lab.render(self.label, True, WHITE), (x, y))
        v = f_val.render(self.display(), True, ACCENT)
        s.blit(v, (x, y + 24))
        self.rect = pygame.Rect(x + 10, y + 66, w - 20, 8)
        pygame.draw.rect(s, LINE, self.rect, border_radius=4)
        pygame.draw.rect(s, ACCENT, (self.rect.x, self.rect.y, self._x() - self.rect.x, 8), border_radius=4)
        if self.options:
            for i, opt in enumerate(self.options):
                tx = self.rect.x + int(i / (len(self.options) - 1) * self.rect.w)
                pygame.draw.circle(s, LINE, (tx, self.rect.centery), 6)
                t = f_tick.render(opt, True, MUTED)
                tx = min(max(tx - t.get_width() // 2, x), x + w - t.get_width())
                s.blit(t, (tx, y + 82))
        pygame.draw.circle(s, WHITE, (self._x(), self.rect.centery), 12)
        pygame.draw.circle(s, ACCENT, (self._x(), self.rect.centery), 7)

    def handle(self, e):
        if e.type == pygame.MOUSEBUTTONDOWN and e.button == 1 and self.rect.inflate(24, 30).collidepoint(e.pos):
            self.drag = True
            return self._move(e.pos[0])
        if e.type == pygame.MOUSEBUTTONUP and e.button == 1:
            self.drag = False
        if e.type == pygame.MOUSEMOTION and self.drag:
            return self._move(e.pos[0])
        return False

    def _move(self, mx):
        f = min(1, max(0, (mx - self.rect.x) / max(1, self.rect.w)))
        v = round(round((self.lo + f * (self.hi - self.lo)) / self.step) * self.step, 4)
        if v != self.value:
            self.value = v
            return True
        return False


# =================================================================== characters
def face(s, cx, cy, r, mood, skin=(205, 150, 105)):
    pygame.draw.circle(s, skin, (cx, cy), r)
    ex, ey, er = r // 3, r // 5, max(2, r // 7)
    for dx in (-ex, ex):
        pygame.draw.circle(s, INK, (cx + dx, cy - ey), er)
    w = max(2, r // 9)
    if mood == "happy":
        pygame.draw.arc(s, INK, (cx - r // 2, cy - r // 5, r, int(r / 1.5)), math.pi, 2 * math.pi, w)
    elif mood == "angry":
        pygame.draw.arc(s, INK, (cx - r // 2, cy + r // 4, r, r // 2), 0, math.pi, w)
        pygame.draw.line(s, INK, (cx - ex - er * 2, cy - ey - er * 3), (cx - ex + er, cy - ey - er), 3)
        pygame.draw.line(s, INK, (cx + ex + er * 2, cy - ey - er * 3), (cx + ex - er, cy - ey - er), 3)
    else:
        pygame.draw.line(s, INK, (cx - r // 3, cy + r // 3), (cx + r // 3, cy + r // 3), w)


def draw_manager(s, x, y, sc=1.0, mood="neutral"):
    """Mrs. Rao, the bank manager."""
    r = int(32 * sc)
    pygame.draw.rect(s, (20, 120, 120), (x - int(40 * sc), y + r - 6, int(80 * sc), int(66 * sc)),
                     border_radius=int(16 * sc))
    pygame.draw.polygon(s, WHITE, [(x - 10 * sc, y + r - 2), (x + 10 * sc, y + r - 2), (x, y + r + 20 * sc)])
    pygame.draw.circle(s, (40, 30, 30), (x, y - r + 2), int(14 * sc))
    pygame.draw.circle(s, (40, 30, 30), (x, y - 4), r + 3)
    face(s, x, y + 2, r - 2, mood)
    pygame.draw.circle(s, ACCENT, (x - r + 1, y + 10), int(4 * sc))
    pygame.draw.circle(s, ACCENT, (x + r - 1, y + 10), int(4 * sc))


def draw_phantom(s, x, y, t, sc=1.0):
    """The Phantom: hooded fraudster with a white mask and a bag of money."""
    y += int(4 * math.sin(t * 3))
    pygame.draw.polygon(s, (35, 30, 50), [(x - 46 * sc, y + 80 * sc), (x + 46 * sc, y + 80 * sc),
                                          (x + 30 * sc, y - 10 * sc), (x, y - 44 * sc), (x - 30 * sc, y - 10 * sc)])
    pygame.draw.ellipse(s, (235, 235, 240), (x - 20 * sc, y - 20 * sc, 40 * sc, 44 * sc))
    for dx in (-8, 8):
        pygame.draw.ellipse(s, INK, (x + dx * sc - 5 * sc, y - 6 * sc, 10 * sc, 6 * sc))
    pygame.draw.arc(s, INK, (x - 9 * sc, y + 4 * sc, 18 * sc, 10 * sc), math.pi, 2 * math.pi, 2)
    pygame.draw.circle(s, (90, 160, 80), (int(x + 44 * sc), int(y + 60 * sc)), int(18 * sc))
    lab = pygame.font.SysFont("arial", int(20 * sc), bold=True).render("Rs", True, WHITE)
    s.blit(lab, lab.get_rect(center=(int(x + 44 * sc), int(y + 60 * sc))))


# =================================================================== the game
class Game:
    def __init__(self):
        pygame.init()
        pygame.display.set_caption("Fraud Patrol")
        self.screen = pygame.display.set_mode((W, H))
        self.clock = pygame.time.Clock()
        sf = pygame.font.SysFont
        self.f_huge = sf("arial", 54, bold=True)
        self.f_big = sf("arial", 30, bold=True)
        self.f_mid = sf("arial", 20, bold=True)
        self.f_txt = sf("arial", 18)
        self.f_sm = sf("arial", 15)
        self.f_tiny = sf("arial", 13)
        self.f_code = sf("consolas,couriernew,monospace", 16)
        self.f_cell = sf("consolas,couriernew,monospace", 15)
        self.reset()

    # ----------------------------------------------------------- state
    def reset(self):
        self.state = "title"
        self.player = ""
        self.day = 1
        self.money = F.START_MONEY
        self.trust = F.START_TRUST
        self.stars = {}
        self.runs = []
        self.t = 0.0
        self.won = False
        self.saved = False
        self.seed = int(datetime.now().timestamp())

    def make_sliders(self):
        st = F.DAYS[self.day]["start"]
        all_ = {
            "duplicates": Slider("duplicates", "Duplicate rows", options=["keep all rows", "drop_duplicates()"],
                                 values=["keep", "drop"]),
            "missing": Slider("missing", "Rows with a missing amount (NaN)",
                              options=["dropna()", "fillna(mean)", "fillna(median)", "flag with isna()"],
                              values=["dropna", "mean", "median", "flag"]),
            "m": Slider("m", "IQR multiplier m  (fences = Q1 - m*IQR, Q3 + m*IQR)", 0.2, 4.0, 0.1, st.get("m", 1.5)),
            "k": Slider("k", "Threshold k", 0.3, 4.0, 0.1, st.get("k", 2.0)),
            "limit": Slider("limit", "Filter: only use payments below this for mean/std", 2, F.NO_LIMIT, 1,
                            st.get("limit", F.NO_LIMIT), fmt="Rs {:.0f},000"),
            "tail": Slider("tail", "Tail (which direction is suspicious?)",
                           options=["upper", "lower", "both"], values=["upper", "lower", "both"]),
        }
        for k in ("duplicates", "missing", "tail"):
            if k in st:
                all_[k].set(st[k])
        self.sl = {k: all_[k] for k in F.DAYS[self.day]["controls"]}

    def val(self, key, default=None):
        return self.sl[key].get() if key in self.sl else default

    def start_day(self, replay=False):
        self.seed += 1
        self.raw = F.make_day(self.day, self.seed)
        self.make_sliders()
        self.result = None
        self.hint_on = False
        self.chars, self.line_i = 0, 0
        self.state = "play" if replay else "intro"

    def run_day(self):
        d = self.day
        if d == 1:
            self.result = F.eval_duplicates(self.raw, self.val("duplicates"))
        elif d == 2:
            self.result = F.eval_missing(self.raw, self.val("missing"))
        elif d == 3:
            self.result = F.eval_iqr(self.raw, self.val("m"))
        else:
            self.result = F.eval_z(self.raw, self.val("k"), self.val("limit", F.NO_LIMIT), self.val("tail", "upper"))
        r = self.result
        self.money -= r["loss"]
        self.trust -= r["trust_loss"]
        self.stars[d] = max(self.stars.get(d, 0), r["stars"])
        self.runs.append({"time": datetime.now().isoformat(timespec="seconds"), "player": self.player, "day": d,
                          "settings": "; ".join(f"{k}={s.get()}" for k, s in self.sl.items()),
                          "stars": r["stars"], "money_lost": round(r["loss"]), "trust_lost": r["trust_loss"],
                          "money_left": round(self.money), "trust_left": self.trust})
        self.chars = 0
        self.state = "result"

    def finish(self, won):
        self.won = won
        self.save_logs()
        self.state = "end"

    # ----------------------------------------------------------- shared drawing
    def hud(self):
        pygame.draw.rect(self.screen, PANEL, (0, 0, W, 50))
        text(self.screen, "FRAUD PATROL", self.f_big, ACCENT, 16, 8)
        if self.state in ("intro", "play", "result", "learned"):
            text(self.screen, F.DAYS[self.day]["title"], self.f_mid, WHITE, 280, 14)
        for i, (lab, v, mx, fmt) in enumerate([("Money", self.money, F.START_MONEY, rs),
                                               ("Trust", self.trust, F.START_TRUST, lambda x: f"{x:.0f}%")]):
            x = 720 + i * 280
            frac = max(0, min(1, v / mx))
            col = GREEN if frac > 0.5 else ACCENT if frac > 0.25 else RED
            text(self.screen, lab, self.f_sm, MUTED, x, 6)
            pygame.draw.rect(self.screen, LINE, (x, 26, 150, 14), border_radius=5)
            pygame.draw.rect(self.screen, col, (x, 26, int(150 * frac), 14), border_radius=5)
            text(self.screen, fmt(max(v, 0)), self.f_sm, WHITE, x + 158, 23)

    def speech(self, who, msg, typing=True, mood="neutral"):
        box = pygame.Rect(20, 560, 860, 172)
        pygame.draw.rect(self.screen, CARD, box, border_radius=12)
        if who == "phantom":
            draw_phantom(self.screen, box.x + 80, box.y + 60, self.t, 0.95)
            name, col = "THE PHANTOM", PURPLE
        else:
            draw_manager(self.screen, box.x + 80, box.y + 52, 1.0, mood)
            name, col = "MRS. RAO, BANK MANAGER", (20, 110, 110)
        text(self.screen, name, self.f_mid, col, box.x + 165, box.y + 12)
        lim = int(self.chars) if typing else None
        text(self.screen, msg, self.f_txt, INK, box.x + 165, box.y + 42, width=box.w - 185, limit=lim)
        return (not typing) or self.chars >= len(msg) + 30

    def code_box(self, code, y=470, h=80):
        box = pygame.Rect(20, y, 860, h)
        pygame.draw.rect(self.screen, CODE_BG, box, border_radius=8)
        text(self.screen, "pandas code being run:", self.f_tiny, MUTED, box.x + 12, box.y + 6)
        text(self.screen, code, self.f_code, CODE_FG, box.x + 12, box.y + 26, gap=4)

    # ----------------------------------------------------------- the ledger table (days 1-2)
    def ledger_view(self, reveal=False):
        df = self.raw
        d = self.day
        rows = []                                   # (id, customer, amount_text, row_colour, tag, tag_colour)
        if d == 1:
            choice = self.val("duplicates")
            shown = F.clean_duplicates(df, choice)
            copy = shown.duplicated()
            for (_, r), is_copy in zip(shown.iterrows(), copy):
                rows.append((r["payment_id"], r["customer"], f"{r['amount']:,.0f}",
                             YELLOW_ROW if is_copy else None, "COPY of the row above" if is_copy else "", ACCENT))
            info = f"len(df) = {len(shown)}    df['payment_id'].nunique() = {shown['payment_id'].nunique()}"
            code = "df = df.drop_duplicates()" if choice == "drop" else "# nothing: all rows kept"
        else:
            choice = self.val("missing")
            fill = df["amount"].mean() if choice == "mean" else df["amount"].median()
            for _, r in df.iterrows():
                if pd.notna(r["amount"]):
                    rows.append((r["payment_id"], r["customer"], f"{r['amount']:,.0f}", None, "", None))
                    continue
                truth = ""
                if reveal:
                    truth = " - FRAUD" if r["is_fraud"] else " - honest"
                if choice == "dropna":
                    rows.append((r["payment_id"], r["customer"], "deleted", RED_ROW, "dropped" + truth, MUTED))
                elif choice in ("mean", "median"):
                    rows.append((r["payment_id"], r["customer"], f"{fill:,.0f}", RED_ROW, "filled" + truth, ACCENT))
                else:
                    rows.append((r["payment_id"], r["customer"], "NaN", RED_ROW, "FLAGGED" + truth, GREEN))
            info = f"df['amount'].isna().sum() = {int(df['amount'].isna().sum())} missing amounts"
            code = {"dropna": "df = df.dropna(subset=['amount'])",
                    "mean": "df['amount'] = df['amount'].fillna(df['amount'].mean())",
                    "median": "df['amount'] = df['amount'].fillna(df['amount'].median())",
                    "flag": "df['needs_review'] = df['amount'].isna()"}[choice]

        box = pygame.Rect(20, 58, 860, 404)
        pygame.draw.rect(self.screen, PANEL, box, border_radius=10)
        text(self.screen, "PAYMENT LEDGER", self.f_mid, ACCENT, box.x + 14, box.y + 8)
        text(self.screen, info, self.f_cell, WHITE, box.x + 250, box.y + 12)
        half = (len(rows) + 1) // 2
        for col_i, chunk in enumerate((rows[:half], rows[half:])):
            x0 = box.x + 10 + col_i * 425
            y0 = box.y + 40
            for j, h in enumerate(["Payment ID", "Customer", "Amount"]):
                self.screen.blit(self.f_tiny.render(h, True, ACCENT), (x0 + [8, 100, 196][j], y0))
            for i, (pid, cust, amt, rc, tag, tc) in enumerate(chunk):
                y = y0 + 20 + i * 22
                pygame.draw.rect(self.screen, rc or (PANEL2 if i % 2 else PANEL), (x0, y, 415, 21))
                self.screen.blit(self.f_cell.render(str(pid), True, WHITE), (x0 + 8, y + 2))
                self.screen.blit(self.f_cell.render(cust, True, WHITE), (x0 + 100, y + 2))
                self.screen.blit(self.f_cell.render(amt, True, RED if amt in ("NaN", "deleted") else WHITE),
                                 (x0 + 196, y + 2))
                if tag:
                    self.screen.blit(self.f_tiny.render(tag, True, tc or WHITE), (x0 + 270, y + 4))
        self.code_box(code)

    # ----------------------------------------------------------- the scatter plot (days 3-5)
    def plot_view(self, reveal=False):
        k, limit, tail = self.val("k"), self.val("limit", F.NO_LIMIT), self.val("tail", "upper")
        d, mean, std = F.detect(self.raw, k, limit, tail)
        box = pygame.Rect(20, 58, 860, 404)
        pygame.draw.rect(self.screen, PANEL, box, border_radius=10)
        area = pygame.Rect(box.x + 70, box.y + 16, box.w - 90, box.h - 50)
        vals = d["amount"].to_numpy(float)
        lo, hi = vals.min(), vals.max()
        lo, hi = min(lo, mean - k * std), max(hi, mean + k * std)
        pad = (hi - lo) * 0.06
        lo, hi = lo - pad, hi + pad

        def Y(v):
            return area.bottom - (v - lo) / (hi - lo) * area.h

        for i in range(6):
            v = lo + (hi - lo) * i / 5
            yy = int(Y(v))
            pygame.draw.line(self.screen, LINE, (area.x, yy), (area.right, yy), 1)
            t = self.f_tiny.render(f"{v / 1000:.0f}k" if abs(v) >= 10000 else f"{v:,.0f}", True, MUTED)
            self.screen.blit(t, (area.x - t.get_width() - 6, yy - 7))
        self.screen.blit(self.f_tiny.render("amount (Rs) of each payment      white dashes = mean      red lines = "
                                            "mean +/- k x std (the threshold)", True, MUTED), (area.x, box.bottom - 24))
        yy = Y(mean)
        for xx in range(area.x, area.right, 14):
            pygame.draw.line(self.screen, WHITE, (xx, yy), (xx + 7, yy), 1)
        if tail in ("upper", "both"):
            pygame.draw.line(self.screen, RED, (area.x, Y(mean + k * std)), (area.right, Y(mean + k * std)), 2)
        if tail in ("lower", "both"):
            pygame.draw.line(self.screen, RED, (area.x, Y(mean - k * std)), (area.right, Y(mean - k * std)), 2)
        if limit < F.NO_LIMIT:
            yl = Y(limit * 1000)
            if area.y < yl < area.bottom:
                for xx in range(area.x, area.right, 10):
                    pygame.draw.line(self.screen, PURPLE, (xx, yl), (xx + 4, yl), 2)
                self.screen.blit(self.f_tiny.render("filter limit", True, PURPLE), (area.x + 4, yl - 16))

        n = len(d)
        self.dots = []
        for i, r in d.iterrows():
            px = area.x + int((i + 0.5) / n * area.w)
            py = int(Y(r["amount"]))
            if r["blocked"]:
                pygame.draw.circle(self.screen, RED, (px, py), 6)
            else:
                pygame.draw.circle(self.screen, BLUE, (px, py), 4)
            if reveal:
                if r["is_fraud"] and r["blocked"]:
                    pygame.draw.circle(self.screen, GREEN, (px, py), 10, 3)
                elif r["is_fraud"]:
                    pygame.draw.line(self.screen, RED, (px - 8, py - 8), (px + 8, py + 8), 3)
                    pygame.draw.line(self.screen, RED, (px - 8, py + 8), (px + 8, py - 8), 3)
                elif r["blocked"]:
                    pygame.draw.circle(self.screen, ACCENT, (px, py), 10, 2)
            self.dots.append((px, py, r))

        rule = {"upper": f"z > {k:.1f}", "lower": f"z < -{k:.1f}", "both": f"|z| > {k:.1f}"}[tail]
        code = (f"mean = {mean:,.0f}    std = {std:,.0f}" + ("    (from payments below the filter only)"
                                                           if limit < F.NO_LIMIT else "") +
                f"\nz = (amount - mean) / std    block if {rule}    blocked: {int(d['blocked'].sum())}/{n}"
                f"    alpha = {F.alpha(k, tail):.1%}")
        self.code_box(code)
        self.tooltip(mean, std, reveal)

    def tooltip(self, mean, std, reveal):
        mx, my = pygame.mouse.get_pos()
        best = min(self.dots, key=lambda p: (p[0] - mx) ** 2 + (p[1] - my) ** 2, default=None)
        if not best or (best[0] - mx) ** 2 + (best[1] - my) ** 2 > 100:
            return
        r = best[2]
        lines = [f"Payment #{r['payment_id']}:  {rs(r['amount'])}",
                 f"z = ({r['amount']:,.0f} - {mean:,.0f}) / {std:,.0f} = {r['z']:.2f}",
                 "BLOCKED" if r["blocked"] else "approved"]
        if reveal:
            lines.append("TRUTH: FRAUD!" if r["is_fraud"] else "TRUTH: honest customer")
        wmax = max(self.f_sm.size(l)[0] for l in lines) + 20
        bx, by = min(mx + 14, W - wmax - 6), min(my + 14, H - 100)
        pygame.draw.rect(self.screen, (10, 12, 18), (bx, by, wmax, len(lines) * 20 + 12), border_radius=8)
        pygame.draw.rect(self.screen, ACCENT, (bx, by, wmax, len(lines) * 20 + 12), 1, border_radius=8)
        for j, l in enumerate(lines):
            col = RED if ("BLOCKED" in l or "FRAUD" in l) else WHITE
            self.screen.blit(self.f_sm.render(l, True, col), (bx + 10, by + 6 + j * 20))

    def main_view(self, reveal=False):
        kind = F.DAYS[self.day]["kind"]
        if kind == "table":
            self.ledger_view(reveal)
        elif kind == "iqr":
            self.iqr_view(reveal)
        else:
            self.plot_view(reveal)

    # ----------------------------------------------------------- the IQR view (day 3)
    def iqr_view(self, reveal=False):
        m = self.val("m")
        d, q1, q3, iqr, lo_f, hi_f = F.detect_iqr(self.raw, m)
        med = float(d["amount"].median())
        box = pygame.Rect(20, 58, 860, 404)
        pygame.draw.rect(self.screen, PANEL, box, border_radius=10)
        area = pygame.Rect(box.x + 70, box.y + 16, box.w - 90, box.h - 50)
        vals = d["amount"].to_numpy(float)
        lo, hi = min(vals.min(), lo_f), max(vals.max(), hi_f)
        pad = (hi - lo) * 0.06
        lo, hi = lo - pad, hi + pad

        def Y(v):
            return area.bottom - (v - lo) / (hi - lo) * area.h

        for i in range(6):
            v = lo + (hi - lo) * i / 5
            yy = int(Y(v))
            pygame.draw.line(self.screen, LINE, (area.x, yy), (area.right, yy), 1)
            t = self.f_tiny.render(f"{v:,.0f}", True, MUTED)
            self.screen.blit(t, (area.x - t.get_width() - 6, yy - 7))
        # middle 50% band (Q1 to Q3)
        band = pygame.Surface((area.w, max(1, int(Y(q1) - Y(q3)))), pygame.SRCALPHA)
        band.fill((80, 150, 255, 45))
        self.screen.blit(band, (area.x, int(Y(q3))))
        for v, lab in ((q1, "Q1"), (q3, "Q3")):
            pygame.draw.line(self.screen, BLUE, (area.x, Y(v)), (area.right, Y(v)), 1)
            self.screen.blit(self.f_tiny.render(lab, True, BLUE), (area.right - 22, Y(v) - (16 if lab == "Q3" else 0)))
        for xx in range(area.x, area.right, 14):
            pygame.draw.line(self.screen, WHITE, (xx, Y(med)), (xx + 7, Y(med)), 1)
        for v, lab in ((lo_f, "lower fence"), (hi_f, "upper fence")):
            pygame.draw.line(self.screen, RED, (area.x, Y(v)), (area.right, Y(v)), 2)
            self.screen.blit(self.f_tiny.render(lab, True, RED), (area.x + 4, Y(v) - 16))
        self.screen.blit(self.f_tiny.render("amount (Rs)      blue band = middle 50% (Q1 to Q3)      white dashes = median"
                                            "      red = fences", True, MUTED), (area.x, box.bottom - 24))
        n = len(d)
        self.dots = []
        for i, r in d.iterrows():
            px = area.x + int((i + 0.5) / n * area.w)
            py = int(Y(r["amount"]))
            pygame.draw.circle(self.screen, RED if r["blocked"] else BLUE, (px, py), 6 if r["blocked"] else 4)
            if reveal:
                if r["is_fraud"] and r["blocked"]:
                    pygame.draw.circle(self.screen, GREEN, (px, py), 10, 3)
                elif r["is_fraud"]:
                    pygame.draw.line(self.screen, RED, (px - 8, py - 8), (px + 8, py + 8), 3)
                    pygame.draw.line(self.screen, RED, (px - 8, py + 8), (px + 8, py - 8), 3)
                elif r["blocked"]:
                    pygame.draw.circle(self.screen, ACCENT, (px, py), 10, 2)
            self.dots.append((px, py, r))
        code = (f"Q1 = {q1:,.0f}   Q3 = {q3:,.0f}   IQR = Q3 - Q1 = {iqr:,.0f}      removed: {int(d['blocked'].sum())} of {n}\n"
                f"fences: {q1:,.0f} - {m:.1f} x {iqr:,.0f} = {lo_f:,.0f}   and   {q3:,.0f} + {m:.1f} x {iqr:,.0f} = "
                f"{hi_f:,.0f}")
        self.code_box(code)
        # tooltip
        mx, my = pygame.mouse.get_pos()
        best = min(self.dots, key=lambda p: (p[0] - mx) ** 2 + (p[1] - my) ** 2, default=None)
        if best and (best[0] - mx) ** 2 + (best[1] - my) ** 2 <= 100:
            r = best[2]
            where = "above the upper fence" if r["amount"] > hi_f else "below the lower fence" \
                if r["amount"] < lo_f else "between the fences"
            lines = [f"Payment #{r['payment_id']}:  {rs(r['amount'])}", where,
                     "REMOVED (outlier)" if r["blocked"] else "kept"]
            if reveal:
                lines.append("TRUTH: FRAUD!" if r["is_fraud"] else "TRUTH: honest customer")
            wmax = max(self.f_sm.size(l)[0] for l in lines) + 20
            bx, by = min(mx + 14, W - wmax - 6), min(my + 14, H - 100)
            pygame.draw.rect(self.screen, (10, 12, 18), (bx, by, wmax, len(lines) * 20 + 12), border_radius=8)
            pygame.draw.rect(self.screen, ACCENT, (bx, by, wmax, len(lines) * 20 + 12), 1, border_radius=8)
            for j, l in enumerate(lines):
                col = RED if ("REMOVED" in l or "FRAUD" in l) else WHITE
                self.screen.blit(self.f_sm.render(l, True, col), (bx + 10, by + 6 + j * 20))

    # ----------------------------------------------------------- right panel
    def controls(self, events, enabled=True):
        x, w = 900, 360
        pygame.draw.rect(self.screen, PANEL, (892, 58, 376, 490), border_radius=10)
        text(self.screen, "YOUR CONTROLS", self.f_mid, ACCENT, x, 66)
        y = 100
        for s in self.sl.values():
            s.draw(self.screen, x, y, w, self.f_sm, self.f_mid, self.f_tiny)
            if enabled:
                for e in events:
                    s.handle(e)
            y += 112
        return y

    # ----------------------------------------------------------- screens
    def s_title(self, ev):
        self.screen.fill(BG)
        draw_phantom(self.screen, 960, 250, self.t, 2.1)
        draw_manager(self.screen, 1170, 470, 1.7)
        text(self.screen, "FRAUD PATROL", self.f_huge, ACCENT, 80, 80)
        text(self.screen, "Clean the data. Catch the fraudsters.", self.f_big, WHITE, 80, 146)
        text(self.screen,
             "A fraudster called THE PHANTOM is attacking your bank.\n\n"
             "Days 1-2: clean the payment ledger with pandas (duplicates and missing values).\n"
             "Day 3: remove outliers with the IQR rule.\n"
             "Days 4-6: catch fraud with z-scores.\n\n"
             "Every choice you make really runs on the data. Missed fraud costs MONEY. Hurting honest "
             "customers costs TRUST. Survive all 6 days!",
             self.f_txt, WHITE, 80, 205, width=760)
        text(self.screen, "Your name:", self.f_mid, ACCENT, 80, 440)
        pygame.draw.rect(self.screen, PANEL2, (80, 470, 380, 46), border_radius=8)
        text(self.screen, self.player + ("|" if int(self.t * 2) % 2 == 0 else ""), self.f_mid, WHITE, 94, 481)
        go = Button((80, 545, 260, 60), "START", self.f_big, GREEN, enabled=bool(self.player.strip()))
        go.draw(self.screen)
        for e in ev:
            if e.type == pygame.KEYDOWN:
                if e.key == pygame.K_BACKSPACE:
                    self.player = self.player[:-1]
                elif e.key == pygame.K_RETURN and self.player.strip():
                    self.start_day()
                elif e.unicode and e.unicode.isprintable() and len(self.player) < 20:
                    self.player += e.unicode
            if go.hit(e):
                self.start_day()

    def s_intro(self, ev):
        self.screen.fill(BG)
        self.hud()
        self.main_view()
        self.controls(ev, enabled=False)
        who, msg = F.DAYS[self.day]["intro"][self.line_i]
        self.chars += 1.5
        done = self.speech(who, msg)
        if done:
            text(self.screen, "click to continue >", self.f_tiny, MUTED, 760, 712)
        for e in ev:
            if e.type == pygame.MOUSEBUTTONDOWN and e.button == 1:
                if not done:
                    self.chars = len(msg) + 30
                elif self.line_i < len(F.DAYS[self.day]["intro"]) - 1:
                    self.line_i += 1
                    self.chars = 0
                else:
                    self.state = "play"

    def s_play(self, ev):
        self.screen.fill(BG)
        self.hud()
        self.controls(ev)
        self.main_view()
        hint = Button((900, 486, 360, 48), "Show a hint" if not self.hint_on else "Hint shown below",
                      self.f_mid, PURPLE, enabled=not self.hint_on)
        hint.draw(self.screen)
        run = Button((900, 560, 360, 64), F.DAYS[self.day]["run"], self.f_big, GREEN)
        run.draw(self.screen)
        msg = F.DAYS[self.day]["hint"] if self.hint_on else \
            "Try every option and watch the table / plot change. When you're ready, press the green button."
        self.speech("manager", msg, typing=False)
        for e in ev:
            if hint.hit(e):
                self.hint_on = True
            elif run.hit(e):
                self.run_day()

    def s_result(self, ev):
        self.screen.fill(BG)
        self.hud()
        self.controls(ev, enabled=False)
        self.main_view(reveal=True)
        r = self.result
        mood = {3: "happy", 2: "neutral", 1: "angry", 0: "angry"}[r["stars"]]
        stars = "*" * r["stars"] + "-" * (3 - r["stars"])
        msg = "\n".join(r["lines"]) + f"\nMoney lost: {rs(r['loss'])}    Trust lost: {r['trust_loss']}%    Rating: {stars}"
        self.chars += 2.5
        self.speech("manager", msg, mood=mood)
        if F.DAYS[self.day]["kind"] in ("plot", "iqr"):
            for i, (c, l) in enumerate([(GREEN, "green ring = fraud caught"), (RED, "red X = fraud missed"),
                                        (ACCENT, "orange ring = honest payment hit")]):
                pygame.draw.circle(self.screen, c, (912, 380 + i * 24), 7, 3)
                text(self.screen, l, self.f_sm, WHITE, 928, 371 + i * 24)
        dead = self.money <= 0 or self.trust <= 0
        if dead:
            text(self.screen, "The bank can't survive this!", self.f_mid, RED, 900, 455)
        learn = Button((900, 486, 360, 56), "What did I learn?", self.f_mid, BLUE)
        again = Button((900, 552, 360, 50), "Play this day again", self.f_mid, (90, 96, 112), enabled=not dead)
        learn.draw(self.screen)
        again.draw(self.screen)
        for e in ev:
            if learn.hit(e):
                self.state = "learned"
            elif again.hit(e):
                self.start_day(replay=True)

    def s_learned(self, ev):
        self.screen.fill(BG)
        self.hud()
        sh = pygame.Surface((W, H), pygame.SRCALPHA)
        sh.fill((0, 0, 0, 120))
        self.screen.blit(sh, (0, 0))
        r = pygame.Rect(160, 130, 960, 440)
        pygame.draw.rect(self.screen, CARD, r, border_radius=16)
        title, body, code = F.DEBRIEF[self.day]
        text(self.screen, "WHAT YOU LEARNED  -  " + F.DAYS[self.day]["concept"], self.f_sm, (110, 110, 110),
             r.x + 40, r.y + 24)
        y = text(self.screen, title, self.f_big, INK, r.x + 40, r.y + 50)
        y = text(self.screen, body, self.f_txt, INK, r.x + 40, y + 14, width=r.w - 80)
        lines = code.split("\n")
        pygame.draw.rect(self.screen, CODE_BG, (r.x + 40, y + 16, r.w - 80, len(lines) * 23 + 20), border_radius=8)
        for i, l in enumerate(lines):
            self.screen.blit(self.f_code.render(l, True, CODE_FG), (r.x + 56, y + 27 + i * 23))
        dead = self.money <= 0 or self.trust <= 0
        last = self.day == len(F.DAYS)
        nxt = Button((r.right - 300, r.bottom - 80, 260, 56), "See the ending" if (dead or last) else "Next day >",
                     self.f_mid, GREEN)
        nxt.draw(self.screen)
        for e in ev:
            if nxt.hit(e):
                if dead or last:
                    self.finish(won=not dead)
                else:
                    self.day += 1
                    self.start_day()

    def s_end(self, ev):
        self.screen.fill(BG)
        if self.won:
            text(self.screen, "THE BANK IS SAFE!", self.f_huge, GREEN, 80, 70)
            draw_manager(self.screen, 1060, 230, 2.0, "happy")
        else:
            text(self.screen, "THE PHANTOM WINS...", self.f_huge, RED, 80, 70)
            draw_phantom(self.screen, 1060, 230, self.t, 2.4)
        total = sum(self.stars.values())
        top = 3 * len(F.DAYS)
        rank = "Fraud Hunter Legend" if total >= top - 2 else "Senior Analyst" if total >= top - 5 else \
            "Analyst" if total >= top // 2 else "Trainee"
        lines = [f"Analyst: {self.player}      Rank: {rank}  ({total}/{top} stars)",
                 f"Money left: {rs(max(self.money, 0))}      Trust left: {max(self.trust, 0):.0f}%",
                 "Stars: " + "   ".join(f"Day {d}: {'*' * s or '0'}" for d, s in sorted(self.stars.items())),
                 f"Attempts (including replays): {len(self.runs)}"]
        text(self.screen, "\n".join(lines), self.f_mid, WHITE, 80, 160)
        text(self.screen, "What you learned:", self.f_sm, ACCENT, 80, 300)
        for i, d in enumerate(sorted(self.stars)):
            text(self.screen, f"Day {d}: {F.DAYS[d]['concept']}", self.f_sm, MUTED, 80, 322 + i * 22)
        text(self.screen, "Results saved to playtest_results.csv and playtest_runs.csv", self.f_sm, MUTED, 80, 460)
        again = Button((80, 640, 240, 56), "Play again", self.f_mid, BLUE)
        quit_ = Button((340, 640, 160, 56), "Quit", self.f_mid, (100, 104, 118))
        again.draw(self.screen)
        quit_.draw(self.screen)
        for e in ev:
            if again.hit(e):
                self.reset()
            elif quit_.hit(e):
                pygame.quit()
                sys.exit()

    # ----------------------------------------------------------- playtest evidence
    def save_logs(self):
        if self.saved:
            return
        self.saved = True
        summary = [{"time": datetime.now().isoformat(timespec="seconds"), "player": self.player,
                    "result": "bank safe" if self.won else "phantom wins", "total_stars": sum(self.stars.values()),
                    "stars_per_day": " ".join(f"D{d}:{s}" for d, s in sorted(self.stars.items())),
                    "attempts": len(self.runs), "money_left": round(max(self.money, 0)),
                    "trust_left": max(self.trust, 0)}]
        for name, rows in (("playtest_results.csv", summary), ("playtest_runs.csv", self.runs)):
            if not rows:
                continue
            path = os.path.join(HERE, name)
            new = not os.path.exists(path)
            with open(path, "a", newline="", encoding="utf-8") as fh:
                wr = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
                if new:
                    wr.writeheader()
                wr.writerows(rows)

    # ----------------------------------------------------------- loop
    def step(self, ev):
        self.t += 1 / FPS
        {"title": self.s_title, "intro": self.s_intro, "play": self.s_play, "result": self.s_result,
         "learned": self.s_learned, "end": self.s_end}[self.state](ev)

    def run(self):
        while True:
            ev = pygame.event.get()
            for e in ev:
                if e.type == pygame.QUIT:
                    pygame.quit()
                    sys.exit()
            self.step(ev)
            pygame.display.flip()
            self.clock.tick(FPS)


if __name__ == "__main__":
    Game().run()
