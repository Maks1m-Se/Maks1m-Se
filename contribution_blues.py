"""Contribution Blues: the last 384 days of GitHub contributions, played as a 12-bar blues.

A cursor snakes through the contribution graph (down one week, up the next). Each row
is a note of the C blues scale (Sunday high C ... Saturday low C), every day is a swung
eighth, 8 days make a bar, 48 bars make 4 choruses. Darker squares play louder.

Subcommands
  fetch  LOGIN OUT.json         levels of the last 384 days (token in GH_TOKEN, fallback BLUES_PAT)
  render LEVELS.json OUT.mp4    video with sound (needs ffmpeg, numpy, pillow)
  upload VIDEO NAME REPO_ID     upload as a GitHub attachment, print its URL (token in BLUES_PAT)
  readme README.md URL          put the URL between the contribution-blues markers
"""
import bisect
import datetime as dt
import json
import os
import re
import subprocess
import sys
import urllib.error
import urllib.parse
import urllib.request
import wave

import numpy as np

# ---------- music ----------
BPM, SWING, BARS, DAYS_PER_BAR = 152, 0.57, 48, 8
DAYS = BARS * DAYS_PER_BAR                        # 384 days = 4 choruses of 12 bars
BEAT, BAR = 60 / BPM, 240 / BPM
PROG = "C C C C F F C C G F C G".split()
ROOT = {"C": 36, "F": 41, "G": 43}
FIGURE = [7, 7, 9, 9, 7, 7, 9, 9]                 # left hand over the root, per eighth: G G | A A | G G | A A
LH_VEL = (0.21, 0.16)                             # left hand loudness: root, upper note
SCALE = [60, 63, 65, 66, 67, 70, 72]              # C blues scale, Saturday (bottom row) ... Sunday (top row)
VEL = [0, 0.07, 0.105, 0.135, 0.155]              # loudness per level (top kept modest for small speakers)
BRIGHT = [0, 0.55, 0.75, 0.95, 1.05]              # softer notes are darker, like a real piano
OCTAVE = 0.035                                    # level 4 adds a soft octave above
ROOT_BODY = 0.3                                   # overtones on the left-hand root, so it is heard clearly
FINAL_BASS, FINAL_OCT, FINAL_VEL = 0.055, 0.035, 0.021  # final C9: low C, its octave, the other notes
RING, ROLL = 7.0, 0.045                           # the final chord rings 7 s, rolled upwards 45 ms per note
TARGET_RMS = -20.0                                # average loudness in dBFS; peaks are limited to -1.5 dBFS
SR, FPS, LEAD = 48000, 30, 0.6
TAIL = RING + 4 * ROLL + 0.6                      # the final chord rings after the last bar


def slot_off(s):
    return (s // 2) * BEAT + (SWING * BEAT if s % 2 else 0)


def slot_dur(s):
    return ((1 - SWING) if s % 2 else SWING) * BEAT


def score(levels, end):
    """One entry per day in playback order (the snake through the graph)."""
    start = end - dt.timedelta(days=DAYS - 1)
    first_sun = start - dt.timedelta(days=(start.weekday() + 1) % 7)
    days = [start + dt.timedelta(days=i) for i in range(DAYS)]
    col = lambda d: (d - first_sun).days // 7
    row = lambda d: (d.weekday() + 1) % 7         # GitHub: Sunday on top
    order = []
    for c in range(col(end) + 1):
        week = [d for d in days if col(d) == c]
        order += week[::-1] if c % 2 else week
    slots, quiet = [], 0
    for p, d in enumerate(order):
        k, s = divmod(p, DAYS_PER_BAR)
        lv = max(0, min(4, int(levels.get(d.isoformat(), 0))))
        slots.append(dict(day=d, t=LEAD + k * BAR + slot_off(s), bar=k, slot=s, lv=lv,
                          ghost=lv > 0 and quiet > 7,     # first note after more than 7 silent days
                          midi=SCALE[6 - row(d)], col=col(d), row=row(d), chord=PROG[k % 12]))
        quiet = 0 if lv else quiet + 1
    return slots


# ---------- sound: a port of the lab's Web Audio synth ----------
def hz(m):
    return 440 * 2 ** ((m - 69) / 12)


_TRI = {}


def triangle(m, n):
    """Band-limited triangle like Web Audio's, n samples from phase 0."""
    if m not in _TRI:
        k = np.arange(1, int(20000 // hz(m)) + 1, 2)
        w = ((8 / np.pi ** 2) * (-1.0) ** ((k - 1) // 2) / k ** 2) @ np.sin(2 * np.pi * np.outer(k, np.arange(4096) / 4096))
        w /= np.abs(w).max()
        _TRI[m] = np.append(w, w[0])
    ph = (hz(m) * np.arange(n) / SR) % 1 * 4096
    i = ph.astype(int)
    return _TRI[m][i] * (i + 1 - ph) + _TRI[m][i + 1] * (ph - i)


def envelope(n, attack, peak, end):
    """setValueAtTime(1e-4) -> exponential ramp to peak -> exponential ramp to 1e-4 at `end`."""
    t = np.arange(n) / SR
    e = np.full(n, 1e-4)
    a, d = t < attack, (t >= attack) & (t < end)
    e[a] = 1e-4 * (peak / 1e-4) ** (t[a] / attack)
    e[d] = peak * (1e-4 / peak) ** ((t[d] - attack) / (end - attack))
    return e


def add(buf, t0, sig):
    i = int(round(t0 * SR))
    buf[i:i + len(sig)] += sig[:len(buf) - i]


def piano(buf, m, t0, vel, length=0.9, bright=1.0):
    n = int((length + 0.05) * SR)
    w = 2 * np.pi * hz(m) * np.arange(n) / SR
    sig = triangle(m, n) + 0.35 * bright * np.sin(2 * w) + 0.12 * bright * np.sin(3 * w)
    add(buf, t0, sig * envelope(n, 0.006, vel, length))


def bass(buf, m, t0, dur, vel, body=0.0):
    """Left-hand note: a triangle; `body` adds overtones so the pitch also comes through small speakers."""
    n = int((dur + 0.05) * SR)
    sig = triangle(m, n)
    if body:
        w = 2 * np.pi * hz(m) * np.arange(n) / SR
        sig = sig + body * (np.sin(2 * w) + 0.5 * np.sin(3 * w) + 0.2 * np.sin(4 * w))
    add(buf, t0, sig * envelope(n, 0.01, vel, max(0.05, dur * 0.95)))


def compress(x, thr=-10.0, knee=12.0, ratio=3.0, att=0.005, rel=0.2, look=0.006, block=32):
    """Gentle peak compressor modelled on Web Audio's DynamicsCompressor (incl. its makeup gain)."""
    def curve(db):                                # static curve: soft knee from thr to thr + knee
        y = np.where(db < thr, db, db + (1 / ratio - 1) * (db - thr) ** 2 / (2 * knee))
        hi = db > thr + knee
        return np.where(hi, thr + knee / 2 + knee / (2 * ratio) + (db - thr - knee) / ratio, y)
    nb = -(-len(x) // block)
    peaks = np.abs(np.pad(x, (0, nb * block - len(x)))).reshape(nb, block).max(1)
    a_att, a_rel = np.exp(-block / (att * SR)), np.exp(-block / (rel * SR))
    env, e = np.empty(nb), 0.0
    for i, p in enumerate(peaks):
        e = (a_att if p > e else a_rel) * e + (1 - (a_att if p > e else a_rel)) * p
        env[i] = e
    db = 20 * np.log10(np.maximum(env, 1e-9))
    gain_db = curve(db) - db - 0.6 * float(curve(np.array(0.0)))     # makeup: (1 / full-range gain) ** 0.6
    centres = (np.arange(nb) + 0.5) * block - look * SR              # look ahead like the browser's pre-delay
    return x * 10 ** (np.interp(np.arange(len(x)), centres, gain_db) / 20)


def render_audio(slots, left=True, right=True):
    te = LEAD + BARS * BAR
    buf = np.zeros(int((te + TAIL) * SR))
    for sl in slots:
        root, dur = ROOT[sl["chord"]], slot_dur(sl["slot"])
        if left:
            bass(buf, root, sl["t"], dur, LH_VEL[0], ROOT_BODY)
            bass(buf, root + FIGURE[sl["slot"]], sl["t"], dur, LH_VEL[1])
        lv, m = sl["lv"], sl["midi"]
        if lv and right:
            piano(buf, m, sl["t"], VEL[lv], 0.9, BRIGHT[lv])
            if lv == 4:
                piano(buf, m + 12, sl["t"], OCTAVE, 0.7, 0.35)
            if sl["ghost"]:
                piano(buf, m - 1, sl["t"] - BEAT / 4, 0.04, 0.16, 0.4)
    if left and right:
        bass(buf, 36, te, RING, FINAL_BASS, 1.0)  # final C9: a clear low C with its octave ...
        piano(buf, 48, te, FINAL_OCT, RING)
        for k, m in enumerate((64, 67, 70, 74)):  # ... then E G Bb D, rolled upwards
            piano(buf, m, te + ROLL * (k + 1), FINAL_VEL, RING)
    if not (left and right):
        return buf                                # a single hand, unmastered (for checks)
    out = compress(0.8 * buf)
    out *= 10 ** (TARGET_RMS / 20) / np.sqrt(np.mean(out ** 2))
    return limit(out)


def limit(x, ceiling_db=-1.5, look=0.005, rel=0.12, block=32):
    """Look-ahead peak limiter: only the loudest moments are turned down, quickly and smoothly."""
    ceiling = 10 ** (ceiling_db / 20)
    nb = -(-len(x) // block)
    peaks = np.abs(np.pad(x, (0, nb * block - len(x)))).reshape(nb, block).max(1)
    need = np.minimum(1.0, ceiling / np.maximum(peaks, 1e-9))
    L = max(1, round(look * SR / block))
    ahead = np.lib.stride_tricks.sliding_window_view(np.pad(need, (1, L), constant_values=1.0), L + 2).min(1)
    a, g, cur = np.exp(-block / (rel * SR)), np.empty(nb), 1.0
    for i, v in enumerate(ahead):                 # instant attack, smooth release
        cur = v if v < cur else a * cur + (1 - a) * v
        g[i] = cur
    y = x * np.interp(np.arange(len(x)), (np.arange(nb) + 0.5) * block, g)
    return np.clip(y, -ceiling, ceiling)


def write_wav(path, x):
    pcm = (np.clip(x, -1, 1) * 32767).astype("<i2")
    with wave.open(path, "wb") as w:
        w.setnchannels(2)
        w.setsampwidth(2)
        w.setframerate(SR)
        w.writeframes(np.repeat(pcm, 2).tobytes())


# ---------- picture: GitHub's dark contribution graph ----------
S = 2                                             # render scale (crisp on retina screens)
CELL, GAP = 10, 3
STEP = CELL + GAP
M, HEAD, PAD, LABEL_W, MONTH_H, STATUS_H = 16, 44, 14, 30, 20, 40
MONTHS = "Jan Feb Mar Apr May Jun Jul Aug Sep Oct Nov Dec".split()
BG, FG, MUTED, BORDER, CURSOR = "#0d1117", "#f0f6fc", "#9198a1", "#3d444d", "#4493f8"
LV = ["#151b23", "#033a16", "#196c2e", "#2ea043", "#56d364"]
POP = [1, 1.33, 1.46, 1.59, 1.72]                 # how far a square jumps when its note sounds
HINT = "Play with sound on"
FONTS = {"sans": ["/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf",
                  "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"],
         "serif": ["/usr/share/fonts/truetype/liberation/LiberationSerif-BoldItalic.ttf",
                   "/usr/share/fonts/truetype/dejavu/DejaVuSerif-BoldItalic.ttf"]}


def rgb(h):
    return tuple(int(h[i:i + 2], 16) for i in (1, 3, 5))


def mix(a, b, t):
    return tuple(round(x + (y - x) * t) for x, y in zip(rgb(a) if isinstance(a, str) else a, rgb(b)))


def font(kind, px):
    from PIL import ImageFont
    for path in FONTS[kind]:
        if os.path.exists(path):
            return ImageFont.truetype(path, px)
    return ImageFont.load_default(px)


class Picture:
    def __init__(self, slots, end):
        from PIL import Image, ImageDraw
        self.Image, self.slots = Image, slots
        self.times = [sl["t"] for sl in slots]
        ncols = slots and max(sl["col"] for sl in slots) + 1
        self.gx, self.gy = M + PAD + LABEL_W, HEAD + PAD + MONTH_H
        self.W = M + PAD + LABEL_W + ncols * STEP - GAP + PAD + M
        box_bottom = self.gy + 7 * STEP - GAP + 12 + 14 + 12
        self.H = box_bottom + STATUS_H
        self.status_y = box_bottom + 25
        start = end - dt.timedelta(days=DAYS - 1)
        year_ago = end - dt.timedelta(days=364)
        notes = sum(1 for sl in slots if sl["lv"] and sl["day"] >= year_ago)
        self.idle = (f"12-bar blues · {BARS} bars · {BPM} bpm",
                     f"{start:%d %b %Y} – {end:%d %b %Y}".lstrip("0").replace(" 0", " "))

        Z = 2 * S                                 # draw the still parts at 4x, then shrink for clean edges
        img = Image.new("RGB", (self.W * Z, self.H * Z), BG)
        d = ImageDraw.Draw(img)
        f12, f16, fdyn = font("sans", 12 * Z), font("sans", 16 * Z), font("serif", 13 * Z)
        d.text((M * Z, 27 * Z), f"{notes} notes in the last year", font=f16, fill=FG, anchor="ls")
        d.rounded_rectangle([M * Z, HEAD * Z, (self.W - M) * Z - 1, box_bottom * Z - 1], 6 * Z,
                            outline=BORDER, width=Z)
        first_sun = start - dt.timedelta(days=(start.weekday() + 1) % 7)
        labels = []
        for c in range(ncols):
            sun = first_sun + dt.timedelta(days=7 * c)
            if c == 0 or sun.month != (sun - dt.timedelta(days=7)).month:
                if labels and c - labels[-1][0] < 3:
                    labels.pop()                  # too close: keep the newer month
                labels.append((c, MONTHS[sun.month - 1]))
        for c, name in labels:
            d.text(((self.gx + c * STEP) * Z, (self.gy - 8) * Z), name, font=f12, fill=FG, anchor="ls")
        for r, name in ((1, "Mon"), (3, "Wed"), (5, "Fri")):
            d.text(((M + PAD) * Z, (self.gy + r * STEP + CELL - 1) * Z), name, font=f12, fill=FG, anchor="ls")
        for sl in slots:
            self._cell(d, self.gx + sl["col"] * STEP, self.gy + sl["row"] * STEP, sl["lv"], Z)
        ly = self.gy + 7 * STEP - GAP + 12 + 11
        right = self.W - M - PAD
        d.text((right * Z, ly * Z), "ff", font=fdyn, fill=MUTED, anchor="rs")
        x0 = right - 16 - 5 * STEP + GAP
        for k in range(5):
            self._cell(d, x0 + k * STEP, ly - CELL + 1, k, Z)
        d.text(((x0 - 5) * Z, ly * Z), "pp", font=fdyn, fill=MUTED, anchor="rs")
        hint_w = 13 + 26 + round(font("sans", 12).getlength(HINT)) + 14
        self.hint_box = (self.W - M - hint_w, 8, self.W - M, 36)
        self.base = img.resize((self.W * S, self.H * S), Image.Resampling.LANCZOS)
        self.hints = []
        for colour in (CURSOR, MUTED):            # "play with sound on", first bright, later quiet
            h = img.copy()
            self._hint(ImageDraw.Draw(h), colour, Z)
            box = tuple(v * Z for v in self.hint_box)
            self.hints.append(h.crop(box).resize(((box[2] - box[0]) // 2, (box[3] - box[1]) // 2),
                                                 Image.Resampling.LANCZOS))
        self.f_status = font("sans", 12 * S)
        self.sprites = {}

    @staticmethod
    def _cell(d, x, y, lv, Z):
        d.rounded_rectangle([x * Z, y * Z, (x + CELL) * Z - 1, (y + CELL) * Z - 1], 2 * Z,
                            fill=LV[lv], outline=mix(LV[lv], FG, 0.06), width=Z)

    def _hint(self, d, colour, Z):
        x0, y0, x1, y1 = (v * Z for v in self.hint_box)
        d.rounded_rectangle([x0, y0, x1 - 1, y1 - 1], (y1 - y0) // 2, outline=colour, width=Z)
        cy, sx = (y0 + y1) // 2, x0 + 13 * Z      # speaker icon with two sound waves
        d.polygon([(sx, cy - 3 * Z), (sx + 4 * Z, cy - 3 * Z), (sx + 9 * Z, cy - 7 * Z), (sx + 9 * Z, cy + 7 * Z),
                   (sx + 4 * Z, cy + 3 * Z), (sx, cy + 3 * Z)], fill=colour)
        for r in (5, 9):
            d.arc([sx + 9 * Z - r * Z, cy - r * Z, sx + 9 * Z + r * Z + 2 * Z, cy + r * Z], -50, 50, fill=colour, width=Z * 3 // 2)
        d.text((sx + 26 * Z, cy + 4 * Z), HINT, font=font("sans", 12 * Z), fill=colour, anchor="ls")

    def _sprite(self, lv, scale, ring):
        """A square (or the cursor ring) scaled around its centre, antialiased, cached."""
        key = (lv, round(scale, 2), ring)
        if key not in self.sprites:
            from PIL import ImageDraw
            Z, half = 4, (CELL * key[1] / 2 + 2) * S
            n = 2 * int(half + 1)
            big = self.Image.new("RGBA", (n * Z, n * Z), rgb(CURSOR if ring else LV[lv]) + (0,))
            d, c, h = ImageDraw.Draw(big), n * Z / 2, CELL * key[1] * S * Z / 2
            w = S * Z * key[1]
            if ring:                              # 2 px stroke centred on the square's edge
                d.rounded_rectangle([c - h - w, c - h - w, c + h + w, c + h + w], 3 * w, outline=CURSOR + "ff", width=int(2 * w))
            else:
                d.rounded_rectangle([c - h, c - h, c + h, c + h], 2 * w, fill=LV[lv] + "ff",
                                    outline=mix(LV[lv], FG, 0.06) + (255,), width=int(w))
            self.sprites[key] = big.resize((n, n), self.Image.Resampling.BOX)
        return self.sprites[key]

    def _paste(self, img, sl, sprite, alpha=1.0):
        if alpha < 1:
            sprite = sprite.copy()
            sprite.putalpha(sprite.getchannel("A").point(lambda v: int(v * alpha)))
        cx, cy = (self.gx + sl["col"] * STEP + CELL / 2) * S, (self.gy + sl["row"] * STEP + CELL / 2) * S
        img.paste(sprite, (int(cx - sprite.width / 2), int(cy - sprite.height / 2)), sprite)

    def frame(self, t):
        from PIL import ImageDraw
        img = self.base.copy()
        fade = min(1.0, max(0.0, (t - 8) / 1.0))
        hint = self.hints[0] if fade == 0 else self.hints[1] if fade == 1 else \
            self.Image.blend(self.hints[0], self.hints[1], fade)
        img.paste(hint, (self.hint_box[0] * S, self.hint_box[1] * S))
        p = bisect.bisect_right(self.times, t) - 1
        last = len(self.slots) - 1
        end_t = LEAD + BARS * BAR
        for q in range(max(0, p - 3), p + 1):     # squares jump when their note sounds
            sl = self.slots[q]
            u = (t - sl["t"]) / 0.4
            if sl["lv"] and u < 1:
                self._paste(img, sl, self._sprite(sl["lv"], 1 + (POP[sl["lv"]] - 1) * (1 - u) ** 2, False))
        for q in range(max(0, p - 3), p + 1):     # cursor, with a short fading tail
            leave = self.times[q + 1] if q < last else end_t
            alpha = 1.0 if t < leave else 1 - (t - leave) / 0.3
            if alpha > 0:
                sl, u = self.slots[q], (t - self.slots[q]["t"]) / 0.4
                scale = 1 + (POP[sl["lv"]] - 1) * (1 - u) ** 2 if sl["lv"] and u < 1 else 1
                self._paste(img, sl, self._sprite(0, scale, True), alpha)
        d = ImageDraw.Draw(img)
        if end_t <= t < end_t + 4:
            left, right = "C9", self.idle[1]
        elif 0 <= p and t < end_t:
            sl = self.slots[p]
            left = f"{sl['chord']}7 · chorus {sl['bar'] // 12 + 1} of 4 · bar {sl['bar'] % 12 + 1}"
            right = f"{sl['day']:%a, %d %b %Y}".replace(" 0", " ")
        else:
            left, right = self.idle
        d.text(((M + 2) * S, self.status_y * S), left, font=self.f_status, fill=MUTED, anchor="ls")
        d.text(((self.W - M - 2) * S, self.status_y * S), right, font=self.f_status, fill=MUTED, anchor="rs")
        return img


def render(levels_path, out_path):
    with open(levels_path, encoding="utf-8") as f:
        data = json.load(f)
    levels = data.get("levels", data)
    end = dt.date.fromisoformat(data.get("end") or data.get("to") or max(levels))
    slots = score(levels, end)
    audio = render_audio(slots)
    wav = out_path + ".wav"
    write_wav(wav, audio)
    pic = Picture(slots, end)
    n = int(len(audio) / SR * FPS)
    cmd = ["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24",
           "-s", f"{pic.W * S}x{pic.H * S}", "-r", str(FPS), "-i", "-", "-i", wav,
           "-c:v", "libx264", "-preset", "slow", "-crf", "20", "-tune", "animation", "-pix_fmt", "yuv420p",
           "-c:a", "aac", "-b:a", "160k", "-movflags", "+faststart", "-shortest", out_path]
    ff = subprocess.Popen(cmd, stdin=subprocess.PIPE)
    for i in range(n):
        ff.stdin.write(pic.frame(i / FPS).tobytes())
    ff.stdin.close()
    if ff.wait():
        sys.exit("ffmpeg failed")
    os.remove(wav)
    mb = os.path.getsize(out_path) / 1e6
    print(f"wrote {out_path}: {n / FPS:.1f} s, {mb:.1f} MB, {sum(1 for s in slots if s['lv'])} notes")
    if mb > 10:
        sys.exit("video is larger than GitHub's 10 MB limit")


# ---------- GitHub ----------
def api(url, token, body, content_type="application/json"):
    req = urllib.request.Request(url, data=body, headers={
        "Authorization": f"Bearer {token}", "Content-Type": content_type, "Accept": "application/json"})
    try:
        with urllib.request.urlopen(req) as r:
            return json.load(r)
    except urllib.error.HTTPError as e:
        raise RuntimeError(f"HTTP {e.code} from {url.split('?')[0]}: {e.read().decode(errors='replace')[:500]}")


CAL = "contributionCalendar { totalContributions weeks { contributionDays { date contributionCount contributionLevel } } }"
LEVEL = {"NONE": 0, "FIRST_QUARTILE": 1, "SECOND_QUARTILE": 2, "THIRD_QUARTILE": 3, "FOURTH_QUARTILE": 4}


def calendar(login, token, span=None):
    args = '(from: "%sT00:00:00Z", to: "%sT23:59:59Z")' % span if span else ""
    query = 'query { user(login: "%s") { contributionsCollection%s { %s } } }' % (login, args, CAL)
    res = api("https://api.github.com/graphql", token, json.dumps({"query": query}).encode())
    if "errors" in res:
        raise RuntimeError(f"GitHub API error: {res['errors']}")
    cal = res["data"]["user"]["contributionsCollection"]["contributionCalendar"]
    days = {d["date"]: d for w in cal["weeks"] for d in w["contributionDays"]}
    return days, cal["totalContributions"]


def fetch(login, out_path):
    tokens = [(k, os.environ[k]) for k in ("GH_TOKEN", "BLUES_PAT") if os.environ.get(k)]
    for name, token in tokens:                    # the Actions token shows what visitors see; the PAT is a fallback
        try:
            year, total = calendar(login, token)  # exactly what the profile shows
            break
        except RuntimeError as e:
            print(f"{name} could not read the calendar: {e}")
    else:
        raise RuntimeError("no token could read the contribution calendar")
    end = dt.date.fromisoformat(max(year))
    start = end - dt.timedelta(days=DAYS - 1)
    first = dt.date.fromisoformat(min(year))
    older = calendar(login, token, (start.isoformat(), (first - dt.timedelta(days=1)).isoformat()))[0] if first > start else {}
    # GitHub's colours are quartiles of this year, so older days get this year's thresholds
    floors = sorted({(min(d["contributionCount"] for d in year.values() if LEVEL[d["contributionLevel"]] == lv), lv)
                     for lv in range(1, 5) if any(LEVEL[d["contributionLevel"]] == lv for d in year.values())})
    def level(count):
        return 0 if count == 0 else max([lv for floor, lv in floors if count >= floor] or [1])
    levels = {}
    for i in range(DAYS):
        day = (start + dt.timedelta(days=i)).isoformat()
        if day in year:
            levels[day] = LEVEL[year[day]["contributionLevel"]]
        else:
            levels[day] = level(older.get(day, {"contributionCount": 0})["contributionCount"])
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump({"end": end.isoformat(), "levels": levels}, f)
    active = sum(1 for d in year.values() if d["contributionCount"])
    print(f"read with {name}: {total} contributions in the last year on {active} days (as above your profile graph)")
    print(f"the piece: {sum(1 for v in levels.values() if v)} notes from {start} to {end}")


def upload(video, name, repo_id):
    url = "https://uploads.github.com/user-attachments/assets?" + urllib.parse.urlencode(
        {"name": name, "content_type": "video/mp4", "repository_id": repo_id})
    with open(video, "rb") as f:
        res = api(url, os.environ["BLUES_PAT"], f.read(), "video/mp4")
    print(res["url"])


START, END = "<!-- contribution-blues:start -->", "<!-- contribution-blues:end -->"


def readme(path, url):
    if not url.startswith("https://github.com/user-attachments/"):
        sys.exit(f"not a GitHub attachment URL: {url!r}")
    with open(path, encoding="utf-8") as f:
        text = f.read()
    if START not in text or END not in text:
        sys.exit(f"Put these two lines into {path} where the video should go:\n{START}\n{END}")
    new = re.sub(re.escape(START) + ".*?" + re.escape(END), f"{START}\n\n{url}\n\n{END}", text, flags=re.S)
    with open(path, "w", encoding="utf-8") as f:
        f.write(new)
    print("README updated" if new != text else "README unchanged")


if __name__ == "__main__":
    commands = {"fetch": (fetch, 2), "render": (render, 2), "upload": (upload, 3), "readme": (readme, 2)}
    if len(sys.argv) < 2 or sys.argv[1] not in commands or len(sys.argv) != commands[sys.argv[1]][1] + 2:
        sys.exit(__doc__)
    try:
        commands[sys.argv[1]][0](*sys.argv[2:])
    except RuntimeError as e:
        sys.exit(str(e))
