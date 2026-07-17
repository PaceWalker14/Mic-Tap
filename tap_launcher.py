#!/usr/bin/env python3
"""
Tap Launcher — listens to your microphone for finger taps and runs actions.

Double-tap or triple-tap near the mic to launch the apps you assign.
Optional (experimental): drag your finger along the desk toward / away from
the mic to scroll.

All audio is processed live, in memory. Nothing is recorded or saved.

Run:  python tap_launcher.py
Deps: pip install numpy scipy sounddevice pyautogui
"""

import json
import os
import queue
import subprocess
import sys
import threading
import time
from collections import deque

import numpy as np

try:
    from scipy import signal as sps
except ImportError:
    sps = None

try:
    import sounddevice as sd
except Exception:
    sd = None

try:
    import pyautogui
except Exception:
    pyautogui = None

try:
    import tkinter as tk
    from tkinter import filedialog, ttk
except Exception:
    tk = None


# ---------------------------------------------------------------- constants

SAMPLE_RATE = 44100
BLOCK = 512                  # ~11.6 ms per block

TAP_MAX_DUR = 0.12           # a burst shorter than this is a tap
DRAG_MIN_DUR = 0.25          # a burst longer than this is a drag
EVENT_ABORT_DUR = 3.0        # anything longer is background noise
REFRACTORY = 0.06            # dead time after an event ends
RELEASE_RATIO = 0.5          # event ends when level falls below thr * this
NOISE_ALPHA = 0.02           # how fast the noise floor adapts
MIN_NOISE = 1e-5
SPEC_N = 4096                # samples analysed around a transient's peak
REJECT_MARGIN = 0.05         # bias toward rejecting near-ties with ignores
REJECT_FLOOR = 0.45          # an ignore-sound must at least somewhat match

APP_DIR = os.path.dirname(os.path.abspath(__file__))
CONFIG_PATH = os.path.join(APP_DIR, "tap_launcher_config.json")

DEFAULT_CONFIG = {
    "sensitivity": 4.0,       # threshold = noise floor x sensitivity
    "multi_tap_window": 0.40, # max seconds between taps in one pattern
    "band": None,             # [lo_hz, hi_hz] set by calibration
    "tap_template": None,     # spectral signature of your taps
    "min_match": 0.6,         # similarity a sound needs to count as a tap
    "reject_templates": [],   # signatures of sounds to ignore (keyboard...)
    "apps": {"2": "", "3": "", "4": ""},
    "actions_enabled": False,
    "drag_enabled": False,
    "input_device": None,
}


def load_config():
    cfg = dict(DEFAULT_CONFIG)
    try:
        with open(CONFIG_PATH, "r", encoding="utf-8") as f:
            saved = json.load(f)
        for k in cfg:
            if k in saved:
                cfg[k] = saved[k]
        cfg["apps"] = {**DEFAULT_CONFIG["apps"], **(cfg.get("apps") or {})}
    except Exception:
        pass
    return cfg


def save_config(cfg):
    try:
        with open(CONFIG_PATH, "w", encoding="utf-8") as f:
            json.dump(cfg, f, indent=2)
    except Exception:
        pass


def rms_of(x):
    return float(np.sqrt(np.mean(np.square(x)) + 1e-12))


def to_db(v):
    return 20.0 * np.log10(max(float(v), 1e-7))


# ------------------------------------------------------- sound fingerprints

_BAND_EDGES = np.geomspace(80.0, 16000.0, 49)   # 48 log-spaced bands


def spectral_signature(x, sr=SAMPLE_RATE):
    """
    Fingerprint a transient: the 4096 samples around its loudest point are
    reduced to 48 log-spaced band energies (in dB), mean-centred and
    normalised. Two sounds with the same tonal character give vectors whose
    dot product (correlation) is near 1; a dull finger thud vs a clicky
    keystroke land far apart.
    """
    x = np.asarray(x, dtype=float)
    if len(x) < SPEC_N:
        x = np.pad(x, (0, SPEC_N - len(x)))
    peak = int(np.argmax(np.abs(x)))
    start = min(max(peak - SPEC_N // 4, 0), len(x) - SPEC_N)
    seg = x[start:start + SPEC_N] * np.hanning(SPEC_N)
    power = np.abs(np.fft.rfft(seg)) ** 2
    freqs = np.fft.rfftfreq(SPEC_N, 1.0 / sr)
    sig = np.empty(len(_BAND_EDGES) - 1)
    for i in range(len(sig)):
        m = (freqs >= _BAND_EDGES[i]) & (freqs < _BAND_EDGES[i + 1])
        sig[i] = np.log10(power[m].sum() + 1e-12)
    sig -= sig.mean()
    n = np.linalg.norm(sig)
    return sig / n if n > 0 else sig


def similarity(a, b):
    """Correlation between two signatures, roughly -1..1."""
    return float(np.dot(a, b))


# ---------------------------------------------------------------- filtering

def design_band(lo, hi, sr=SAMPLE_RATE):
    """Butterworth band-pass as second-order sections."""
    lo = max(40.0, float(lo))
    hi = min(sr * 0.45, float(hi))
    if hi - lo < 150.0:
        c = (hi + lo) / 2.0
        lo, hi = max(40.0, c - 100.0), min(sr * 0.45, c + 100.0)
    return sps.butter(4, [lo, hi], btype="bandpass", fs=sr, output="sos")


class StreamFilter:
    """Stateful band-pass so filtering is continuous across blocks."""

    def __init__(self, sos):
        self.sos = sos
        self.zi = np.zeros((sos.shape[0], 2))

    def process(self, x):
        y, self.zi = sps.sosfilt(self.sos, x, zi=self.zi)
        return y


# ----------------------------------------------------------------- detector

class TapDetector:
    """
    Block-by-block energy detector.

    - Learns an adaptive noise floor; the trigger threshold is
      noise_floor x sensitivity.
    - A burst that rises above threshold and falls back quickly is a *tap*.
    - Taps close together in time are grouped into one pattern
      (2 taps = double, 3 = triple, ...).
    - A burst that stays loud for a while is a *drag*; its direction is
      inferred from whether it gets louder (toward the mic) or quieter
      (away from it).

    Emits event dicts through the `emit` callback:
      {'type': 'level', 'rms', 'thr', 'state'}         every block
      {'type': 'tap', 't', 'count_so_far'}
      {'type': 'group', 'n', 't'}
      {'type': 'drag', 'direction', 'duration'}
      {'type': 'ignored', 'duration'}                  between tap and drag
      {'type': 'noise'}                                over-long burst
    """

    def __init__(self, emit, sr=SAMPLE_RATE):
        self.emit = emit
        self.sr = sr
        self.sensitivity = DEFAULT_CONFIG["sensitivity"]
        self.group_window = DEFAULT_CONFIG["multi_tap_window"]

        self._lock = threading.Lock()
        self.filter = None
        self.raw_ring = deque(maxlen=16)      # ~190 ms of raw audio
        self.tap_template = None              # signature of a real tap
        self.reject_templates = []            # signatures to ignore
        self.min_match = DEFAULT_CONFIG["min_match"]
        self.noise = None
        self.warmup = []
        self.state = "idle"
        self.evt_start = 0.0
        self.evt_rms = []
        self.low_count = 0
        self.refract_until = 0.0
        self.taps = []
        self.samples_seen = 0

    # -- configuration -------------------------------------------------

    def set_band(self, band):
        """band: (lo_hz, hi_hz) or None to disable filtering."""
        with self._lock:
            if band and sps is not None:
                self.filter = StreamFilter(design_band(band[0], band[1]))
            else:
                self.filter = None
            self.noise = None          # relearn the floor for the new band
            self.warmup = []
            self.state = "idle"
            self.taps = []

    @property
    def threshold(self):
        n = self.noise if self.noise is not None else 1e-3
        return max(n * self.sensitivity, 5e-4)

    # -- processing ----------------------------------------------------

    def process(self, block):
        with self._lock:
            self._process(block)

    def _process(self, block):
        self.samples_seen += len(block)
        t = self.samples_seen / self.sr

        self.raw_ring.append(np.asarray(block, dtype=float))
        x = self.filter.process(block) if self.filter is not None else block
        rms = rms_of(x)

        # learn the initial noise floor from the first ~0.3 s
        if self.noise is None:
            self.warmup.append(rms)
            if len(self.warmup) >= 25:
                self.noise = max(float(np.median(self.warmup)), MIN_NOISE)
            self.emit({"type": "level", "rms": rms, "thr": None,
                       "state": "learning"})
            return

        thr = self.threshold

        if self.state == "idle":
            if rms < thr * 0.7:
                self.noise = max(
                    (1 - NOISE_ALPHA) * self.noise + NOISE_ALPHA * rms,
                    MIN_NOISE)
            if self.taps and (t - self.taps[-1]) > self.group_window:
                self._finalize_group(t)
            if t >= self.refract_until and rms >= thr:
                self.state = "event"
                self.evt_start = t
                self.evt_rms = [rms]
                self.low_count = 0

        elif self.state == "event":
            self.evt_rms.append(rms)
            dur = t - self.evt_start
            if rms < thr * RELEASE_RATIO:
                self.low_count += 1
            else:
                self.low_count = 0
            if self.low_count >= 2:
                self._end_event(t, dur)
            elif dur > EVENT_ABORT_DUR:
                self.state = "idle"
                self.refract_until = t + REFRACTORY
                self.evt_rms = []
                self.taps = []
                self.emit({"type": "noise"})

        self.emit({"type": "level", "rms": rms, "thr": thr,
                   "state": self.state})

    def _end_event(self, t, dur):
        self.state = "idle"
        self.refract_until = t + REFRACTORY
        if dur <= TAP_MAX_DUR:
            ok, info = self._classify_tap()
            if ok:
                self.taps.append(t)
                self.emit({"type": "tap", "t": t,
                           "count_so_far": len(self.taps), **info})
            else:
                self.emit({"type": "rejected", "t": t, **info})
        elif dur >= DRAG_MIN_DUR:
            if self.taps:
                self._finalize_group(t)
            arr = np.asarray(self.evt_rms, dtype=float)
            third = max(1, len(arr) // 3)
            head = float(arr[:third].mean())
            tail = float(arr[-third:].mean())
            if tail > head * 1.3:
                direction = "toward"
            elif tail < head * 0.75:
                direction = "away"
            else:
                direction = "flat"
            self.emit({"type": "drag", "direction": direction,
                       "duration": dur})
        else:
            self.emit({"type": "ignored", "duration": dur})
        self.evt_rms = []

    def _finalize_group(self, t):
        n = len(self.taps)
        self.taps = []
        self.emit({"type": "group", "n": n, "t": t})

    def _classify_tap(self):
        """
        Compare the just-heard transient's fingerprint to the calibrated
        tap template and to any learned ignore-sounds.

        Returns (accepted, info). Without a template (not calibrated yet)
        every candidate is accepted, as before.
        """
        if self.tap_template is None:
            return True, {}
        win = np.concatenate(list(self.raw_ring))
        sig = spectral_signature(win, self.sr)
        tap_sim = similarity(sig, self.tap_template)
        rej_sim = None
        if self.reject_templates:
            rej_sim = max(similarity(sig, r) for r in self.reject_templates)
        if (rej_sim is not None and rej_sim > REJECT_FLOOR
                and rej_sim >= tap_sim - REJECT_MARGIN):
            return False, {"reason": "ignored", "sim": tap_sim,
                           "rej_sim": rej_sim}
        if tap_sim < self.min_match:
            return False, {"reason": "low-match", "sim": tap_sim,
                           "rej_sim": rej_sim}
        return True, {"sim": tap_sim}


# --------------------------------------------------------------- calibrator

class Calibrator:
    """
    Two-phase calibration, fed raw audio blocks:

      1. ~1.2 s of quiet to measure the noise floor.
      2. Wait for N taps; a short window of audio around each is captured.

    From the captured taps it finds the frequency band where your taps on
    *this* surface put their energy, builds a band-pass around it, and
    suggests a trigger sensitivity that sits between the (filtered) noise
    floor and your tap loudness.
    """

    QUIET_SECS = 1.2
    NEED_TAPS = 5
    TIMEOUT = 18.0

    def __init__(self, sr=SAMPLE_RATE, block=BLOCK):
        self.sr = sr
        self.block = block
        self.phase = "quiet"          # quiet -> taps -> done | failed
        self.status = "Stay quiet for a second..."
        self.noise_blocks = []
        self.ring = deque(maxlen=12)  # ~140 ms of recent audio
        self.windows = []
        self.capture_in = None
        self.refract = 0.0
        self.blocks_seen = 0
        self.noise = None
        self.result = None

    def feed(self, raw):
        self.blocks_seen += 1
        t = self.blocks_seen * self.block / self.sr
        self.ring.append(np.array(raw, dtype=float))

        if self.phase == "quiet":
            self.noise_blocks.append(np.array(raw, dtype=float))
            if t >= self.QUIET_SECS:
                cat = np.concatenate(self.noise_blocks)
                self.noise = max(rms_of(cat), MIN_NOISE)
                self.phase = "taps"
                self.status = (f"Now tap {self.NEED_TAPS} times on the "
                               f"surface, near the mic")

        elif self.phase == "taps":
            rms = rms_of(raw)
            if self.capture_in is not None:
                self.capture_in -= 1
                if self.capture_in <= 0:
                    self.windows.append(np.concatenate(list(self.ring)))
                    self.capture_in = None
                    self.status = (f"Heard {len(self.windows)} of "
                                   f"{self.NEED_TAPS} taps...")
                    if len(self.windows) >= self.NEED_TAPS:
                        self.phase = "done"
                        self._compute()
            elif t > self.refract and rms > max(self.noise * 6.0, 0.004):
                self.capture_in = 5      # let the tap ring into the buffer
                self.refract = t + 0.18

            if self.phase == "taps" and t > self.TIMEOUT:
                if len(self.windows) >= 3:
                    self.phase = "done"
                    self._compute()
                else:
                    self.phase = "failed"
                    self.status = ("Calibration failed - I could not hear "
                                   "enough taps. Tap harder or move closer "
                                   "to the mic, then try again.")

    def _compute(self):
        # average magnitude spectrum of the captured taps
        n = min(len(w) for w in self.windows)
        spec = None
        for w in self.windows:
            seg = w[:n] * np.hanning(n)
            mag = np.abs(np.fft.rfft(seg))
            spec = mag if spec is None else spec + mag
        freqs = np.fft.rfftfreq(n, 1.0 / self.sr)
        spec[freqs < 60] = 0.0

        peak = int(np.argmax(spec))
        pm = spec[peak]
        lo_i = peak
        while lo_i > 0 and spec[lo_i] > pm * 0.2:
            lo_i -= 1
        hi_i = peak
        while hi_i < len(spec) - 1 and spec[hi_i] > pm * 0.2:
            hi_i += 1
        lo = max(60.0, freqs[lo_i] * 0.8)
        hi = min(self.sr * 0.45, freqs[hi_i] * 1.25)
        if hi - lo < 200.0:
            c = (hi + lo) / 2.0
            lo, hi = max(60.0, c - 150.0), c + 150.0

        sens = 4.0
        if sps is not None:
            sos = design_band(lo, hi)
            nb = np.concatenate(self.noise_blocks)
            noise_f = max(rms_of(sps.sosfilt(sos, nb)), MIN_NOISE)
            peaks = []
            for w in self.windows:
                fw = sps.sosfilt(sos, w[:n])
                k = self.block
                block_rms = [rms_of(fw[i:i + k])
                             for i in range(0, len(fw) - k + 1, k)]
                if block_rms:
                    peaks.append(max(block_rms))
            if peaks:
                med = float(np.median(peaks))
                thr = noise_f + 0.18 * max(med - noise_f, 0.0)
                sens = float(np.clip(thr / noise_f, 2.5, 15.0))

        sigs = [spectral_signature(w[:n], self.sr) for w in self.windows]
        template = np.mean(sigs, axis=0)
        template -= template.mean()
        template /= max(np.linalg.norm(template), 1e-9)
        pair = [similarity(a, b)
                for i, a in enumerate(sigs) for b in sigs[i + 1:]]
        min_pair = min(pair) if pair else 0.7
        min_match = float(np.clip(min_pair - 0.15, 0.30, 0.90))

        self.result = {"band": [round(float(lo), 1), round(float(hi), 1)],
                       "sensitivity": round(float(sens), 2),
                       "tap_template": [round(float(v), 5)
                                        for v in template],
                       "min_match": round(min_match, 2)}
        self.status = (f"Calibrated: listening in the "
                       f"{lo:.0f}-{hi:.0f} Hz band")


# ----------------------------------------------------------- reject learner

class RejectLearner:
    """
    Learns the fingerprint of a sound that should NOT count as a tap.

    Click the button, then make the offending noise (type on your keyboard,
    click your mouse...) for a few seconds. Every transient heard is
    fingerprinted and averaged into one ignore-template.
    """

    QUIET_SECS = 0.5
    LISTEN_SECS = 6.0

    def __init__(self, sr=SAMPLE_RATE, block=BLOCK):
        self.sr = sr
        self.block = block
        self.phase = "quiet"          # quiet -> listen -> done | failed
        self.status = "One moment..."
        self.quiet = []
        self.ring = deque(maxlen=16)
        self.sigs = []
        self.capture_in = None
        self.refract = 0.0
        self.blocks_seen = 0
        self.noise = None
        self.result = None

    def feed(self, raw):
        self.blocks_seen += 1
        t = self.blocks_seen * self.block / self.sr
        self.ring.append(np.array(raw, dtype=float))
        rms = rms_of(raw)

        if self.phase == "quiet":
            self.quiet.append(rms)
            if t >= self.QUIET_SECS:
                self.noise = max(float(np.median(self.quiet)), MIN_NOISE)
                self.phase = "listen"
                self.status = ("Now make the sound to ignore - keep it "
                               "going for a few seconds...")

        elif self.phase == "listen":
            if self.capture_in is not None:
                self.capture_in -= 1
                if self.capture_in <= 0:
                    self.sigs.append(spectral_signature(
                        np.concatenate(list(self.ring)), self.sr))
                    self.capture_in = None
                    self.status = f"Captured {len(self.sigs)} sounds..."
            elif t > self.refract and rms > max(self.noise * 5.0, 0.004):
                self.capture_in = 5
                self.refract = t + 0.12

            if t >= self.QUIET_SECS + self.LISTEN_SECS:
                if len(self.sigs) >= 3:
                    m = np.mean(self.sigs, axis=0)
                    m -= m.mean()
                    m /= max(np.linalg.norm(m), 1e-9)
                    self.result = [round(float(v), 5) for v in m]
                    self.phase = "done"
                else:
                    self.phase = "failed"
                    self.status = (f"Only heard {len(self.sigs)} sounds - "
                                   "try again, louder or closer to the mic.")


# ------------------------------------------------------------------ actions

def launch_target(target):
    """Open a file/app path, or run a command if the path doesn't exist."""
    if sys.platform.startswith("win") and os.path.exists(target):
        os.startfile(target)  # noqa: S606 - user-chosen local target
    elif os.path.exists(target):
        opener = "open" if sys.platform == "darwin" else "xdg-open"
        subprocess.Popen([opener, target])
    else:
        subprocess.Popen(target, shell=True)


# ----------------------------------------------------------------------- UI

ACCENT = "#22d3ee"
BG_DARK = "#0f172a"
BG_MID = "#1e293b"
FG_DIM = "#94a3b8"
GOOD = "#4ade80"
WARN = "#fbbf24"


class TapLauncherApp:
    def __init__(self, root):
        self.root = root
        self.cfg = load_config()

        self.events = queue.Queue()
        self.detector = TapDetector(self.events.put)
        self.detector.sensitivity = float(self.cfg["sensitivity"])
        self.detector.group_window = float(self.cfg["multi_tap_window"])
        if self.cfg.get("band"):
            self.detector.set_band(tuple(self.cfg["band"]))
        if self.cfg.get("tap_template"):
            self.detector.tap_template = np.asarray(
                self.cfg["tap_template"], dtype=float)
        self.detector.min_match = float(self.cfg.get("min_match") or 0.6)
        self.detector.reject_templates = [
            np.asarray(r, dtype=float)
            for r in (self.cfg.get("reject_templates") or [])]

        self.calibrator = None
        self.learner = None
        self.stream = None
        self.level = 0.0
        self.thr = None
        self.state_txt = "starting"
        self.flash_until = 0.0
        self.big_text = ""
        self.big_until = 0.0

        self._build_ui()
        self._start_stream(self.cfg.get("input_device"))
        self.root.after(30, self._poll)
        self.root.protocol("WM_DELETE_WINDOW", self._on_close)

    # -- UI construction ----------------------------------------------

    def _build_ui(self):
        self.root.title("Tap Launcher")
        self.root.geometry("560x780")
        self.root.minsize(520, 700)
        style = ttk.Style(self.root)
        try:
            style.theme_use("clam")
        except Exception:
            pass

        outer = ttk.Frame(self.root, padding=10)
        outer.pack(fill="both", expand=True)

        # live view
        self.canvas = tk.Canvas(outer, height=170, bg=BG_DARK,
                                highlightthickness=0)
        self.canvas.pack(fill="x")

        self.status_lbl = ttk.Label(outer, text="", foreground="#334155")
        self.status_lbl.pack(fill="x", pady=(4, 8))

        # device picker
        dev = ttk.Frame(outer)
        dev.pack(fill="x", pady=(0, 6))
        ttk.Label(dev, text="Microphone:").pack(side="left")
        self.device_var = tk.StringVar()
        self.device_box = ttk.Combobox(dev, textvariable=self.device_var,
                                       state="readonly", width=42)
        self.device_box.pack(side="left", padx=6, fill="x", expand=True)
        self.device_box.bind("<<ComboboxSelected>>", self._on_device_change)
        self._fill_devices()

        # calibration
        cal = ttk.LabelFrame(outer, text="Calibration", padding=8)
        cal.pack(fill="x", pady=6)
        row1 = ttk.Frame(cal)
        row1.pack(fill="x")
        self.cal_btn = ttk.Button(row1, text="Calibrate for this surface",
                                  command=self._start_calibration)
        self.cal_btn.pack(side="left")
        ttk.Button(row1, text="Reset",
                   command=self._reset_calibration).pack(side="left", padx=6)
        band = self.cfg.get("band")
        band_txt = (f"Band: {band[0]:.0f}-{band[1]:.0f} Hz" if band
                    else "Not calibrated - listening to all frequencies")
        self.band_lbl = ttk.Label(row1, text=band_txt)
        self.band_lbl.pack(side="left", padx=8)

        row2 = ttk.Frame(cal)
        row2.pack(fill="x", pady=(6, 0))
        self.learn_btn = ttk.Button(row2, text="Learn a sound to ignore",
                                    command=self._start_learn)
        self.learn_btn.pack(side="left")
        self.clear_btn = ttk.Button(row2, text="Clear ignored",
                                    command=self._clear_ignored)
        self.clear_btn.pack(side="left", padx=6)
        ttk.Label(row2, foreground="#64748b",
                  text="click, then e.g. type for 6 s"
                  ).pack(side="left", padx=8)
        self._update_clear_btn()

        # tuning sliders
        tune = ttk.LabelFrame(outer, text="Tuning", padding=8)
        tune.pack(fill="x", pady=6)
        ttk.Label(tune, text="Trigger sensitivity"
                  ).grid(row=0, column=0, sticky="w")
        self.sens_var = tk.DoubleVar(value=float(self.cfg["sensitivity"]))
        ttk.Scale(tune, from_=15.0, to=2.0, variable=self.sens_var,
                  command=self._on_sens).grid(row=0, column=1, sticky="ew",
                                              padx=8)
        self.sens_lbl = ttk.Label(tune, width=18)
        self.sens_lbl.grid(row=0, column=2, sticky="e")

        ttk.Label(tune, text="Multi-tap window"
                  ).grid(row=1, column=0, sticky="w")
        self.win_var = tk.DoubleVar(value=float(self.cfg["multi_tap_window"]))
        ttk.Scale(tune, from_=0.25, to=0.70, variable=self.win_var,
                  command=self._on_window).grid(row=1, column=1, sticky="ew",
                                                padx=8)
        self.win_lbl = ttk.Label(tune, width=18)
        self.win_lbl.grid(row=1, column=2, sticky="e")

        ttk.Label(tune, text="Tap match strictness"
                  ).grid(row=2, column=0, sticky="w")
        self.match_var = tk.DoubleVar(
            value=float(self.cfg.get("min_match") or 0.6))
        ttk.Scale(tune, from_=0.20, to=0.95, variable=self.match_var,
                  command=self._on_match).grid(row=2, column=1, sticky="ew",
                                               padx=8)
        self.match_lbl = ttk.Label(tune, width=18)
        self.match_lbl.grid(row=2, column=2, sticky="e")

        tune.columnconfigure(1, weight=1)
        self._on_sens()
        self._on_window()
        self._on_match()

        # actions / app assignment
        acts = ttk.LabelFrame(outer, text="Actions", padding=8)
        acts.pack(fill="x", pady=6)

        self.enable_var = tk.BooleanVar(value=bool(self.cfg["actions_enabled"]))
        ttk.Checkbutton(
            acts, variable=self.enable_var, command=self._on_toggles,
            text="Launch apps on tap patterns (leave off while testing)"
        ).grid(row=0, column=0, columnspan=3, sticky="w", pady=(0, 6))

        self.app_vars = {}
        row = 1
        names = {2: "Double tap", 3: "Triple tap", 4: "Quadruple tap"}
        for n in (2, 3, 4):
            ttk.Label(acts, text=names[n]).grid(row=row, column=0, sticky="w")
            var = tk.StringVar(value=self.cfg["apps"].get(str(n), ""))
            self.app_vars[n] = var
            ttk.Entry(acts, textvariable=var).grid(
                row=row, column=1, sticky="ew", padx=6)
            ttk.Button(acts, text="Browse", width=8,
                       command=lambda n=n: self._browse(n)
                       ).grid(row=row, column=2)
            var.trace_add("write", lambda *_a, n=n: self._on_app_change(n))
            row += 1

        self.drag_var = tk.BooleanVar(value=bool(self.cfg["drag_enabled"]))
        drag_cb = ttk.Checkbutton(
            acts, variable=self.drag_var, command=self._on_toggles,
            text="Drag scrolling - away scrolls down, toward jumps to top "
                 "(experimental)")
        drag_cb.grid(row=row, column=0, columnspan=3, sticky="w", pady=(6, 0))
        if pyautogui is None:
            drag_cb.state(["disabled"])
            ttk.Label(acts, foreground="#b45309",
                      text="pip install pyautogui to enable scrolling"
                      ).grid(row=row + 1, column=0, columnspan=3, sticky="w")
        acts.columnconfigure(1, weight=1)

        # log
        logf = ttk.LabelFrame(outer, text="What I'm hearing", padding=6)
        logf.pack(fill="both", expand=True, pady=6)
        self.log = tk.Text(logf, height=9, state="disabled", bg="#f8fafc",
                           relief="flat", wrap="word")
        self.log.pack(fill="both", expand=True)

    def _fill_devices(self):
        items, self._device_ids = [], []
        if sd is not None:
            try:
                default_in = sd.default.device[0]
                for i, d in enumerate(sd.query_devices()):
                    if d.get("max_input_channels", 0) > 0:
                        tag = " (default)" if i == default_in else ""
                        items.append(f"{i}: {d['name']}{tag}")
                        self._device_ids.append(i)
            except Exception:
                pass
        self.device_box["values"] = items
        want = self.cfg.get("input_device")
        if want in self._device_ids:
            self.device_var.set(items[self._device_ids.index(want)])
        elif items:
            self.device_box.current(0 if sd is None else
                                    self._nearest_default(items))

    def _nearest_default(self, items):
        for k, s in enumerate(items):
            if "(default)" in s:
                return k
        return 0

    # -- audio stream --------------------------------------------------

    def _start_stream(self, device=None):
        if sd is None:
            self._log("sounddevice is not installed - run: "
                      "pip install sounddevice")
            self.state_txt = "no audio"
            return
        if sps is None:
            self._log("scipy is not installed (calibration filtering needs "
                      "it) - run: pip install scipy")
        try:
            if self.stream is not None:
                self.stream.stop()
                self.stream.close()
            self.stream = sd.InputStream(
                samplerate=SAMPLE_RATE, blocksize=BLOCK, channels=1,
                dtype="float32", device=device, callback=self._audio_cb)
            self.stream.start()
            self._log("Listening. Tap near the mic to see it register.")
        except Exception as e:
            self._log(f"Could not open the microphone: {e}")
            self.state_txt = "mic error"

    def _audio_cb(self, indata, frames, time_info, status):
        mono = np.asarray(indata[:, 0], dtype=np.float32)
        cal = self.calibrator
        if cal is not None and cal.phase in ("quiet", "taps"):
            cal.feed(mono)
            self.events.put({"type": "level", "rms": rms_of(mono),
                             "thr": None, "state": "calibrating"})
            return
        lrn = self.learner
        if lrn is not None and lrn.phase in ("quiet", "listen"):
            lrn.feed(mono)
            self.events.put({"type": "level", "rms": rms_of(mono),
                             "thr": None, "state": "learning ignore"})
            return
        self.detector.process(mono)

    # -- UI callbacks --------------------------------------------------

    def _on_device_change(self, _evt=None):
        sel = self.device_box.current()
        if 0 <= sel < len(self._device_ids):
            dev = self._device_ids[sel]
            self.cfg["input_device"] = dev
            save_config(self.cfg)
            self._start_stream(dev)

    def _on_sens(self, _v=None):
        v = float(self.sens_var.get())
        self.detector.sensitivity = v
        self.sens_lbl.config(text=f"{v:.1f}x noise floor")
        self.cfg["sensitivity"] = round(v, 2)
        save_config(self.cfg)

    def _on_window(self, _v=None):
        v = float(self.win_var.get())
        self.detector.group_window = v
        self.win_lbl.config(text=f"{v * 1000:.0f} ms between taps")
        self.cfg["multi_tap_window"] = round(v, 3)
        save_config(self.cfg)

    def _on_toggles(self):
        self.cfg["actions_enabled"] = bool(self.enable_var.get())
        self.cfg["drag_enabled"] = bool(self.drag_var.get())
        save_config(self.cfg)

    def _on_app_change(self, n):
        self.cfg["apps"][str(n)] = self.app_vars[n].get()
        save_config(self.cfg)

    def _browse(self, n):
        ft = [("Programs", "*.exe"), ("All files", "*.*")] \
            if sys.platform.startswith("win") else [("All files", "*.*")]
        path = filedialog.askopenfilename(title=f"App for {n} taps",
                                          filetypes=ft)
        if path:
            self.app_vars[n].set(path)

    def _on_match(self, _v=None):
        v = float(self.match_var.get())
        self.detector.min_match = v
        self.match_lbl.config(text=f"{v * 100:.0f}% similarity")
        self.cfg["min_match"] = round(v, 2)
        save_config(self.cfg)

    def _set_busy(self, busy):
        state = ["disabled"] if busy else ["!disabled"]
        self.cal_btn.state(state)
        self.learn_btn.state(state)
        if not busy:
            self._update_clear_btn()

    def _update_clear_btn(self):
        n = len(self.cfg.get("reject_templates") or [])
        self.clear_btn.config(text=f"Clear ignored ({n})")
        self.clear_btn.state(["!disabled"] if n else ["disabled"])

    def _start_learn(self):
        self.learner = RejectLearner()
        self._set_busy(True)
        self._log("Learning a sound to ignore - make the noise "
                  "(e.g. type normally) for about 6 seconds...")

    def _clear_ignored(self):
        self.cfg["reject_templates"] = []
        save_config(self.cfg)
        self.detector.reject_templates = []
        self._update_clear_btn()
        self._log("Cleared all learned ignore-sounds.")

    def _start_calibration(self):
        self.calibrator = Calibrator()
        self._set_busy(True)
        self._log("Calibration started - stay quiet for a second...")

    def _reset_calibration(self):
        self.cfg["band"] = None
        self.cfg["tap_template"] = None
        save_config(self.cfg)
        self.detector.set_band(None)
        self.detector.tap_template = None
        self.band_lbl.config(text="Not calibrated - "
                                  "listening to all frequencies")
        self._log("Calibration cleared.")

    # -- event loop ----------------------------------------------------

    def _poll(self):
        now = time.monotonic()
        try:
            while True:
                e = self.events.get_nowait()
                self._handle_event(e, now)
        except queue.Empty:
            pass
        self._tick_calibration()
        self._tick_learner()
        self._redraw(now)
        self.root.after(30, self._poll)

    def _handle_event(self, e, now):
        kind = e["type"]
        if kind == "level":
            self.level = e["rms"]
            self.thr = e["thr"]
            self.state_txt = e["state"]
        elif kind == "tap":
            self.flash_until = now + 0.15
            sim = e.get("sim")
            extra = f", {sim * 100:.0f}% match" if sim is not None else ""
            self._log(f"Tap{extra}  ({e['count_so_far']} so far...)")
        elif kind == "rejected":
            if e.get("reason") == "ignored":
                self._log(f"Ignored - {e['rej_sim'] * 100:.0f}% match to a "
                          f"learned ignore-sound (only "
                          f"{e['sim'] * 100:.0f}% like your taps).")
            else:
                self._log(f"Rejected - only {e['sim'] * 100:.0f}% similar "
                          f"to your calibrated taps (strictness "
                          f"{self.detector.min_match * 100:.0f}%).")
        elif kind == "group":
            n = e["n"]
            label = {1: "SINGLE TAP", 2: "DOUBLE TAP",
                     3: "TRIPLE TAP", 4: "QUADRUPLE TAP"}.get(n, f"{n} TAPS")
            self.big_text, self.big_until = label, now + 1.3
            self._run_group(n, label)
        elif kind == "drag":
            d = e["direction"]
            arrow = {"toward": "UP - toward mic", "away": "DOWN - away",
                     "flat": "flat"}[d]
            self.big_text, self.big_until = f"DRAG {arrow}", now + 1.3
            self._run_drag(d, e["duration"])
        elif kind == "ignored":
            self._log(f"Heard a sound ({e['duration'] * 1000:.0f} ms) - "
                      f"too long for a tap, too short for a drag. Ignored.")
        elif kind == "noise":
            self._log("Sustained noise - ignoring it and resetting.")

    def _run_group(self, n, label):
        target = (self.cfg["apps"].get(str(n)) or "").strip()
        if n == 1:
            self._log("Single tap (no action is bound to one tap).")
            return
        if not target:
            self._log(f"{label} - nothing assigned yet.")
            return
        if not self.cfg["actions_enabled"]:
            self._log(f"{label} - would launch: {target} "
                      f"(launching is turned off)")
            return
        try:
            launch_target(target)
            self._log(f"{label} - launching {os.path.basename(target)}")
        except Exception as ex:
            self._log(f"{label} - failed to launch: {ex}")

    def _run_drag(self, direction, dur):
        desc = {"toward": "louder toward the end",
                "away": "quieter toward the end",
                "flat": "steady volume"}[direction]
        if direction == "flat":
            self._log(f"Drag ({dur:.2f} s, {desc}) - direction unclear.")
            return
        if not self.cfg["drag_enabled"]:
            self._log(f"Drag {direction} ({dur:.2f} s, {desc}) - "
                      f"scrolling is turned off.")
            return
        if pyautogui is None:
            self._log("Drag detected but pyautogui is not installed.")
            return
        try:
            if direction == "toward":
                pyautogui.press("home")          # jump to top of the page
                self._log(f"Drag toward mic ({dur:.2f} s) - jumped to top.")
            else:
                pyautogui.scroll(-600)           # scroll down
                self._log(f"Drag away from mic ({dur:.2f} s) - scrolled "
                          f"down.")
        except Exception as ex:
            self._log(f"Drag action failed: {ex}")

    def _tick_calibration(self):
        cal = self.calibrator
        if cal is None:
            return
        self.status_lbl.config(text=cal.status)
        if cal.phase == "done" and cal.result:
            band = cal.result["band"]
            sens = cal.result["sensitivity"]
            self.cfg["band"] = band
            self.cfg["sensitivity"] = sens
            self.cfg["tap_template"] = cal.result["tap_template"]
            self.cfg["min_match"] = cal.result["min_match"]
            save_config(self.cfg)
            self.detector.set_band(tuple(band))
            self.detector.tap_template = np.asarray(
                cal.result["tap_template"], dtype=float)
            self.sens_var.set(sens)
            self._on_sens()
            self.match_var.set(cal.result["min_match"])
            self._on_match()
            self.band_lbl.config(
                text=f"Band: {band[0]:.0f}-{band[1]:.0f} Hz")
            self._log(f"Calibrated. Watching {band[0]:.0f}-{band[1]:.0f} Hz "
                      f"and fingerprinting your taps (strictness "
                      f"{cal.result['min_match'] * 100:.0f}%) - try a "
                      f"double tap.")
            self.calibrator = None
            self._set_busy(False)
            self.status_lbl.config(text="")
        elif cal.phase == "failed":
            self._log(cal.status)
            self.calibrator = None
            self._set_busy(False)
            self.status_lbl.config(text="")

    def _tick_learner(self):
        lrn = self.learner
        if lrn is None:
            return
        self.status_lbl.config(text=lrn.status)
        if lrn.phase == "done" and lrn.result:
            self.cfg.setdefault("reject_templates", [])
            self.cfg["reject_templates"].append(lrn.result)
            save_config(self.cfg)
            self.detector.reject_templates = [
                np.asarray(r, dtype=float)
                for r in self.cfg["reject_templates"]]
            self._log(f"Learned an ignore-sound from {len(lrn.sigs)} "
                      f"samples ({len(self.cfg['reject_templates'])} "
                      f"total). Make that noise again - the log should "
                      f"now say Ignored.")
            self.learner = None
            self._set_busy(False)
            self.status_lbl.config(text="")
        elif lrn.phase == "failed":
            self._log(lrn.status)
            self.learner = None
            self._set_busy(False)
            self.status_lbl.config(text="")

    # -- drawing -------------------------------------------------------

    def _redraw(self, now):
        c = self.canvas
        c.delete("all")
        w = max(c.winfo_width(), 2)
        h = 170

        bg = BG_MID if now < self.flash_until else BG_DARK
        c.create_rectangle(0, 0, w, h, fill=bg, width=0)

        # level meter (log scale) with threshold marker
        pad, bar_y, bar_h = 16, h - 34, 16
        c.create_rectangle(pad, bar_y, w - pad, bar_y + bar_h,
                           fill="#020617", width=0)

        def frac(v):
            return min(max((to_db(v) + 70.0) / 60.0, 0.0), 1.0)

        lv = frac(self.level)
        color = ACCENT if (self.thr is None or self.level < self.thr) \
            else GOOD
        c.create_rectangle(pad, bar_y, pad + (w - 2 * pad) * lv,
                           bar_y + bar_h, fill=color, width=0)
        if self.thr is not None:
            tx = pad + (w - 2 * pad) * frac(self.thr)
            c.create_line(tx, bar_y - 4, tx, bar_y + bar_h + 4,
                          fill=WARN, width=2)
            c.create_text(tx, bar_y + bar_h + 12, text="threshold",
                          fill=WARN, font=("Segoe UI", 8))

        # headline text
        if now < self.big_until and self.big_text:
            c.create_text(w / 2, 62, text=self.big_text, fill=ACCENT,
                          font=("Segoe UI", 26, "bold"))
        else:
            hint = {"calibrating": "Calibrating...",
                    "learning": "Learning the room's noise floor...",
                    "event": "Hearing something...",
                    "no audio": "Audio unavailable",
                    "mic error": "Microphone error - pick another device",
                    }.get(self.state_txt, "Listening for taps")
            c.create_text(w / 2, 62, text=hint, fill=FG_DIM,
                          font=("Segoe UI", 15))

        # status line
        noise = self.detector.noise
        parts = [f"state: {self.state_txt}"]
        if noise is not None:
            parts.append(f"noise: {to_db(noise):.0f} dB")
        if self.thr is not None:
            parts.append(f"threshold: {to_db(self.thr):.0f} dB")
        c.create_text(pad, 16, anchor="w", text="   |   ".join(parts),
                      fill="#475569", font=("Segoe UI", 9))

    def _log(self, msg):
        stamp = time.strftime("%H:%M:%S")
        self.log.configure(state="normal")
        self.log.insert("end", f"[{stamp}] {msg}\n")
        if int(self.log.index("end-1c").split(".")[0]) > 250:
            self.log.delete("1.0", "2.0")
        self.log.see("end")
        self.log.configure(state="disabled")

    def _on_close(self):
        try:
            if self.stream is not None:
                self.stream.stop()
                self.stream.close()
        except Exception:
            pass
        save_config(self.cfg)
        self.root.destroy()


# --------------------------------------------------------------------- main

def main():
    if tk is None:
        print("tkinter is not available in this Python install.")
        sys.exit(1)
    missing = [name for name, mod in
               [("numpy", np), ("sounddevice", sd), ("scipy", sps)]
               if mod is None]
    if "numpy" in missing:
        print("numpy is required: pip install numpy")
        sys.exit(1)
    root = tk.Tk()
    TapLauncherApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()
