#!/usr/bin/env python3
"""
Tap Launcher — listens to your microphone for finger taps and runs actions.

Double-tap or triple-tap near the mic to launch apps or fire quick actions
(mute the microphone, minimize all windows, media keys, lock the screen...).
Distance moves add two more gestures: Close·Far (tap by the mic, then tap
farther away) and Far·Close, told apart by the loudness gap between taps.

All audio is processed live, in memory. Nothing is recorded or saved.

Run:  python tap_launcher.py
Deps: pip install numpy scipy sounddevice pyautogui
"""

import base64
import ctypes
import json
import os
import queue
import struct
import subprocess
import sys
import threading
import time
import zlib
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
    import pystray
    from PIL import Image as PILImage, ImageDraw as PILImageDraw
except Exception:
    pystray = None

try:
    import soundcard as _soundcard
except Exception:
    _soundcard = None

try:
    import tkinter as tk
    from tkinter import filedialog, ttk
except Exception:
    tk = None


# ---------------------------------------------------------------- constants

SAMPLE_RATE = 44100
BLOCK = 512                  # ~11.6 ms per block

TAP_MAX_DUR = 0.12           # a burst shorter than this is a tap
EVENT_ABORT_DUR = 3.0        # anything longer is background noise
REFRACTORY = 0.06            # dead time after an event ends
RELEASE_RATIO = 0.5          # event ends when level falls below thr * this
NOISE_ALPHA = 0.02           # how fast the noise floor adapts
MIN_NOISE = 1e-5
SPEC_N = 4096                # samples analysed around a transient's peak
REJECT_MARGIN = 0.05         # bias toward rejecting near-ties with ignores
REJECT_FLOOR = 0.45          # an ignore-sound must at least somewhat match

APP_VERSION = "1.0"
APP_DIR = os.path.dirname(os.path.abspath(__file__))
CONFIG_PATH = os.path.join(APP_DIR, "tap_launcher_config.json")

DEFAULT_CONFIG = {
    "sensitivity": 4.0,       # threshold = noise floor x sensitivity
    "sensitivity_base": 4.0,  # baseline set by calibration
    "sensitivity_level": "normal",  # user preset, see SENS_LEVELS
    "multi_tap_window": 0.45, # max seconds between taps in one pattern
    "multi_tap_level": "medium",    # preset, see MULTITAP_LEVELS
    "near_far_db": 8.0,       # loudness gap that makes a distance move
    "gap_level": "medium",          # preset, see GAP_LEVELS
    "tap_max_dur": TAP_MAX_DUR,     # longest burst still counted as a tap
    "tap_len_level": "short",       # preset, see TAPLEN_LEVELS
    "shape_gate": True,       # reject sounds without a sharp, tap-like onset
    "speaker_gate": False,    # pause detection while the PC speakers play
    "echo_reject": False,     # ignore only taps that match speaker output
    "band": None,             # [lo_hz, hi_hz] set by calibration
    "tap_templates": [],      # spectral signatures, one per learned tap type
    "min_match": 0.6,         # similarity a sound needs to count as a tap
    "reject_templates": [],   # signatures of sounds to ignore (keyboard...)
    "patterns": {},           # per-pattern action, see _default_patterns()
    "profiles": {},           # named calibrations, see PROFILE_KEYS
    "active_profile": None,
    "actions_enabled": False,
    "input_device": None,
    "theme": "dark",
}

# sensitivity presets: the calibrated baseline is scaled by these factors
# (a lower threshold multiplier = picks up softer taps)
SENS_LEVELS = {
    "more": ("More sensitive", 0.65),
    "normal": ("Normal", 1.0),
    "less": ("Less sensitive", 1.6),
}


def effective_sensitivity(cfg):
    base = float(cfg.get("sensitivity_base") or 4.0)
    factor = SENS_LEVELS.get(cfg.get("sensitivity_level"),
                             SENS_LEVELS["normal"])[1]
    return float(np.clip(base * factor, 2.0, 15.0))


# preset button groups for the timing / distance / tap-length knobs.
# each maps a level key -> (label, numeric value the detector uses)
MULTITAP_LEVELS = {   # how close together taps must land to count as one group
    "tight": ("Tight", 0.30),
    "medium": ("Medium", 0.45),
    "relaxed": ("Relaxed", 0.65),
}
GAP_LEVELS = {        # dB louder/softer the 2nd tap needs for a distance move
    "small": ("Small", 5.0),
    "medium": ("Medium", 8.0),
    "large": ("Large", 12.0),
}
TAPLEN_LEVELS = {     # longest burst still treated as a tap (vs "too long")
    "short": ("Short", 0.12),
    "medium": ("Medium", 0.22),
    "long": ("Long", 0.35),
}

# ties each preset group to the numeric config key the detector reads
PRESET_GROUPS = {
    "multi_tap_level": (MULTITAP_LEVELS, "multi_tap_window", "medium"),
    "gap_level": (GAP_LEVELS, "near_far_db", "medium"),
    "tap_len_level": (TAPLEN_LEVELS, "tap_max_dur", "short"),
}


def _level_value(levels, key, default_key):
    return levels.get(key, levels[default_key])[1]


def _closest_level(levels, value, default_key):
    if value is None:
        return default_key
    return min(levels, key=lambda k: abs(levels[k][1] - float(value)))


# what a tap pattern can do: key -> label shown in the dropdown
ACTIONS = {
    "none": "Do nothing",
    "app": "Launch an app or file",
    "app_dictate": "Launch app + voice typing (Win+H)",
    "voice_typing": "Start voice typing (Win+H)",
    "minimize": "Minimize / maximize all windows",
    "mute_mic": "Mute / unmute the microphone",
    "mute_speakers": "Mute / unmute the speakers",
    "play_pause": "Play / pause media",
    "next_track": "Next track",
    "prev_track": "Previous track",
    "volume_up": "Volume up",
    "volume_down": "Volume down",
    "lock": "Lock the screen",
    "screenshot": "Take a screenshot",
}
LABEL_TO_ACTION = {v: k for k, v in ACTIONS.items()}

# actions that launch the target chosen in the app row
APP_ACTIONS = ("app", "app_dictate")

# bindable tap patterns: plain counts, then distance moves — two taps
# where the second is much closer to / farther from the mic than the first
PATTERN_KEYS = ("2", "3", "4", "close-far", "far-close")
PATTERN_NAMES = {
    "2": "Double tap", "3": "Triple tap", "4": "Quadruple tap",
    "close-far": "Close·Far / Hard·Soft", "far-close": "Far·Close / Soft·Hard",
}

NEAR_FAR_DB = 8.0    # default loudness gap to count as a distance move


def tap_trend(peaks, min_db=NEAR_FAR_DB):
    """For a two-tap group, classify the loudness trend: first tap much
    louder = moved away ('close-far'), much quieter = moved in
    ('far-close'), similar = None (a plain double tap)."""
    if not peaks or len(peaks) != 2 or min(peaks) <= 0:
        return None
    db = 20.0 * float(np.log10(peaks[0] / peaks[1]))
    if db >= min_db:
        return "close-far"
    if db <= -min_db:
        return "far-close"
    return None

# what a surface profile stores (calibration only; actions stay global)
PROFILE_KEYS = ("band", "tap_templates", "min_match", "sensitivity_base",
                "reject_templates")


def _default_patterns():
    return {k: {"action": "none", "target": "", "arg": ""}
            for k in PATTERN_KEYS}


def load_config():
    cfg = dict(DEFAULT_CONFIG)
    cfg["patterns"] = _default_patterns()
    try:
        with open(CONFIG_PATH, "r", encoding="utf-8") as f:
            saved = json.load(f)
    except Exception:
        return cfg
    for k in cfg:
        if k in saved and k != "patterns":
            cfg[k] = saved[k]
    pats = _default_patterns()
    saved_pats = saved.get("patterns")
    if isinstance(saved_pats, dict):
        for n, p in saved_pats.items():
            if n in pats and isinstance(p, dict):
                action = p.get("action")
                pats[n] = {
                    "action": action if action in ACTIONS else "none",
                    "target": str(p.get("target") or ""),
                    "arg": str(p.get("arg") or ""),
                }
    else:
        # migrate the old {"apps": {"2": "path"}} format
        for n, path in (saved.get("apps") or {}).items():
            if n in pats and (path or "").strip():
                pats[n] = {"action": "app", "target": path.strip(),
                           "arg": ""}
    cfg["patterns"] = pats
    cfg["profiles"] = {str(k): v
                       for k, v in (cfg.get("profiles") or {}).items()
                       if isinstance(v, dict)}
    if "sensitivity_base" not in saved:
        # older configs stored only the raw multiplier; use it as baseline
        try:
            cfg["sensitivity_base"] = float(saved.get("sensitivity", 4.0))
        except (TypeError, ValueError):
            cfg["sensitivity_base"] = 4.0
    if cfg.get("sensitivity_level") not in SENS_LEVELS:
        cfg["sensitivity_level"] = "normal"
    cfg["sensitivity"] = effective_sensitivity(cfg)
    if cfg.get("theme") not in ("dark", "light"):
        cfg["theme"] = "dark"
    tpls = cfg.get("tap_templates")
    if not (isinstance(tpls, list)
            and all(isinstance(t, list) for t in tpls)):
        tpls = []
    if not tpls and isinstance(saved.get("tap_template"), list):
        tpls = [saved["tap_template"]]      # migrate the single-template era
    cfg["tap_templates"] = tpls
    # timing / gap / tap-length presets: the level key is the source of
    # truth. An old config without one maps its former slider value to the
    # nearest preset; either way the numeric the detector reads is derived.
    for lvlkey, (levels, numkey, dfl) in PRESET_GROUPS.items():
        if saved.get(lvlkey) in levels:
            cfg[lvlkey] = saved[lvlkey]
        else:
            cfg[lvlkey] = _closest_level(levels, saved.get(numkey), dfl)
        cfg[numkey] = _level_value(levels, cfg[lvlkey], dfl)
    cfg["speaker_gate"] = bool(cfg.get("speaker_gate"))
    cfg["echo_reject"] = bool(cfg.get("echo_reject"))
    cfg["shape_gate"] = bool(cfg.get("shape_gate"))
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


SHAPE_MAX_ATTACK = 0.008    # a real tap's onset edge rises within ~8 ms


def onset_attack_time(win, sr=SAMPLE_RATE):
    """
    Seconds the waveform takes to rise into its peak - how sharp the onset
    is. A finger tap snaps up in a couple of ms; speech, music and other
    sustained sounds swell in over tens of ms. Surface- and mic-agnostic,
    so it complements the spectral fingerprint. Returns 0 for anything too
    quiet or short to judge (so it always passes).
    """
    x = np.abs(np.asarray(win, dtype=float))
    if x.size < 64 or x.max() <= 0:
        return 0.0
    k = max(1, int(sr * 0.008))                 # ~8 ms envelope smoothing
    env = np.convolve(x, np.ones(k) / k, mode="same")
    peak = float(env.max())
    if peak <= 0:
        return 0.0
    # measure the RISING EDGE: how long from crossing 10% of peak to
    # crossing 50%. A step (tap) crosses both almost at once; a swell
    # (speech/music) takes tens of ms. Robust to where the single loudest
    # sample happens to land in a flat-topped sound.
    lo = np.flatnonzero(env >= peak * 0.1)
    if lo.size == 0:
        return 0.0
    i_lo = int(lo[0])
    hi = np.flatnonzero(env[i_lo:] >= peak * 0.5)
    if hi.size == 0:
        return 0.0
    return int(hi[0]) / float(sr)


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
    - A burst that rises above threshold and falls back quickly is a *tap*;
      its peak level is kept so two-tap groups can be read as distance
      moves (loud->soft = moved away from the mic, soft->loud = moved in).
    - Taps close together in time are grouped into one pattern
      (2 taps = double, 3 = triple, ...).

    Emits event dicts through the `emit` callback:
      {'type': 'level', 'rms', 'thr', 'state'}         every block
      {'type': 'tap', 't', 'count_so_far', 'peak', 'thr'}
      {'type': 'group', 'n', 'peaks', 't'}
      {'type': 'ignored', 'duration'}                  too long for a tap
      {'type': 'noise'}                                over-long burst
    """

    def __init__(self, emit, sr=SAMPLE_RATE):
        self.emit = emit
        self.sr = sr
        self.sensitivity = DEFAULT_CONFIG["sensitivity"]
        self.group_window = DEFAULT_CONFIG["multi_tap_window"]
        self.tap_max_dur = DEFAULT_CONFIG["tap_max_dur"]
        self.suppress_fn = None       # optional () -> True to drop a tap
        self.shape_gate = DEFAULT_CONFIG["shape_gate"]
        self.shape_max_attack = SHAPE_MAX_ATTACK

        self._lock = threading.Lock()
        self.filter = None
        self.raw_ring = deque(maxlen=16)      # ~190 ms of raw audio
        self.tap_templates = []               # one signature per tap type
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
        self.tap_peaks = []           # peak level of each tap in the group
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
            self.tap_peaks = []

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
                self.tap_peaks = []
                self.emit({"type": "noise"})

        self.emit({"type": "level", "rms": rms, "thr": thr,
                   "state": self.state})

    def _end_event(self, t, dur):
        self.state = "idle"
        self.refract_until = t + REFRACTORY
        peak = max(self.evt_rms) if self.evt_rms else 0.0
        thr = self.threshold
        if dur <= self.tap_max_dur:
            if self.suppress_fn is not None and self.suppress_fn():
                self.emit({"type": "rejected", "t": t, "peak": peak,
                           "thr": thr, "reason": "speaker"})
                self.evt_rms = []
                return
            if self.shape_gate:
                attack = onset_attack_time(
                    np.concatenate(list(self.raw_ring)), self.sr)
                if attack > self.shape_max_attack:
                    self.emit({"type": "rejected", "t": t, "peak": peak,
                               "thr": thr, "reason": "shape",
                               "attack": attack})
                    self.evt_rms = []
                    return
            ok, info = self._classify_tap()
            if ok:
                self.taps.append(t)
                self.tap_peaks.append(peak)
                self.emit({"type": "tap", "t": t,
                           "count_so_far": len(self.taps),
                           "peak": peak, "thr": thr, **info})
            else:
                self.emit({"type": "rejected", "t": t,
                           "peak": peak, "thr": thr, **info})
        else:
            self.emit({"type": "ignored", "duration": dur})
        self.evt_rms = []

    def _finalize_group(self, t):
        n = len(self.taps)
        peaks = list(self.tap_peaks)
        self.taps = []
        self.tap_peaks = []
        self.emit({"type": "group", "n": n, "peaks": peaks, "t": t})

    def _classify_tap(self):
        """
        Compare the just-heard transient's fingerprint against every
        learned tap type (best match wins) and any learned ignore-sounds.

        Returns (accepted, info). Without templates (not calibrated yet)
        every candidate is accepted, as before.
        """
        if not self.tap_templates:
            return True, {}
        win = np.concatenate(list(self.raw_ring))
        sig = spectral_signature(win, self.sr)
        tap_sim = max(similarity(sig, t) for t in self.tap_templates)
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
        self.rejected = 0
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
                    win = np.concatenate(list(self.ring))
                    self.capture_in = None
                    if float(np.abs(win).max()) >= 0.98:
                        self.rejected += 1        # clipped - too loud
                        self.status = ("That tap was too loud (clipped) - "
                                       "tap a little softer.")
                    elif rms_of(win) < max(self.noise * 3.0, 0.003):
                        self.rejected += 1        # too weak to fingerprint
                        self.status = ("That tap was very faint - tap a "
                                       "bit firmer or closer.")
                    else:
                        self.windows.append(win)
                        self.status = (f"Heard {len(self.windows)} of "
                                       f"{self.NEED_TAPS} good taps...")
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
        self._drop_outliers()
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
        lo = max(50.0, freqs[lo_i] * 0.7)
        hi = min(self.sr * 0.45, freqs[hi_i] * 1.6)
        if hi - lo < 400.0:
            c = (hi + lo) / 2.0
            lo, hi = max(50.0, c - 200.0), c + 200.0

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

    def _drop_outliers(self):
        """Discard a captured tap whose fingerprint disagrees with the
        others, so one stray sound can't poison the template. Keeps at
        least three samples."""
        if len(self.windows) < 4:
            return
        m = min(len(w) for w in self.windows)
        sigs = [spectral_signature(w[:m], self.sr) for w in self.windows]
        agree = [float(np.mean([similarity(s, o)
                                for j, o in enumerate(sigs) if j != i]))
                 for i, s in enumerate(sigs)]
        med = float(np.median(agree))
        keep = [w for w, a in zip(self.windows, agree)
                if a >= med - 0.30]
        if 3 <= len(keep) < len(self.windows):
            self.rejected += len(self.windows) - len(keep)
            self.windows = keep


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

    def __init__(self, sr=SAMPLE_RATE, block=BLOCK, prompt="the sound to "
                 "ignore"):
        self.sr = sr
        self.block = block
        self.prompt = prompt
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
                self.status = (f"Now {self.prompt} - keep it going for a "
                               f"few seconds...")

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

def normalize_open_arg(arg):
    """'netflix.com' -> 'https://netflix.com'; leaves real URLs, paths
    and anything ambiguous untouched."""
    arg = (arg or "").strip()
    if (arg and "://" not in arg and "." in arg and " " not in arg
            and not os.path.exists(arg)):
        arg = "https://" + arg
    return arg


def launch_target(target, arg=None):
    """Open a file/app path — optionally handing it a URL or file to open
    (e.g. a browser + a website) — or run a command if the path doesn't
    exist."""
    arg = normalize_open_arg(arg)
    if os.path.exists(target) and arg:
        exe = target
        if target.lower().endswith(".lnk"):
            exe = _resolve_lnk(target) or target
        subprocess.Popen([exe, arg])
    elif sys.platform.startswith("win") and os.path.exists(target):
        os.startfile(target)  # noqa: S606 - user-chosen local target
    elif os.path.exists(target):
        opener = "open" if sys.platform == "darwin" else "xdg-open"
        subprocess.Popen([opener, target])
    else:
        subprocess.Popen(target, shell=True)


def list_installed_apps():
    """
    Installed apps as (name, shortcut_path) from the Start Menu.
    Each shortcut is resolved and kept only if it points at a real .exe,
    which drops folders, web links, help files and other non-apps.
    """
    if not sys.platform.startswith("win"):
        return []
    roots = [os.path.join(os.environ.get("APPDATA", ""),
                          "Microsoft", "Windows", "Start Menu", "Programs"),
             os.path.join(os.environ.get("PROGRAMDATA", ""),
                          "Microsoft", "Windows", "Start Menu", "Programs")]
    found = {}
    for root in roots:
        if not os.path.isdir(root):
            continue
        for dirpath, _dirs, files in os.walk(root):
            for f in files:
                if not f.lower().endswith(".lnk"):
                    continue
                name = f[:-4]
                if "uninstall" in name.lower() or name in found:
                    continue
                full = os.path.join(dirpath, f)
                try:
                    target = _resolve_lnk(full)
                except OSError:
                    target = ""
                if (target.lower().endswith(".exe")
                        and os.path.exists(target)):
                    found[name] = full
    return sorted(found.items(), key=lambda kv: kv[0].lower())


def _resolve_lnk(path):
    """Return the target path of a .lnk shortcut ('' if it has none)."""
    ole32 = ctypes.oledll.ole32
    try:
        ole32.CoInitialize(None)
    except OSError:
        pass
    HRESULT, VOIDP = ctypes.HRESULT, ctypes.c_void_p
    link_p, pf_p = VOIDP(), VOIDP()
    try:
        ole32.CoCreateInstance(
            ctypes.byref(_GUID("{00021401-0000-0000-C000-000000000046}")),
            None, 1,                          # CLSCTX_INPROC_SERVER
            ctypes.byref(_GUID("{000214F9-0000-0000-C000-000000000046}")),
            ctypes.byref(link_p))             # IShellLinkW
        _com_method(link_p, 0, HRESULT, ctypes.POINTER(_GUID),
                    ctypes.POINTER(VOIDP))(
            link_p,
            ctypes.byref(_GUID("{0000010B-0000-0000-C000-000000000046}")),
            ctypes.byref(pf_p))               # QI -> IPersistFile
        _com_method(pf_p, 5, HRESULT, ctypes.c_wchar_p, ctypes.c_uint)(
            pf_p, path, 0)                    # IPersistFile::Load(STGM_READ)
        buf = ctypes.create_unicode_buffer(1024)
        _com_method(link_p, 3, HRESULT, ctypes.c_wchar_p, ctypes.c_int,
                    VOIDP, ctypes.c_uint)(
            link_p, buf, 1024, None, 0)       # IShellLinkW::GetPath
        return buf.value
    finally:
        for p in (pf_p, link_p):
            if p:
                _com_method(p, 2, ctypes.c_ulong)(p)


# -- shell icons for the app picker (ctypes GDI + a tiny stdlib PNG writer) --

class _SHFILEINFOW(ctypes.Structure):
    _fields_ = [("hIcon", ctypes.c_void_p), ("iIcon", ctypes.c_int),
                ("dwAttributes", ctypes.c_uint),
                ("szDisplayName", ctypes.c_wchar * 260),
                ("szTypeName", ctypes.c_wchar * 80)]


class _BITMAPINFOHEADER(ctypes.Structure):
    _fields_ = [("biSize", ctypes.c_uint32), ("biWidth", ctypes.c_int32),
                ("biHeight", ctypes.c_int32), ("biPlanes", ctypes.c_uint16),
                ("biBitCount", ctypes.c_uint16),
                ("biCompression", ctypes.c_uint32),
                ("biSizeImage", ctypes.c_uint32),
                ("biXPelsPerMeter", ctypes.c_int32),
                ("biYPelsPerMeter", ctypes.c_int32),
                ("biClrUsed", ctypes.c_uint32),
                ("biClrImportant", ctypes.c_uint32)]


class _BITMAPINFO(ctypes.Structure):
    _fields_ = [("bmiHeader", _BITMAPINFOHEADER),
                ("bmiColors", ctypes.c_uint32 * 3)]


class _RECT(ctypes.Structure):
    _fields_ = [("left", ctypes.c_long), ("top", ctypes.c_long),
                ("right", ctypes.c_long), ("bottom", ctypes.c_long)]


def _png_from_bgra(raw, w, h):
    """Encode top-down BGRA pixel bytes as an opaque RGB PNG."""
    rows = bytearray()
    for y in range(h):
        rows.append(0)                        # PNG row filter: none
        base = y * w * 4
        for x in range(w):
            i = base + x * 4
            rows += bytes((raw[i + 2], raw[i + 1], raw[i]))

    def chunk(tag, data):
        return (struct.pack(">I", len(data)) + tag + data
                + struct.pack(">I", zlib.crc32(tag + data) & 0xFFFFFFFF))

    ihdr = struct.pack(">IIBBBBB", w, h, 8, 2, 0, 0, 0)
    return (b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", ihdr)
            + chunk(b"IDAT", zlib.compress(bytes(rows)))
            + chunk(b"IEND", b""))


def shell_icon_png(path, size=32, bg=None):
    """PNG bytes of a file's shell icon on the given RGB background
    (default: the current theme's field colour), or None."""
    if not sys.platform.startswith("win"):
        return None
    if bg is None:
        bg = tuple(int(BG_MID[i:i + 2], 16) for i in (1, 3, 5))
    shell32 = ctypes.windll.shell32
    user32 = ctypes.windll.user32
    gdi32 = ctypes.windll.gdi32
    shell32.SHGetFileInfoW.argtypes = [
        ctypes.c_wchar_p, ctypes.c_uint32, ctypes.POINTER(_SHFILEINFOW),
        ctypes.c_uint, ctypes.c_uint]
    shell32.SHGetFileInfoW.restype = ctypes.c_void_p
    gdi32.CreateDIBSection.restype = ctypes.c_void_p
    gdi32.CreateDIBSection.argtypes = [
        ctypes.c_void_p, ctypes.POINTER(_BITMAPINFO), ctypes.c_uint,
        ctypes.POINTER(ctypes.c_void_p), ctypes.c_void_p, ctypes.c_uint32]
    gdi32.CreateCompatibleDC.restype = ctypes.c_void_p
    gdi32.CreateCompatibleDC.argtypes = [ctypes.c_void_p]
    gdi32.SelectObject.restype = ctypes.c_void_p
    gdi32.SelectObject.argtypes = [ctypes.c_void_p, ctypes.c_void_p]
    gdi32.CreateSolidBrush.restype = ctypes.c_void_p
    gdi32.CreateSolidBrush.argtypes = [ctypes.c_uint32]
    user32.FillRect.argtypes = [ctypes.c_void_p, ctypes.POINTER(_RECT),
                                ctypes.c_void_p]
    gdi32.DeleteDC.argtypes = [ctypes.c_void_p]
    gdi32.DeleteObject.argtypes = [ctypes.c_void_p]
    user32.DrawIconEx.argtypes = [
        ctypes.c_void_p, ctypes.c_int, ctypes.c_int, ctypes.c_void_p,
        ctypes.c_int, ctypes.c_int, ctypes.c_uint, ctypes.c_void_p,
        ctypes.c_uint]
    user32.DestroyIcon.argtypes = [ctypes.c_void_p]

    shfi = _SHFILEINFOW()
    flags = 0x100 | (0x1 if size <= 16 else 0x0)  # SHGFI_ICON small/large
    if not shell32.SHGetFileInfoW(path, 0, ctypes.byref(shfi),
                                  ctypes.sizeof(shfi), flags) \
            or not shfi.hIcon:
        return None
    try:
        bmi = _BITMAPINFO()
        bmi.bmiHeader.biSize = ctypes.sizeof(_BITMAPINFOHEADER)
        bmi.bmiHeader.biWidth = size
        bmi.bmiHeader.biHeight = -size        # negative = top-down rows
        bmi.bmiHeader.biPlanes = 1
        bmi.bmiHeader.biBitCount = 32
        bits = ctypes.c_void_p()
        hbm = gdi32.CreateDIBSection(None, ctypes.byref(bmi), 0,
                                     ctypes.byref(bits), None, 0)
        if not hbm:
            return None
        dc = gdi32.CreateCompatibleDC(None)
        old = gdi32.SelectObject(dc, hbm)
        brush = gdi32.CreateSolidBrush(               # COLORREF is BGR
            bg[0] | (bg[1] << 8) | (bg[2] << 16))
        rect = _RECT(0, 0, size, size)
        user32.FillRect(dc, ctypes.byref(rect), brush)
        gdi32.DeleteObject(brush)
        user32.DrawIconEx(dc, 0, 0, shfi.hIcon, size, size, 0, None, 3)
        gdi32.GdiFlush()
        raw = ctypes.string_at(bits, size * size * 4)
        gdi32.SelectObject(dc, old)
        gdi32.DeleteDC(dc)
        gdi32.DeleteObject(hbm)
        return _png_from_bgra(raw, size, size)
    finally:
        user32.DestroyIcon(shfi.hIcon)


# -- microphone mute via Windows Core Audio (raw COM, no extra packages) --

class _GUID(ctypes.Structure):
    _fields_ = [("d1", ctypes.c_ulong), ("d2", ctypes.c_ushort),
                ("d3", ctypes.c_ushort), ("d4", ctypes.c_ubyte * 8)]

    def __init__(self, s):
        super().__init__()
        ctypes.oledll.ole32.CLSIDFromString(s, ctypes.byref(self))


def _com_method(ptr, index, restype, *argtypes):
    """Bind method #index of a COM interface's vtable to a callable."""
    vtbl = ctypes.cast(ptr, ctypes.POINTER(ctypes.POINTER(ctypes.c_void_p)))
    proto = ctypes.WINFUNCTYPE(restype, ctypes.c_void_p, *argtypes)
    return proto(vtbl[0][index])


def toggle_mic_mute(probe=False):
    """
    Toggle the system-wide mute of the default microphone.
    With probe=True, just read the state. Returns True if the mic ends
    up muted. Raises OSError if the audio system can't be reached.
    """
    ole32 = ctypes.oledll.ole32
    try:
        ole32.CoInitialize(None)
    except OSError:
        pass                                  # already initialised is fine
    HRESULT, VOIDP = ctypes.HRESULT, ctypes.c_void_p
    enum_p, dev_p, vol_p = VOIDP(), VOIDP(), VOIDP()
    try:
        ole32.CoCreateInstance(
            ctypes.byref(_GUID("{BCDE0395-E52F-467C-8E3D-C4579291692E}")),
            None, 23,                         # CLSCTX_ALL
            ctypes.byref(_GUID("{A95664D2-9614-4F35-A746-DE8DB63617E6}")),
            ctypes.byref(enum_p))
        # IMMDeviceEnumerator::GetDefaultAudioEndpoint(eCapture, eConsole)
        _com_method(enum_p, 4, HRESULT, ctypes.c_int, ctypes.c_int,
                    ctypes.POINTER(VOIDP))(enum_p, 1, 0, ctypes.byref(dev_p))
        # IMMDevice::Activate(IAudioEndpointVolume, CLSCTX_ALL)
        _com_method(dev_p, 3, HRESULT, ctypes.POINTER(_GUID), ctypes.c_ulong,
                    VOIDP, ctypes.POINTER(VOIDP))(
            dev_p, ctypes.byref(
                _GUID("{5CDF2C82-841E-4546-9722-0CF74078229A}")),
            23, None, ctypes.byref(vol_p))
        muted = ctypes.c_int()
        _com_method(vol_p, 15, HRESULT, ctypes.POINTER(ctypes.c_int))(
            vol_p, ctypes.byref(muted))       # GetMute
        state = bool(muted.value)
        if not probe:
            state = not state
            _com_method(vol_p, 14, HRESULT, ctypes.c_int, VOIDP)(
                vol_p, int(state), None)      # SetMute
        return state
    finally:
        for p in (vol_p, dev_p, enum_p):
            if p:
                _com_method(p, 2, ctypes.c_ulong)(p)   # IUnknown::Release


class SpeakerMonitor:
    """
    Polls the default speaker's output peak (Windows Core Audio
    IAudioMeterInformation) on a background thread. `active()` is True for
    a short while after the speakers push sound past a small threshold, so
    the detector can drop taps that are really the mic hearing the PC's own
    audio. No stream is captured - only the level meter is read.
    """

    THRESHOLD = 0.02       # output peak (0..1) that counts as "playing"
    HANGOVER = 0.18        # keep suppressing this long after it goes quiet
    POLL = 0.015

    def __init__(self):
        self._loud_until = 0.0
        self._peak = 0.0
        self._stop = threading.Event()
        self._thread = None
        self.ok = sys.platform.startswith("win")

    def start(self):
        if not self.ok or self._thread is not None:
            return
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def stop(self):
        self._stop.set()

    def active(self):
        return time.monotonic() < self._loud_until

    @property
    def peak(self):
        return self._peak

    def _make_meter(self):
        ole32 = ctypes.oledll.ole32
        HRESULT, VOIDP = ctypes.HRESULT, ctypes.c_void_p
        enum_p, dev_p, meter_p = VOIDP(), VOIDP(), VOIDP()
        ole32.CoCreateInstance(
            ctypes.byref(_GUID("{BCDE0395-E52F-467C-8E3D-C4579291692E}")),
            None, 23,
            ctypes.byref(_GUID("{A95664D2-9614-4F35-A746-DE8DB63617E6}")),
            ctypes.byref(enum_p))
        # GetDefaultAudioEndpoint(eRender=0, eConsole=0)
        _com_method(enum_p, 4, HRESULT, ctypes.c_int, ctypes.c_int,
                    ctypes.POINTER(VOIDP))(enum_p, 0, 0, ctypes.byref(dev_p))
        # Activate(IAudioMeterInformation)
        _com_method(dev_p, 3, HRESULT, ctypes.POINTER(_GUID), ctypes.c_ulong,
                    VOIDP, ctypes.POINTER(VOIDP))(
            dev_p, ctypes.byref(
                _GUID("{C02216F6-8C67-4B5B-9D00-D008E73E0064}")),
            23, None, ctypes.byref(meter_p))
        for p in (dev_p, enum_p):
            _com_method(p, 2, ctypes.c_ulong)(p)
        return meter_p

    def _run(self):
        ole32 = ctypes.oledll.ole32
        try:
            ole32.CoInitialize(None)
        except OSError:
            pass
        HRESULT = ctypes.HRESULT
        meter = None
        peak = ctypes.c_float()
        while not self._stop.is_set():
            try:
                if meter is None:
                    meter = self._make_meter()
                _com_method(meter, 3, HRESULT,
                            ctypes.POINTER(ctypes.c_float))(
                    meter, ctypes.byref(peak))
                self._peak = float(peak.value)
                if self._peak >= self.THRESHOLD:
                    self._loud_until = time.monotonic() + self.HANGOVER
            except Exception:
                if meter is not None:
                    try:
                        _com_method(meter, 2, ctypes.c_ulong)(meter)
                    except Exception:
                        pass
                meter = None                 # rebuild (e.g. device changed)
                time.sleep(0.2)
            self._stop.wait(self.POLL)
        if meter is not None:
            try:
                _com_method(meter, 2, ctypes.c_ulong)(meter)
            except Exception:
                pass


class SpeakerLoopback:
    """
    Captures the actual audio the default speakers are playing (a WASAPI
    loopback, via the `soundcard` package) on a background thread, and
    flags the brief moments the PC produces a *new* sound - an onset.

    Unlike SpeakerMonitor (which suppresses every tap while the speakers
    are loud), this only marks the instant a fresh sound starts. A real
    finger tap makes no sound in the loopback, so tapping over steady
    music still registers; only mic transients that line up with the PC's
    own sound get dropped. Approximate, but it lets you tap while audio
    plays.
    """

    ONSET_RATIO = 2.2     # block this much louder than the trailing avg = onset
    ONSET_FLOOR = 2e-3    # ignore near-silent "onsets"
    HOLD = 0.16           # an onset marks echo active for this long (covers
    #                       the speaker -> air -> mic travel + stream skew)

    def __init__(self):
        self.ok = _soundcard is not None
        self._active_until = 0.0
        self._avg = 1e-4
        self._peak = 0.0
        self._stop = threading.Event()
        self._thread = None
        self._err = None

    def start(self):
        if not self.ok or self._thread is not None:
            return
        self._stop.clear()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def stop(self):
        self._stop.set()

    def active(self):
        return time.monotonic() < self._active_until

    @property
    def peak(self):
        return self._peak

    def _run(self):
        if sys.platform.startswith("win"):
            try:
                ctypes.oledll.ole32.CoInitialize(None)  # soundcard needs COM
            except OSError:
                pass
        try:
            spk = _soundcard.default_speaker()
            mic = _soundcard.get_microphone(str(spk.name),
                                            include_loopback=True)
            with mic.recorder(samplerate=SAMPLE_RATE, channels=1,
                              blocksize=BLOCK) as rec:
                while not self._stop.is_set():
                    data = rec.record(numframes=BLOCK)
                    rms = rms_of(np.asarray(data[:, 0], dtype=float))
                    self._peak = rms
                    if rms > self.ONSET_FLOOR and \
                            rms > self._avg * self.ONSET_RATIO:
                        self._active_until = time.monotonic() + self.HOLD
                    self._avg = 0.85 * self._avg + 0.15 * rms
        except Exception as e:               # device gone, no loopback, etc.
            self._err = repr(e)


def run_quick_action(key):
    """Perform a built-in quick action. Returns a short log message."""

    def need_pyautogui():
        if pyautogui is None:
            raise RuntimeError(
                "pyautogui is not installed - pip install pyautogui")

    if key == "minimize":
        need_pyautogui()
        pyautogui.hotkey("win", "d")
        return ("toggled minimize / maximize of all windows (repeat the "
                "pattern to flip it back).")
    if key == "mute_speakers":
        need_pyautogui()
        pyautogui.press("volumemute")
        return "toggled speaker mute."
    if key == "play_pause":
        need_pyautogui()
        pyautogui.press("playpause")
        return "play / pause."
    if key == "next_track":
        need_pyautogui()
        pyautogui.press("nexttrack")
        return "next track."
    if key == "prev_track":
        need_pyautogui()
        pyautogui.press("prevtrack")
        return "previous track."
    if key == "volume_up":
        need_pyautogui()
        pyautogui.press("volumeup", presses=5)
        return "volume up."
    if key == "volume_down":
        need_pyautogui()
        pyautogui.press("volumedown", presses=5)
        return "volume down."
    if key == "screenshot":
        need_pyautogui()
        pyautogui.hotkey("win", "printscreen")
        return "screenshot saved to Pictures > Screenshots."
    if key == "voice_typing":
        need_pyautogui()
        pyautogui.hotkey("win", "h")
        return "voice typing on (Win+H) - speak away."
    if key == "lock":
        if not sys.platform.startswith("win"):
            raise RuntimeError("locking is only supported on Windows")
        ctypes.windll.user32.LockWorkStation()
        return "locking the screen."
    if key == "mute_mic":
        if not sys.platform.startswith("win"):
            raise RuntimeError("mic mute is only supported on Windows")
        if toggle_mic_mute():
            return ("microphone MUTED. Taps can't be heard while it's "
                    "muted - unmute from Sound settings or tap the "
                    "taskbar mic icon.")
        return "microphone unmuted."
    raise RuntimeError(f"unknown action '{key}'")


# ------------------------------------------ app icon + Windows shortcuts

def make_app_icon_image(size=256):
    """Draw the app's mic-in-range-rings mark at the given size (RGBA)."""
    if pystray is None:                     # PIL rides in with pystray
        return None
    img = PILImage.new("RGBA", (size, size), (0, 0, 0, 0))
    d = PILImageDraw.Draw(img)
    s = size
    d.rounded_rectangle((s * 0.03, s * 0.03, s * 0.97, s * 0.97),
                        radius=s * 0.22, fill=(15, 23, 42, 255))
    cx, cy = s / 2, s * 0.45
    accent = (34, 211, 238, 255)
    ring = (45, 70, 110, 255)
    lw = max(1, int(s * 0.012))
    for r in (s * 0.29, s * 0.40):
        d.ellipse((cx - r, cy - r, cx + r, cy + r), outline=ring, width=lw)
    w, top, bot = s * 0.13, cy - s * 0.17, cy + s * 0.05
    d.rounded_rectangle((cx - w / 2, top, cx + w / 2, bot),
                        radius=w / 2, fill=accent)
    r2, aw = s * 0.11, max(2, int(s * 0.022))
    d.arc((cx - r2, cy - r2 * 0.7, cx + r2, cy + r2 * 1.2),
          start=0, end=180, fill=accent, width=aw)
    d.line((cx, cy + r2 * 1.2, cx, cy + s * 0.19), fill=accent, width=aw)
    d.line((cx - s * 0.06, cy + s * 0.19, cx + s * 0.06, cy + s * 0.19),
           fill=accent, width=aw)
    return img


def ensure_app_icon():
    """Write tap_launcher.ico next to the script (once) and return it."""
    path = os.path.join(APP_DIR, "tap_launcher.ico")
    if os.path.exists(path):
        return path
    img = make_app_icon_image(256)
    if img is None:
        return None
    try:
        img.save(path, format="ICO",
                 sizes=[(16, 16), (32, 32), (48, 48), (64, 64),
                        (128, 128), (256, 256)])
        return path
    except Exception:
        return None


def _known_dir(csidl):
    buf = ctypes.create_unicode_buffer(260)
    ctypes.windll.shell32.SHGetFolderPathW(None, csidl, None, 0, buf)
    return buf.value


def _launch_spec():
    """(exe, args) that starts this app - prefers a console-less pythonw,
    or the frozen .exe itself when packaged."""
    if getattr(sys, "frozen", False):
        return sys.executable, ""
    exe = sys.executable
    pw = os.path.join(os.path.dirname(exe), "pythonw.exe")
    if os.path.exists(pw):
        exe = pw
    return exe, f'"{os.path.abspath(__file__)}"'


def create_shortcut(link_path, target, args="", workdir="", icon=""):
    """Write a .lnk shortcut via IShellLinkW (no extra packages)."""
    ole32 = ctypes.oledll.ole32
    try:
        ole32.CoInitialize(None)
    except OSError:
        pass
    HRESULT, VOIDP = ctypes.HRESULT, ctypes.c_void_p
    link_p, pf_p = VOIDP(), VOIDP()
    try:
        ole32.CoCreateInstance(
            ctypes.byref(_GUID("{00021401-0000-0000-C000-000000000046}")),
            None, 1,
            ctypes.byref(_GUID("{000214F9-0000-0000-C000-000000000046}")),
            ctypes.byref(link_p))                       # IShellLinkW
        _com_method(link_p, 20, HRESULT, ctypes.c_wchar_p)(
            link_p, target)                             # SetPath
        if args:
            _com_method(link_p, 11, HRESULT, ctypes.c_wchar_p)(
                link_p, args)                           # SetArguments
        if workdir:
            _com_method(link_p, 9, HRESULT, ctypes.c_wchar_p)(
                link_p, workdir)                        # SetWorkingDirectory
        if icon:
            _com_method(link_p, 17, HRESULT, ctypes.c_wchar_p, ctypes.c_int)(
                link_p, icon, 0)                        # SetIconLocation
        _com_method(link_p, 0, HRESULT, ctypes.POINTER(_GUID),
                    ctypes.POINTER(VOIDP))(
            link_p,
            ctypes.byref(_GUID("{0000010B-0000-0000-C000-000000000046}")),
            ctypes.byref(pf_p))                         # QI -> IPersistFile
        _com_method(pf_p, 6, HRESULT, ctypes.c_wchar_p, ctypes.c_int)(
            pf_p, link_path, 1)                         # IPersistFile::Save
    finally:
        for p in (pf_p, link_p):
            if p:
                _com_method(p, 2, ctypes.c_ulong)(p)


def install_shortcut(link_path):
    """Create our shortcut at link_path (Desktop or Startup)."""
    exe, args = _launch_spec()
    create_shortcut(link_path, exe, args, APP_DIR, ensure_app_icon() or exe)


# ----------------------------------------------------------------------- UI

THEMES = {
    "dark": {
        "ACCENT": "#22d3ee", "BG_DARK": "#0f172a", "BG_MID": "#1e293b",
        "BG_EDGE": "#334155", "BG_DEEP": "#020617", "FG_MAIN": "#e2e8f0",
        "FG_DIM": "#94a3b8", "FG_HINT": "#64748b", "FG_FAINT": "#475569",
        "GOOD": "#4ade80", "WARN": "#fbbf24", "SELECT_BG": "#155e75",
        "SELECT_FG": "#ffffff", "LOG_FG": "#cbd5e1",
        "RING_COL": "#1d2b45", "DISABLED_FG": "#475569",
    },
    "light": {
        "ACCENT": "#0e7490", "BG_DARK": "#eef2f7", "BG_MID": "#ffffff",
        "BG_EDGE": "#c4d0de", "BG_DEEP": "#dfe7f0", "FG_MAIN": "#1e293b",
        "FG_DIM": "#526073", "FG_HINT": "#7a8699", "FG_FAINT": "#8fa0b5",
        "GOOD": "#15803d", "WARN": "#b45309", "SELECT_BG": "#bae6fd",
        "SELECT_FG": "#0f172a", "LOG_FG": "#334155",
        "RING_COL": "#d6dfeb", "DISABLED_FG": "#9aa8ba",
    },
}
CURRENT_THEME = "dark"


def set_theme(name):
    """Point the module-level palette (which all drawing reads live) at
    the named theme."""
    global CURRENT_THEME
    CURRENT_THEME = name if name in THEMES else "dark"
    for k, v in THEMES[CURRENT_THEME].items():
        globals()[k] = v


set_theme("dark")


def apply_theme(root):
    """Style tk + every ttk widget class from the current palette."""
    style = ttk.Style(root)
    try:
        style.theme_use("clam")
    except Exception:
        pass
    root.configure(bg=BG_DARK)
    style.configure(".", background=BG_DARK, foreground=FG_MAIN,
                    bordercolor=BG_EDGE, darkcolor=BG_DARK,
                    lightcolor=BG_DARK, troughcolor=BG_MID,
                    fieldbackground=BG_MID, insertcolor=FG_MAIN,
                    selectbackground=SELECT_BG, selectforeground=SELECT_FG,
                    focuscolor=ACCENT)
    style.configure("TLabelframe.Label", foreground=ACCENT)
    style.configure("Hint.TLabel", foreground=FG_HINT)
    style.configure("Dim.TLabel", foreground=FG_DIM)
    style.configure("Warn.TLabel", foreground=WARN)
    style.configure("Head.TLabel", foreground=ACCENT,
                    font=("Segoe UI", 10, "bold"))
    style.configure("Accent.TButton", background=ACCENT, foreground=BG_DARK)
    style.map("Accent.TButton",
              background=[("active", _mix_color(ACCENT, "#ffffff", 0.15)),
                          ("pressed", _mix_color(ACCENT, BG_DARK, 0.2))],
              foreground=[("disabled", DISABLED_FG)])
    style.configure("TButton", background=BG_MID, padding=4)
    style.map("TButton",
              background=[("disabled", BG_DARK), ("pressed", BG_EDGE),
                          ("active", BG_EDGE)],
              foreground=[("disabled", DISABLED_FG)])
    for w in ("TCheckbutton", "TRadiobutton"):
        style.configure(w, indicatorbackground=BG_MID,
                        indicatorforeground=ACCENT)
        style.map(w, background=[("active", BG_DARK)])
    style.map("TEntry",
              fieldbackground=[("disabled", BG_DARK)],
              foreground=[("disabled", DISABLED_FG)])
    style.configure("TSpinbox", background=BG_MID, arrowcolor=FG_MAIN,
                    fieldbackground=BG_MID)
    style.configure("TCombobox", arrowcolor=FG_MAIN)
    style.map("TCombobox",
              fieldbackground=[("readonly", BG_MID), ("disabled", BG_DARK)],
              foreground=[("disabled", DISABLED_FG)],
              selectbackground=[("readonly", BG_MID)],
              selectforeground=[("readonly", FG_MAIN)])
    style.configure("Horizontal.TScale", background=FG_HINT,
                    troughcolor=BG_MID)
    for sb in ("TScrollbar", "Vertical.TScrollbar", "Horizontal.TScrollbar"):
        style.configure(sb, background=BG_EDGE, troughcolor=BG_DARK,
                        arrowcolor=FG_MAIN, bordercolor=BG_DARK)
        style.map(sb, background=[("active", FG_HINT)])
    style.configure("Treeview", background=BG_MID, foreground=FG_MAIN,
                    fieldbackground=BG_MID)
    style.map("Treeview", background=[("selected", SELECT_BG)],
              foreground=[("selected", SELECT_FG)])
    style.configure("TNotebook", background=BG_DARK, borderwidth=0)
    style.configure("TNotebook.Tab", background=BG_MID, foreground=FG_DIM,
                    padding=(14, 6))
    style.map("TNotebook.Tab",
              background=[("selected", BG_EDGE)],
              foreground=[("selected", ACCENT)])
    # the combobox dropdown is a plain tk listbox; theme it via options
    root.option_add("*TCombobox*Listbox.background", BG_MID)
    root.option_add("*TCombobox*Listbox.foreground", FG_MAIN)
    root.option_add("*TCombobox*Listbox.selectBackground", SELECT_BG)
    root.option_add("*TCombobox*Listbox.selectForeground", SELECT_FG)


def _mix_color(c1, c2, f):
    """Blend two #rrggbb colours; f=0 gives c1, f=1 gives c2."""
    a = [int(c1[i:i + 2], 16) for i in (1, 3, 5)]
    b = [int(c2[i:i + 2], 16) for i in (1, 3, 5)]
    return "#%02x%02x%02x" % tuple(int(x + (y - x) * f)
                                   for x, y in zip(a, b))


def enable_dark_title_bar(window, dark=None):
    """Match the Windows title bar to the theme (no-op elsewhere)."""
    if not sys.platform.startswith("win"):
        return
    if dark is None:
        dark = CURRENT_THEME == "dark"
    try:
        window.update_idletasks()
        hwnd = ctypes.windll.user32.GetParent(window.winfo_id())
        for attr in (20, 19):   # DWMWA_USE_IMMERSIVE_DARK_MODE (19 pre-20H1)
            if ctypes.windll.dwmapi.DwmSetWindowAttribute(
                    hwnd, attr, ctypes.byref(ctypes.c_int(int(dark))),
                    ctypes.sizeof(ctypes.c_int)) == 0:
                break
    except Exception:
        pass


class AppPicker(tk.Toplevel if tk else object):
    """Searchable list of installed apps, with their icons."""

    def __init__(self, parent, apps, on_pick, icon_for=None):
        super().__init__(parent)
        self.title("Choose an app")
        self.geometry("440x520")
        self.configure(bg=BG_DARK)
        self.transient(parent)
        self.grab_set()
        enable_dark_title_bar(self)
        self.apps = apps
        self.on_pick = on_pick
        self.icon_for = icon_for
        self._paths = {}

        frame = ttk.Frame(self, padding=8)
        frame.pack(fill="both", expand=True)
        ttk.Label(frame, text="Search:").pack(anchor="w")
        self.search_var = tk.StringVar()
        ent = ttk.Entry(frame, textvariable=self.search_var)
        ent.pack(fill="x", pady=(2, 6))
        ent.focus_set()
        self.search_var.trace_add("write", lambda *_a: self._refill())

        style = ttk.Style(self)
        style.configure("AppPicker.Treeview", rowheight=36)
        box = ttk.Frame(frame)
        box.pack(fill="both", expand=True)
        self.tree = ttk.Treeview(box, show="tree", selectmode="browse",
                                 style="AppPicker.Treeview")
        sb = ttk.Scrollbar(box, orient="vertical", command=self.tree.yview)
        self.tree.configure(yscrollcommand=sb.set)
        self.tree.pack(side="left", fill="both", expand=True)
        sb.pack(side="right", fill="y")
        self.tree.bind("<Double-Button-1>", self._choose)
        self.tree.bind("<Return>", self._choose)
        ent.bind("<Return>", self._choose)

        btns = ttk.Frame(frame)
        btns.pack(fill="x", pady=(6, 0))
        ttk.Button(btns, text="Select", command=self._choose
                   ).pack(side="left")
        ttk.Button(btns, text="Cancel", command=self.destroy
                   ).pack(side="left", padx=6)
        self._refill()

    def _refill(self):
        self.tree.delete(*self.tree.get_children())
        self._paths = {}
        q = self.search_var.get().strip().lower()
        for name, path in self.apps:
            if q not in name.lower():
                continue
            icon = self.icon_for(path) if self.icon_for else None
            kwargs = {"image": icon} if icon else {}
            iid = self.tree.insert("", "end", text=" " + name, **kwargs)
            self._paths[iid] = path
        kids = self.tree.get_children()
        if kids:
            self.tree.selection_set(kids[0])

    def _choose(self, _evt=None):
        sel = self.tree.selection() or self.tree.get_children()
        if not sel:
            return
        self.on_pick(self._paths[sel[0]])
        self.destroy()


class TapLauncherApp:
    def __init__(self, root):
        self.root = root
        self.cfg = load_config()
        set_theme(self.cfg.get("theme") or "dark")

        self.events = queue.Queue()
        self.detector = TapDetector(self.events.put)
        self.detector.sensitivity = float(self.cfg["sensitivity"])
        self.detector.group_window = float(self.cfg["multi_tap_window"])
        self.detector.tap_max_dur = float(self.cfg["tap_max_dur"])
        self.detector.shape_gate = bool(self.cfg.get("shape_gate", True))
        if self.cfg.get("band"):
            self.detector.set_band(tuple(self.cfg["band"]))
        self.detector.tap_templates = [
            np.asarray(t, dtype=float)
            for t in (self.cfg.get("tap_templates") or [])]
        self.detector.min_match = float(self.cfg.get("min_match") or 0.6)
        self.detector.reject_templates = [
            np.asarray(r, dtype=float)
            for r in (self.cfg.get("reject_templates") or [])]

        self.calibrator = None
        self._cal_mode = "full"
        self.learner = None
        self._neg_queue = []
        self.stream = None
        self._apps_cache = None
        self._icon_cache = {}
        self.level = 0.0
        self.thr = None
        self.state_txt = "starting"
        self.flash_until = 0.0
        self.big_text = ""
        self.big_until = 0.0
        self.radar_pings = []
        self.radar_text = ""
        self.radar_text_color = FG_DIM
        self.radar_text_until = 0.0

        self.speaker_mon = SpeakerMonitor()
        self.speaker_loop = SpeakerLoopback()
        self._spk_warn_at = 0.0

        self._build_ui()
        self._apply_theme_live()
        self._apply_speaker_gate()
        self.tray = None
        self._init_tray()
        self._start_stream(self.cfg.get("input_device"))
        self.root.after(30, self._poll)
        self.root.protocol("WM_DELETE_WINDOW", self._on_close)

    def _init_tray(self):
        if pystray is None:
            return
        img = PILImage.new("RGBA", (64, 64), (0, 0, 0, 0))
        d = PILImageDraw.Draw(img)
        d.rounded_rectangle((4, 4, 60, 60), radius=14, fill=(15, 23, 42, 255))
        d.ellipse((18, 18, 46, 46), outline=(34, 211, 238, 255), width=4)
        d.ellipse((27, 27, 37, 37), fill=(34, 211, 238, 255))
        menu = pystray.Menu(
            pystray.MenuItem("Open Tap Launcher",
                             lambda *_a: self.events.put(
                                 {"type": "tray", "cmd": "open"}),
                             default=True),
            pystray.MenuItem("Quit",
                             lambda *_a: self.events.put(
                                 {"type": "tray", "cmd": "quit"})))
        self.tray = pystray.Icon("tap_launcher", img, "Tap Launcher", menu)
        threading.Thread(target=self.tray.run, daemon=True).start()

    # -- menu + Windows integration ------------------------------------

    def _build_menu(self):
        win = sys.platform.startswith("win")
        menubar = tk.Menu(self.root)
        filem = tk.Menu(menubar, tearoff=0,
                        bg=BG_MID, fg=FG_MAIN, activebackground=SELECT_BG,
                        activeforeground=SELECT_FG)
        filem.add_command(label="Create desktop shortcut",
                          command=self._make_desktop_shortcut)
        self.startup_var = tk.BooleanVar(
            value=win and os.path.exists(self._startup_lnk()))
        filem.add_checkbutton(label="Start with Windows",
                              variable=self.startup_var, selectcolor=GOOD,
                              command=self._toggle_startup)
        self._file_menu = filem
        self._startup_idx = filem.index("end")   # the checkbutton's index
        filem.add_separator()
        filem.add_command(label="Quit", command=self._shutdown)
        self._update_startup_label()
        menubar.add_cascade(label="File", menu=filem)
        helpm = tk.Menu(menubar, tearoff=0,
                        bg=BG_MID, fg=FG_MAIN, activebackground=SELECT_BG,
                        activeforeground=SELECT_FG)
        helpm.add_command(label="About Tap Launcher",
                          command=self._show_about)
        menubar.add_cascade(label="Help", menu=helpm)
        self.root.config(menu=menubar)

    def _desktop_lnk(self):
        return os.path.join(_known_dir(0x0010), "Tap Launcher.lnk")

    def _startup_lnk(self):
        return os.path.join(_known_dir(0x0007), "Tap Launcher.lnk")

    def _update_startup_label(self):
        on = bool(self.startup_var.get())
        self._file_menu.entryconfigure(
            self._startup_idx,
            label=("Start with Windows      ✔ ON" if on
                   else "Start with Windows      (off)"))

    def _make_desktop_shortcut(self):
        if not sys.platform.startswith("win"):
            self._log("Desktop shortcuts are Windows-only.", "warn")
            return
        try:
            install_shortcut(self._desktop_lnk())
            self._log("Added a 'Tap Launcher' shortcut to your desktop - "
                      "double-click it any time to open the app.", "ok")
        except Exception as ex:
            self._log(f"Could not create the desktop shortcut: {ex}", "warn")

    def _toggle_startup(self):
        if not sys.platform.startswith("win"):
            self.startup_var.set(False)
            self._log("Run-at-startup is Windows-only.", "warn")
            return
        path = self._startup_lnk()
        try:
            if self.startup_var.get():
                install_shortcut(path)
                self._log("Tap Launcher will now start automatically when "
                          "you log in to Windows.", "ok")
            else:
                if os.path.exists(path):
                    os.remove(path)
                self._log("Tap Launcher will no longer start automatically "
                          "at login.", "ok")
        except Exception as ex:
            self.startup_var.set(os.path.exists(path))
            self._log(f"Could not change the startup setting: {ex}", "warn")
        self._update_startup_label()

    def _show_about(self):
        from tkinter import messagebox
        messagebox.showinfo(
            "About Tap Launcher",
            f"Tap Launcher  v{APP_VERSION}\n\n"
            "Turn taps on your desk into app launches and shortcuts.\n"
            "All audio is processed live in memory - nothing is recorded.\n\n"
            "Concept inspired by Holo:\n"
            "github.com/JustinGamer191/Holo",
            parent=self.root)

    # -- UI construction ----------------------------------------------

    def _build_ui(self):
        self.root.title("Tap Launcher")
        self.root.geometry("580x900")
        self.root.minsize(540, 680)
        apply_theme(self.root)
        enable_dark_title_bar(self.root)

        self._ico = ensure_app_icon()
        if self._ico:
            try:
                self.root.iconbitmap(self._ico)
            except Exception:
                pass
        self._build_menu()

        outer = ttk.Frame(self.root, padding=10)
        outer.pack(fill="both", expand=True)

        # live view: tap radar on the left, level/status canvas beside it
        top = ttk.Frame(outer)
        top.pack(fill="x")
        self.radar = tk.Canvas(top, width=210, height=210, bg=BG_DARK,
                               highlightthickness=0)
        self.radar.pack(side="left")
        self.canvas = tk.Canvas(top, height=210, bg=BG_DARK,
                                highlightthickness=0)
        self.canvas.pack(side="left", fill="both", expand=True)

        self.status_lbl = ttk.Label(outer, text="", style="Dim.TLabel")
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
        self.theme_btn = ttk.Button(dev, width=11,
                                    command=self._toggle_theme)
        self.theme_btn.pack(side="left", padx=(6, 0))

        spk = ttk.Labelframe(outer, text="Ignore my computer's own audio",
                             padding=(8, 2))
        spk.pack(fill="x", pady=(2, 0))
        self.spk_gate_var = tk.BooleanVar(
            value=bool(self.cfg.get("speaker_gate")))
        ttk.Checkbutton(
            spk, variable=self.spk_gate_var, command=self._on_speaker_gate,
            text="Pause detection while the speakers play  ·  simple, "
                 "reliable"
        ).pack(fill="x")
        self.echo_var = tk.BooleanVar(value=bool(self.cfg.get("echo_reject")))
        ttk.Checkbutton(
            spk, variable=self.echo_var, command=self._on_echo_reject,
            text="Cancel only the PC audio, keep tapping over it  ·  "
                 "experimental"
        ).pack(fill="x")

        # calibration / tuning / actions live in tabs
        nb = ttk.Notebook(outer)
        nb.pack(fill="x", pady=6)

        cal = ttk.Frame(nb, padding=8)
        nb.add(cal, text=" Calibration ")
        row1 = ttk.Frame(cal)
        row1.pack(fill="x")
        self.cal_btn = ttk.Button(row1, text="Calibrate for this surface",
                                  command=self._start_calibration)
        self.cal_btn.pack(side="left")
        ttk.Button(row1, text="Reset",
                   command=self._reset_calibration).pack(side="left", padx=6)
        self.band_lbl = ttk.Label(row1, text="")
        self.band_lbl.pack(side="left", padx=8)

        rowt = ttk.Frame(cal)
        rowt.pack(fill="x", pady=(6, 0))
        self.addtype_btn = ttk.Button(rowt, text="Add a tap type",
                                      command=self._add_tap_type)
        self.addtype_btn.pack(side="left")
        self.types_lbl = ttk.Label(rowt, text="")
        self.types_lbl.pack(side="left", padx=8)
        ttk.Label(rowt, style="Hint.TLabel",
                  text="teach it e.g. hard taps too - matching any counts"
                  ).pack(side="left", padx=4)
        self._update_types_lbl()

        rowb = ttk.Frame(cal)
        rowb.pack(fill="x", pady=(6, 0))
        ttk.Label(rowb, text="Listening band:").pack(side="left")
        self.band_lo_var = tk.StringVar()
        self.band_hi_var = tk.StringVar()
        lo_spin = ttk.Spinbox(rowb, from_=40, to=16000, increment=20,
                              width=6, textvariable=self.band_lo_var,
                              command=self._on_band_edit)
        lo_spin.pack(side="left", padx=(6, 0))
        ttk.Label(rowb, text="to").pack(side="left", padx=4)
        hi_spin = ttk.Spinbox(rowb, from_=40, to=16000, increment=20,
                              width=6, textvariable=self.band_hi_var,
                              command=self._on_band_edit)
        hi_spin.pack(side="left")
        ttk.Label(rowb, text="Hz").pack(side="left", padx=4)
        for s in (lo_spin, hi_spin):
            s.bind("<Return>", self._on_band_edit)
            s.bind("<FocusOut>", self._on_band_edit)
        ttk.Button(rowb, text="Full range",
                   command=self._band_full).pack(side="left", padx=6)
        ttk.Label(rowb, style="Hint.TLabel",
                  text="widen it if taps get missed"
                  ).pack(side="left", padx=4)
        self._sync_band_ui()

        row2 = ttk.Frame(cal)
        row2.pack(fill="x", pady=(6, 0))
        self.learn_btn = ttk.Button(row2, text="Learn a sound to ignore",
                                    command=self._start_learn)
        self.learn_btn.pack(side="left")
        self.autoneg_btn = ttk.Button(
            row2, text="Guided: keyboard + voice",
            command=self._start_learn_guided)
        self.autoneg_btn.pack(side="left", padx=6)
        self.clear_btn = ttk.Button(row2, text="Clear ignored",
                                    command=self._clear_ignored)
        self.clear_btn.pack(side="left", padx=6)
        self._update_clear_btn()

        rowp = ttk.Frame(cal)
        rowp.pack(fill="x", pady=(6, 0))
        ttk.Label(rowp, text="Surface profile:").pack(side="left")
        self.profile_var = tk.StringVar(
            value=self.cfg.get("active_profile") or "")
        self.profile_box = ttk.Combobox(rowp, textvariable=self.profile_var,
                                        width=16)
        self.profile_box.pack(side="left", padx=6)
        self.profile_box.bind("<<ComboboxSelected>>", self._on_profile_load)
        ttk.Button(rowp, text="Save", width=6,
                   command=self._profile_save).pack(side="left")
        ttk.Button(rowp, text="Delete", width=7,
                   command=self._profile_delete).pack(side="left", padx=6)
        ttk.Label(rowp, style="Hint.TLabel",
                  text="type a name, Save stores this calibration"
                  ).pack(side="left", padx=4)
        self._refresh_profiles()

        # tuning: preset button groups instead of raw sliders
        tune = ttk.Frame(nb, padding=8)
        nb.add(tune, text=" Tuning ")

        self.sens_level_var = tk.StringVar(
            value=self.cfg.get("sensitivity_level", "normal"))
        self._preset_row(tune, 0, "Tap sensitivity", SENS_LEVELS,
                         self.sens_level_var, self._on_sens_level,
                         "softer taps register <-> fewer false triggers")

        self.multitap_var = tk.StringVar(
            value=self.cfg.get("multi_tap_level", "medium"))
        self._preset_row(tune, 2, "Gap between taps", MULTITAP_LEVELS,
                         self.multitap_var, self._on_multitap,
                         "how long apart taps can be and still group")

        self.taplen_var = tk.StringVar(
            value=self.cfg.get("tap_len_level", "short"))
        self._preset_row(tune, 4, "Max tap length", TAPLEN_LEVELS,
                         self.taplen_var, self._on_taplen,
                         "raise this if taps get ignored as 'too long'")

        self.gap_var = tk.StringVar(value=self.cfg.get("gap_level", "medium"))
        self._preset_row(tune, 6, "Close / Far distance", GAP_LEVELS,
                         self.gap_var, self._on_gap,
                         "loudness gap a distance move needs (smaller = "
                         "easier)")

        ttk.Label(tune, text="Tap match strictness"
                  ).grid(row=8, column=0, sticky="w", pady=(8, 0))
        self.match_var = tk.DoubleVar(
            value=float(self.cfg.get("min_match") or 0.6))
        ttk.Scale(tune, from_=0.20, to=0.95, variable=self.match_var,
                  command=self._on_match).grid(row=8, column=1, sticky="ew",
                                               padx=8, pady=(8, 0))
        self.match_lbl = ttk.Label(tune, width=16)
        self.match_lbl.grid(row=8, column=2, sticky="e", pady=(8, 0))

        self.shape_var = tk.BooleanVar(
            value=bool(self.cfg.get("shape_gate", True)))
        ttk.Checkbutton(
            tune, variable=self.shape_var, command=self._on_shape_gate,
            text="Only accept sharp, tap-like sounds (rejects sustained "
                 "noise like speech)"
        ).grid(row=9, column=0, columnspan=3, sticky="w", pady=(10, 0))

        tune.columnconfigure(1, weight=1)
        self._on_match()

        # actions / app assignment
        acts = ttk.Frame(nb, padding=8)
        nb.add(acts, text=" Actions ")

        self.enable_var = tk.BooleanVar(value=bool(self.cfg["actions_enabled"]))
        ttk.Checkbutton(
            acts, variable=self.enable_var, command=self._on_toggles,
            text="Run actions on tap patterns (leave off while testing)"
        ).grid(row=0, column=0, columnspan=4, sticky="w", pady=(0, 6))

        self.action_vars, self.target_vars, self.target_rows = {}, {}, {}
        self.arg_vars = {}
        row = 1
        for key in PATTERN_KEYS:
            pat = self.cfg["patterns"].get(key) or {}
            ttk.Label(acts, text=PATTERN_NAMES[key]
                      ).grid(row=row, column=0, sticky="w")
            avar = tk.StringVar(value=ACTIONS.get(pat.get("action", "none"),
                                                  ACTIONS["none"]))
            self.action_vars[key] = avar
            combo = ttk.Combobox(acts, textvariable=avar, state="readonly",
                                 values=list(ACTIONS.values()))
            combo.grid(row=row, column=1, columnspan=2, sticky="ew", padx=6)
            combo.bind("<<ComboboxSelected>>",
                       lambda _e, k=key: self._on_action_change(k))
            ttk.Button(acts, text="Test", width=8,
                       command=lambda k=key: self._test_pattern(k)
                       ).grid(row=row, column=3, padx=(4, 0))
            row += 1

            tvar = tk.StringVar(value=pat.get("target", ""))
            self.target_vars[key] = tvar
            entry = ttk.Entry(acts, textvariable=tvar)
            entry.grid(row=row, column=1, sticky="ew", padx=6, pady=(0, 4))
            pick = ttk.Button(acts, text="Pick app", width=9,
                              command=lambda k=key: self._pick_app(k))
            pick.grid(row=row, column=2, pady=(0, 4))
            browse = ttk.Button(acts, text="Browse", width=8,
                                command=lambda k=key: self._browse(k))
            browse.grid(row=row, column=3, padx=(4, 0), pady=(0, 4))
            tvar.trace_add("write",
                           lambda *_a, k=key: self._on_target_change(k))
            row += 1

            wvar = tk.StringVar(value=pat.get("arg", ""))
            self.arg_vars[key] = wvar
            arg_lbl = ttk.Label(acts, text="then open:", style="Dim.TLabel")
            arg_lbl.grid(row=row, column=0, sticky="e", padx=(0, 2))
            arg_entry = ttk.Entry(acts, textvariable=wvar)
            arg_entry.grid(row=row, column=1, columnspan=3, sticky="ew",
                           padx=6, pady=(0, 4))
            wvar.trace_add("write",
                           lambda *_a, k=key: self._on_arg_change(k))
            self.target_rows[key] = (entry, pick, browse,
                                     arg_lbl, arg_entry)
            row += 1

        ttk.Label(acts, style="Hint.TLabel", wraplength=500,
                  justify="left",
                  text="Close·Far = a loud tap then a soft one - either "
                       "move the second tap away from the mic, or just tap "
                       "it softer in the same spot (it's judged by loudness, "
                       "so both read the same). Far·Close is the reverse."
                  ).grid(row=row, column=0, columnspan=4, sticky="w")
        row += 1

        if pyautogui is None:
            ttk.Label(acts, style="Warn.TLabel",
                      text="pip install pyautogui to enable most quick "
                           "actions"
                      ).grid(row=row, column=0, columnspan=4, sticky="w",
                             pady=(6, 0))
            row += 1
        acts.columnconfigure(1, weight=1)
        for key in PATTERN_KEYS:
            self._sync_target_row(key)

        # footer pinned to the bottom (packed first so the log shrinks
        # around it instead of pushing it off-screen): live status left,
        # product name right
        footer = ttk.Frame(outer)
        footer.pack(side="bottom", fill="x", pady=(8, 0))
        self.footer_lbl = ttk.Label(footer, text="", style="Hint.TLabel")
        self.footer_lbl.pack(side="left")
        ttk.Label(footer, text=f"Tap Launcher  v{APP_VERSION}",
                  style="Hint.TLabel").pack(side="right")
        self._footer_cache = None

        # activity log with its own header row + a Clear button
        logf = ttk.Frame(outer)
        logf.pack(side="bottom", fill="both", expand=True, pady=(8, 0))
        loghead = ttk.Frame(logf)
        loghead.pack(fill="x")
        ttk.Label(loghead, text="Activity", style="Head.TLabel"
                  ).pack(side="left")
        ttk.Button(loghead, text="Clear", width=7,
                   command=self._clear_log).pack(side="right")
        self.log = tk.Text(logf, height=7, state="disabled", bg=BG_DEEP,
                           fg=LOG_FG, relief="flat", wrap="word",
                           padx=8, pady=6, insertbackground=FG_MAIN,
                           selectbackground=SELECT_BG,
                           highlightthickness=0)
        self.log.pack(fill="both", expand=True, pady=(4, 0))
        self.log.tag_configure("ok", foreground=GOOD)
        self.log.tag_configure("warn", foreground=WARN)

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

    def _on_sens_level(self):
        self.cfg["sensitivity_level"] = self.sens_level_var.get()
        self.cfg["sensitivity"] = effective_sensitivity(self.cfg)
        self.detector.sensitivity = self.cfg["sensitivity"]
        save_config(self.cfg)

    def _sync_band_ui(self):
        band = self.cfg.get("band")
        if band:
            self.band_lo_var.set(f"{band[0]:.0f}")
            self.band_hi_var.set(f"{band[1]:.0f}")
            self.band_lbl.config(text=f"Band: {band[0]:.0f}-"
                                      f"{band[1]:.0f} Hz")
        else:
            self.band_lo_var.set("")
            self.band_hi_var.set("")
            self.band_lbl.config(text="Listening to all frequencies")

    def _on_band_edit(self, _evt=None):
        try:
            lo = float(self.band_lo_var.get())
            hi = float(self.band_hi_var.get())
        except (TypeError, ValueError):
            return
        lo = min(max(40.0, lo), 15000.0)
        hi = min(max(lo + 100.0, hi), 16000.0)
        band = [round(lo, 1), round(hi, 1)]
        if band == self.cfg.get("band"):
            return
        self.cfg["band"] = band
        save_config(self.cfg)
        self.detector.set_band((lo, hi))
        self._sync_band_ui()

    def _band_full(self):
        if self.cfg.get("band") is None:
            return
        self.cfg["band"] = None
        save_config(self.cfg)
        self.detector.set_band(None)
        self._sync_band_ui()
        self._log("Band filter off - listening to all frequencies.")

    def _preset_row(self, parent, row, label, levels, var, cb, hint):
        ttk.Label(parent, text=label).grid(row=row, column=0, sticky="w",
                                           pady=(8, 0))
        box = ttk.Frame(parent)
        box.grid(row=row, column=1, columnspan=2, sticky="w", padx=8,
                 pady=(8, 0))
        for key, (lbl, _v) in levels.items():
            ttk.Radiobutton(box, text=lbl, value=key, variable=var,
                            command=cb).pack(side="left", padx=(0, 12))
        ttk.Label(parent, style="Hint.TLabel", text=hint).grid(
            row=row + 1, column=0, columnspan=3, sticky="w")

    def _apply_preset(self, lvlkey, level):
        levels, numkey, dfl = PRESET_GROUPS[lvlkey]
        val = _level_value(levels, level, dfl)
        self.cfg[lvlkey] = level
        self.cfg[numkey] = val
        save_config(self.cfg)
        return val

    def _on_multitap(self):
        self.detector.group_window = self._apply_preset(
            "multi_tap_level", self.multitap_var.get())

    def _on_taplen(self):
        self.detector.tap_max_dur = self._apply_preset(
            "tap_len_level", self.taplen_var.get())

    def _on_gap(self):
        self._apply_preset("gap_level", self.gap_var.get())

    def _on_shape_gate(self):
        on = bool(self.shape_var.get())
        self.detector.shape_gate = on
        self.cfg["shape_gate"] = on
        save_config(self.cfg)
        self._log("Now only sharp, tap-like onsets register." if on else
                  "Onset-shape filter off - any short sound can trigger.",
                  "ok")

    def _on_toggles(self):
        self.cfg["actions_enabled"] = bool(self.enable_var.get())
        save_config(self.cfg)

    def _apply_speaker_gate(self):
        """Build the detector's suppress_fn from whichever of the two
        speaker options are on (gate = pause on any output; echo reject =
        drop only taps that coincide with a fresh speaker sound)."""
        fns = []
        if bool(self.cfg.get("speaker_gate")):
            self.speaker_mon.start()
            fns.append(self.speaker_mon.active)
        if bool(self.cfg.get("echo_reject")):
            self.speaker_loop.start()
            fns.append(self.speaker_loop.active)
        self.detector.suppress_fn = (
            (lambda: any(f() for f in fns)) if fns else None)

    def _on_speaker_gate(self):
        on = bool(self.spk_gate_var.get())
        self.cfg["speaker_gate"] = on
        save_config(self.cfg)
        self._apply_speaker_gate()
        if on and not self.speaker_mon.ok:
            self._log("Speaker gate is Windows-only; it won't do anything "
                      "here.", "warn")
        else:
            self._log("Taps are now ignored while your PC speakers are "
                      "playing." if on else
                      "Speaker gate off.", "ok")

    def _on_echo_reject(self):
        on = bool(self.echo_var.get())
        self.cfg["echo_reject"] = on
        save_config(self.cfg)
        self._apply_speaker_gate()
        if on and not self.speaker_loop.ok:
            self.echo_var.set(False)
            self.cfg["echo_reject"] = False
            save_config(self.cfg)
            self._apply_speaker_gate()
            self._log("Echo rejection needs the 'soundcard' package - run: "
                      "pip install soundcard, then restart.", "warn")
        elif on:
            self._log("Echo rejection on - taps that line up with your PC's "
                      "own audio are dropped, but you can still tap while "
                      "it plays. Approximate; toggle off if it misbehaves.",
                      "ok")
        else:
            self._log("Echo rejection off.", "ok")

    def _sync_target_row(self, key):
        action = LABEL_TO_ACTION.get(self.action_vars[key].get(), "none")
        for widget in self.target_rows[key]:
            if action in APP_ACTIONS:
                widget.grid()
            else:
                widget.grid_remove()

    def _toggle_theme(self):
        set_theme("light" if CURRENT_THEME == "dark" else "dark")
        self.cfg["theme"] = CURRENT_THEME
        save_config(self.cfg)
        self._apply_theme_live()

    def _apply_theme_live(self):
        apply_theme(self.root)
        enable_dark_title_bar(self.root)
        self.canvas.config(bg=BG_DARK)
        self.radar.config(bg=BG_DARK)
        self.log.config(bg=BG_DEEP, fg=LOG_FG, insertbackground=FG_MAIN,
                        selectbackground=SELECT_BG)
        self.log.tag_configure("ok", foreground=GOOD)
        self.log.tag_configure("warn", foreground=WARN)
        self.theme_btn.config(text="Light mode" if CURRENT_THEME == "dark"
                              else "Dark mode")
        self._footer_cache = None  # re-apply footer colour for the new theme
        self._icon_cache.clear()   # icons re-render on the new field colour

    def _on_action_change(self, key):
        action = LABEL_TO_ACTION.get(self.action_vars[key].get(), "none")
        self.cfg["patterns"][key]["action"] = action
        save_config(self.cfg)
        self._sync_target_row(key)
        if action == "mute_mic":
            self._log("Heads up: while the mic is muted at the system "
                      "level I can't hear taps either, so unmute from the "
                      "taskbar or Sound settings, not by tapping.")

    def _on_target_change(self, key):
        self.cfg["patterns"][key]["target"] = self.target_vars[key].get()
        save_config(self.cfg)

    def _on_arg_change(self, key):
        self.cfg["patterns"][key]["arg"] = self.arg_vars[key].get()
        save_config(self.cfg)

    def _pick_app(self, key):
        if self._apps_cache is None:
            self._apps_cache = list_installed_apps()
        if not self._apps_cache:
            self._log("No Start Menu shortcuts found - use Browse instead.")
            return
        AppPicker(self.root, self._apps_cache,
                  lambda path: self.target_vars[key].set(path),
                  icon_for=self._app_icon)

    def _app_icon(self, path):
        if path not in self._icon_cache:
            img = None
            try:
                png = shell_icon_png(path, 32)
                if png:
                    img = tk.PhotoImage(
                        data=base64.b64encode(png).decode("ascii"))
            except Exception:
                img = None
            self._icon_cache[path] = img
        return self._icon_cache[path]

    def _browse(self, key):
        ft = [("Programs", "*.exe"), ("All files", "*.*")] \
            if sys.platform.startswith("win") else [("All files", "*.*")]
        path = filedialog.askopenfilename(
            title=f"App for {PATTERN_NAMES.get(key, key)}", filetypes=ft)
        if path:
            self.target_vars[key].set(path)

    def _refresh_profiles(self):
        self.profile_box["values"] = sorted(self.cfg.get("profiles") or {})

    def _profile_save(self):
        name = self.profile_var.get().strip()
        if not name:
            self._log("Type a name in the profile box first (e.g. 'desk'), "
                      "then hit Save.", "warn")
            return
        self.cfg["profiles"][name] = {k: self.cfg.get(k)
                                      for k in PROFILE_KEYS}
        self.cfg["active_profile"] = name
        save_config(self.cfg)
        self._refresh_profiles()
        self._log(f"Saved this calibration as profile '{name}'.", "ok")

    def _on_profile_load(self, _evt=None):
        name = self.profile_var.get()
        prof = (self.cfg.get("profiles") or {}).get(name)
        if not prof:
            return
        for k in PROFILE_KEYS:
            self.cfg[k] = prof.get(k)
        if not self.cfg.get("tap_templates") and prof.get("tap_template"):
            self.cfg["tap_templates"] = [prof["tap_template"]]
        self.cfg["tap_templates"] = list(self.cfg.get("tap_templates")
                                         or [])
        self.cfg["reject_templates"] = list(
            self.cfg.get("reject_templates") or [])
        self.cfg["active_profile"] = name
        self.cfg["sensitivity"] = effective_sensitivity(self.cfg)
        save_config(self.cfg)
        det = self.detector
        det.set_band(tuple(self.cfg["band"]) if self.cfg.get("band")
                     else None)
        det.tap_templates = [np.asarray(t, dtype=float)
                             for t in self.cfg["tap_templates"]]
        det.reject_templates = [np.asarray(r, dtype=float)
                                for r in self.cfg["reject_templates"]]
        det.sensitivity = self.cfg["sensitivity"]
        self.match_var.set(float(self.cfg.get("min_match") or 0.6))
        self._on_match()
        self._sync_band_ui()
        self._update_clear_btn()
        self._update_types_lbl()
        self._log(f"Loaded profile '{name}'.", "ok")

    def _profile_delete(self):
        name = self.profile_var.get().strip()
        if name in (self.cfg.get("profiles") or {}):
            del self.cfg["profiles"][name]
            if self.cfg.get("active_profile") == name:
                self.cfg["active_profile"] = None
            save_config(self.cfg)
            self._refresh_profiles()
            self.profile_var.set("")
            self._log(f"Deleted profile '{name}'.", "ok")

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
        self.autoneg_btn.state(state)
        self.addtype_btn.state(state)
        if not busy:
            self._update_clear_btn()

    def _update_types_lbl(self):
        n = len(self.cfg.get("tap_templates") or [])
        self.types_lbl.config(
            text=f"Tap types: {n}" if n
            else "Tap types: none (any tap counts)")

    def _update_clear_btn(self):
        n = len(self.cfg.get("reject_templates") or [])
        self.clear_btn.config(text=f"Clear ignored ({n})")
        self.clear_btn.state(["!disabled"] if n else ["disabled"])

    def _clear_log(self):
        self.log.configure(state="normal")
        self.log.delete("1.0", "end")
        self.log.configure(state="disabled")

    def _update_footer(self):
        st = self.state_txt
        if st in ("no audio", "mic error"):
            txt, col = "● Microphone unavailable", WARN
        elif st == "calibrating":
            txt, col = "● Calibrating…", ACCENT
        elif st in ("learning", "learning ignore"):
            txt, col = "● Learning…", ACCENT
        else:
            txt, col = "● Listening", GOOD
        if self._footer_cache != (txt, col):
            self._footer_cache = (txt, col)
            self.footer_lbl.config(text=txt, foreground=col)

    def _start_learn(self):
        self._neg_queue = []
        self.learner = RejectLearner()
        self._set_busy(True)
        self._log("Learning a sound to ignore - make the noise "
                  "(e.g. type normally) for about 6 seconds...")

    def _start_learn_guided(self):
        # walk through the usual culprits back to back, one template each
        self._neg_queue = [("your voice", "talk normally")]
        self.learner = RejectLearner(prompt="type on your keyboard")
        self._set_busy(True)
        self._log("Guided ignore (1/2): type on your keyboard normally for "
                  "about 6 seconds...")

    def _clear_ignored(self):
        self.cfg["reject_templates"] = []
        save_config(self.cfg)
        self.detector.reject_templates = []
        self._update_clear_btn()
        self._log("Cleared all learned ignore-sounds.")

    def _start_calibration(self):
        self._cal_mode = "full"
        self.calibrator = Calibrator()
        self._set_busy(True)
        self._log("Calibration started - stay quiet for a second...")

    def _add_tap_type(self):
        self._cal_mode = "add"
        self.calibrator = Calibrator()
        self._set_busy(True)
        self._log("Adding a tap type - stay quiet a moment, then tap 5 "
                  "times the NEW way (harder, knuckle, other spot...).")

    def _reset_calibration(self):
        self.cfg["band"] = None
        self.cfg["tap_templates"] = []
        save_config(self.cfg)
        self.detector.set_band(None)
        self.detector.tap_templates = []
        self._sync_band_ui()
        self._update_types_lbl()
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
            self._radar_ping("tap", e.get("peak"), e.get("thr"))
            sim = e.get("sim")
            extra = f", {sim * 100:.0f}% match" if sim is not None else ""
            self._log(f"Tap{extra}  ({e['count_so_far']} so far...)", "ok")
        elif kind == "rejected":
            self._radar_ping("rejected", e.get("peak"), e.get("thr"))
            if e.get("reason") == "speaker":
                if now - self._spk_warn_at > 6.0:
                    self._log("Ignored a sound while the PC speakers were "
                              "playing (speaker gate is on).", "warn")
                    self._spk_warn_at = now
            elif e.get("reason") == "shape":
                self._log(f"Rejected - onset too soft/slow to be a tap "
                          f"({e.get('attack', 0) * 1000:.0f} ms rise; taps "
                          f"snap up in a few ms). Turn off 'sharp sounds "
                          f"only' in Tuning if this was a real tap.", "warn")
            elif e.get("reason") == "ignored":
                self._log(f"Ignored - {e['rej_sim'] * 100:.0f}% match to a "
                          f"learned ignore-sound (only "
                          f"{e['sim'] * 100:.0f}% like your taps).", "warn")
            else:
                self._log(f"Rejected - only {e['sim'] * 100:.0f}% similar "
                          f"to your calibrated taps (strictness "
                          f"{self.detector.min_match * 100:.0f}%).", "warn")
        elif kind == "group":
            n = e["n"]
            key = str(n)
            label = {1: "SINGLE TAP", 2: "DOUBLE TAP",
                     3: "TRIPLE TAP", 4: "QUADRUPLE TAP"}.get(n, f"{n} TAPS")
            if n == 2:
                peaks = e.get("peaks") or []
                need = float(self.cfg.get("near_far_db") or NEAR_FAR_DB)
                trend = tap_trend(peaks, need)
                if trend and (self.cfg["patterns"].get(trend) or {}
                              ).get("action", "none") != "none":
                    key = trend
                    label = {"close-far": "CLOSE · FAR",
                             "far-close": "FAR · CLOSE"}[trend]
                if (len(peaks) == 2 and min(peaks) > 0
                        and self._distance_bound()):
                    db = 20.0 * float(np.log10(peaks[0] / peaks[1]))
                    word = "softer (farther)" if db >= 0 \
                        else "louder (closer)"
                    self._log(f"Distance read: 2nd tap {abs(db):.0f} dB "
                              f"{word}; a move needs {need:.0f} dB.")
            self.big_text, self.big_until = label, now + 1.3
            self._run_group(key, label)
        elif kind == "tray":
            if e["cmd"] == "open":
                self.root.deiconify()
                self.root.lift()
            else:
                self._shutdown()
        elif kind == "ignored":
            self._log(f"Heard a sound ({e['duration'] * 1000:.0f} ms) - "
                      f"too long to be a tap. Ignored.", "warn")
        elif kind == "noise":
            self._log("Sustained noise - ignoring it and resetting.", "warn")

    def _test_pattern(self, key):
        self._run_group(key, f"Test {PATTERN_NAMES[key].lower()}",
                        force=True)

    def _run_group(self, key, label, force=False):
        if key == "1":
            self._log("Single tap (no action is bound to one tap).")
            return
        pat = self.cfg["patterns"].get(key) or {}
        action = pat.get("action", "none")
        if action == "none":
            self._log(f"{label} - nothing assigned yet.")
            return
        target = (pat.get("target") or "").strip()
        arg = (pat.get("arg") or "").strip()
        if action in APP_ACTIONS:
            if not target:
                self._log(f"{label} - no app chosen yet (use Pick app "
                          f"or Browse).")
                return
            desc = f"launch {os.path.basename(target)}"
            if arg:
                desc += f" with {normalize_open_arg(arg)}"
            if action == "app_dictate":
                desc += " + voice typing"
        else:
            desc = ACTIONS.get(action, action).lower()
        if not force and not self.cfg["actions_enabled"]:
            self._log(f"{label} - would {desc} (actions are turned off).")
            return
        try:
            if action in APP_ACTIONS:
                launch_target(target, arg)
                shown = os.path.basename(target)
                if arg:
                    shown += f" -> {normalize_open_arg(arg)}"
                if action == "app_dictate":
                    self._log(f"{label} - launching {shown}, voice typing "
                              f"once it has focus...", "ok")
                    self._await_focus_then_dictate()
                else:
                    self._log(f"{label} - launching {shown}", "ok")
            else:
                self._log(f"{label} - {run_quick_action(action)}", "ok")
        except Exception as ex:
            self._log(f"{label} - action failed: {ex}", "warn")

    def _await_focus_then_dictate(self, tries=0, hwnd0=None):
        """Press Win+H once the launched app takes the foreground
        (or after ~8 s if the focus never visibly changes)."""
        if pyautogui is None:
            self._log("Voice typing needs pyautogui.", "warn")
            return
        if not sys.platform.startswith("win"):
            return
        u32 = ctypes.windll.user32
        if hwnd0 is None:
            hwnd0 = u32.GetForegroundWindow()
        cur = u32.GetForegroundWindow()
        if (cur and cur != hwnd0) or tries >= 27:

            def fire():
                try:
                    pyautogui.hotkey("win", "h")
                    self._log("Voice typing on - speak away.", "ok")
                except Exception as ex:
                    self._log(f"Voice typing failed: {ex}", "warn")

            self.root.after(700, fire)      # give the window a beat
        else:
            self.root.after(300, lambda: self._await_focus_then_dictate(
                tries + 1, hwnd0))

    def _distance_bound(self):
        """True if either distance move has an action assigned."""
        return any((self.cfg["patterns"].get(k) or {}).get("action", "none")
                   != "none" for k in ("close-far", "far-close"))

    def _tick_calibration(self):
        cal = self.calibrator
        if cal is None:
            return
        self.status_lbl.config(text=cal.status)
        if cal.phase == "done" and cal.result:
            band = cal.result["band"]
            if self._cal_mode == "add":
                # merge: new fingerprint joins the list, band widens to
                # cover both tap types, sensitivity stays as tuned
                old = self.cfg.get("band")
                if old:
                    band = [min(old[0], band[0]), max(old[1], band[1])]
                self.cfg["band"] = band
                self.cfg["tap_templates"] = (
                    list(self.cfg.get("tap_templates") or [])
                    + [cal.result["tap_template"]])
                self.cfg["min_match"] = min(
                    float(self.cfg.get("min_match") or 0.6),
                    cal.result["min_match"])
            else:
                self.cfg["band"] = band
                self.cfg["sensitivity_base"] = cal.result["sensitivity"]
                self.cfg["sensitivity"] = effective_sensitivity(self.cfg)
                self.cfg["tap_templates"] = [cal.result["tap_template"]]
                self.cfg["min_match"] = cal.result["min_match"]
            save_config(self.cfg)
            self.detector.set_band(tuple(self.cfg["band"]))
            self.detector.sensitivity = self.cfg["sensitivity"]
            self.detector.tap_templates = [
                np.asarray(t, dtype=float)
                for t in self.cfg["tap_templates"]]
            self.match_var.set(self.cfg["min_match"])
            self._on_match()
            self._sync_band_ui()
            self._update_types_lbl()
            if getattr(cal, "rejected", 0):
                self._log(f"(Skipped {cal.rejected} low-quality tap"
                          f"{'s' if cal.rejected != 1 else ''} - clipped, "
                          f"faint, or inconsistent.)")
            n = len(self.cfg["tap_templates"])
            if self._cal_mode == "add":
                self._log(f"Added tap type #{n}. Band widened to "
                          f"{band[0]:.0f}-{band[1]:.0f} Hz - "
                          f"try the new tap.", "ok")
            else:
                self._log(f"Calibrated. Watching {band[0]:.0f}-"
                          f"{band[1]:.0f} Hz and fingerprinting your taps "
                          f"(strictness "
                          f"{self.cfg['min_match'] * 100:.0f}%) - try a "
                          f"double tap.", "ok")
            self.calibrator = None
            self._set_busy(False)
            self.status_lbl.config(text="")
        elif cal.phase == "failed":
            self._log(cal.status, "warn")
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
                      f"total).", "ok")
            self.learner = None
            if not self._next_negative():
                self._set_busy(False)
                self.status_lbl.config(text="")
        elif lrn.phase == "failed":
            self._log(lrn.status, "warn")
            self.learner = None
            if not self._next_negative():
                self._set_busy(False)
                self.status_lbl.config(text="")

    def _next_negative(self):
        """Advance a guided keyboard->voice negative-capture run. Returns
        True if another step was started."""
        if not self._neg_queue:
            return False
        _name, prompt = self._neg_queue.pop(0)
        self.learner = RejectLearner(prompt=prompt)
        self._log(f"Guided ignore (2/2): {prompt} for about 6 seconds...")
        return True

    # -- drawing -------------------------------------------------------

    def _radar_ping(self, kind, peak=None, thr=None):
        """Record a heard tap for the radar: how hard it was and (from
        loudness alone - one mic can't tell direction) roughly how far."""
        now = time.monotonic()
        db = 0.0
        if peak and thr and peak > thr > 0:
            db = 20.0 * float(np.log10(peak / thr))
        frac = 1.0 - min(max((db - 6.0) / 30.0, 0.0), 1.0)
        if db >= 30:
            strength, color = "Hard", WARN
        elif db >= 16:
            strength, color = "Medium", GOOD
        else:
            strength, color = "Soft", ACCENT
        dist = ("very close" if frac < 0.3
                else "mid-range" if frac < 0.7 else "far / faint")
        if kind == "rejected":
            color = "#64748b"
            text = f"{strength} thump · {dist} · rejected"
        else:
            text = f"{strength} tap · {dist} · +{db:.0f} dB"
        self.radar_pings.append({"t": now, "kind": kind, "frac": frac,
                                 "color": color})
        self._set_radar_text(text, color)

    def _set_radar_text(self, text, color):
        self.radar_text = text
        self.radar_text_color = color
        self.radar_text_until = time.monotonic() + 3.0

    def _redraw_radar(self, now):
        c = self.radar
        c.delete("all")
        w = max(c.winfo_width(), 2)
        h = max(c.winfo_height(), 2)
        cx, cy = w / 2, h / 2 - 10
        max_r = min(w, h) / 2 - 26

        for f in (0.33, 0.66, 1.0):
            r = max_r * f
            c.create_oval(cx - r, cy - r, cx + r, cy + r,
                          outline=RING_COL, width=1)
        c.create_text(cx + max_r * 0.33 + 3, cy + 9, text="near",
                      fill=FG_FAINT, font=("Segoe UI", 7), anchor="w")
        c.create_text(cx + max_r - 24, cy + 9, text="far",
                      fill=FG_FAINT, font=("Segoe UI", 7), anchor="w")

        # mic symbol at the centre
        c.create_oval(cx - 7, cy - 17, cx + 7, cy + 3,
                      fill=ACCENT, outline="")
        c.create_arc(cx - 11, cy - 12, cx + 11, cy + 11, start=180,
                     extent=180, style="arc", outline=ACCENT, width=2)
        c.create_line(cx, cy + 11, cx, cy + 17, fill=ACCENT, width=2)
        c.create_line(cx - 6, cy + 17, cx + 6, cy + 17,
                      fill=ACCENT, width=2)

        life = 1.4
        keep = []
        for p in self.radar_pings:
            age = now - p["t"]
            if age > life:
                continue
            keep.append(p)
            fade = age / life
            frac = p["frac"] + 0.05 * fade       # slight outward drift
            r = 16 + frac * (max_r - 16)
            col = _mix_color(p["color"], BG_DARK, fade)
            c.create_oval(cx - r, cy - r, cx + r, cy + r,
                          outline=col, width=2)
        self.radar_pings = keep

        if now < self.radar_text_until:
            c.create_text(w / 2, h - 10, text=self.radar_text,
                          fill=self.radar_text_color, font=("Segoe UI", 8))
        else:
            c.create_text(w / 2, h - 10, text="tap and I'll place it here",
                          fill=FG_FAINT, font=("Segoe UI", 8))

    def _redraw(self, now):
        self._update_footer()
        self._redraw_radar(now)
        c = self.canvas
        c.delete("all")
        w = max(c.winfo_width(), 2)
        h = 210

        bg = BG_MID if now < self.flash_until else BG_DARK
        c.create_rectangle(0, 0, w, h, fill=bg, width=0)

        # level meter (log scale) with threshold marker
        pad, bar_y, bar_h = 16, h - 34, 16
        c.create_rectangle(pad, bar_y, w - pad, bar_y + bar_h,
                           fill=BG_DEEP, width=0)

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
            c.create_text(w / 2, h / 2 - 20, text=self.big_text,
                          fill=ACCENT, font=("Segoe UI", 21, "bold"))
        else:
            hint = {"calibrating": "Calibrating...",
                    "learning": "Learning the room's noise floor...",
                    "event": "Hearing something...",
                    "no audio": "Audio unavailable",
                    "mic error": "Microphone error - pick another device",
                    }.get(self.state_txt, "Listening for taps")
            c.create_text(w / 2, h / 2 - 20, text=hint, fill=FG_DIM,
                          font=("Segoe UI", 13))

        # status line
        noise = self.detector.noise
        parts = [f"state: {self.state_txt}"]
        if noise is not None:
            parts.append(f"noise: {to_db(noise):.0f} dB")
        if self.thr is not None:
            parts.append(f"threshold: {to_db(self.thr):.0f} dB")
        c.create_text(pad, 16, anchor="w", text="   |   ".join(parts),
                      fill=FG_FAINT, font=("Segoe UI", 9))

    def _log(self, msg, kind=None):
        stamp = time.strftime("%H:%M:%S")
        self.log.configure(state="normal")
        self.log.insert("end", f"[{stamp}] {msg}\n", kind or ())
        if int(self.log.index("end-1c").split(".")[0]) > 250:
            self.log.delete("1.0", "2.0")
        self.log.see("end")
        self.log.configure(state="disabled")

    def _on_close(self):
        if self.tray is not None:
            self.root.withdraw()      # keep listening from the tray
            try:
                self.tray.notify("Still listening for taps - right-click "
                                 "the tray icon to quit.")
            except Exception:
                pass
            return
        self._shutdown()

    def _shutdown(self):
        try:
            if self.stream is not None:
                self.stream.stop()
                self.stream.close()
        except Exception:
            pass
        self.speaker_mon.stop()
        self.speaker_loop.stop()
        if self.tray is not None:
            try:
                self.tray.stop()
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
