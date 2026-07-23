# Tap Launcher

A small desktop app that listens to your microphone for finger taps on a surface and turns them into actions. Each tap pattern — double, triple, quadruple, or a *distance move* like **Close · Far** (a loud tap then a soft one, whether by moving away from the mic or just tapping softer) — can either launch an app (picked from a searchable, icon-annotated list of your installed programs) or fire a built-in quick action: minimize / maximize all windows, mute the microphone or speakers, play/pause or skip tracks, nudge the volume, rotate the screen between portrait and landscape, lock the screen, take a screenshot, or start Windows voice typing (Win+H). The "Launch app + voice typing" action chains the two: it opens your browser (or any app), waits for its window to take focus, then starts dictation so you can just speak your search.

The UI ships with matching **dark and light themes** — the button next to the microphone picker flips between them (title bar included) and the choice is remembered. The window is organized into **Calibration / Tuning / Actions tabs**, with a live view up top: a level meter with the trigger threshold, and a **tap radar** — a mic symbol inside range rings where every sound the detector hears lands as a ring. Its distance from the mic reflects how close the tap probably was (one microphone can't tell direction, only loudness, so nearness is estimated from how loud it arrived), its color shows how hard you hit (cyan soft, green medium, amber hard, gray for rejected sounds), and a readout spells it out, e.g. "Hard tap · very close · +32 dB".

Closing the window doesn't stop it: the app minimizes to a tray icon and keeps listening; double-click the icon to bring the window back, right-click for Quit. Calibrations can be stored as named **surface profiles** ("desk", "kitchen table") and swapped from a dropdown, so moving the mic doesn't mean recalibrating from scratch. The narration log is color-coded — green for taps and actions, amber for rejections and warnings.

All audio is processed live in memory. Nothing is recorded or written to disk.

The overall concept and a few of the noise-rejection ideas were inspired by **[Holo](https://github.com/JustinGamer191/Holo)** by JustinGamer191 — see [Credit & how this differs from Holo](#credit--how-this-differs-from-holo) at the bottom for exactly what was borrowed and what's original here.

## Setup

Python 3.9+ with tkinter (the standard python.org Windows installer includes it), then:

```
pip install numpy scipy sounddevice pyautogui pystray pillow soundcard
```

`pyautogui` powers most quick actions; `pystray` + `pillow` are only needed for the minimize-to-tray icon; `soundcard` is only needed for the experimental "ignore only my PC's own audio" option. Everything else works without them. On Windows, if the mic doesn't pick anything up, check Settings > Privacy & security > Microphone and allow desktop apps to use it.

Run it with:

```
python tap_launcher.py
```

### Desktop shortcut & starting with Windows

From the app's **File** menu:

- **Create desktop shortcut** drops a double-clickable "Tap Launcher" icon on your desktop (it launches without a console window).
- **Start with Windows** is a checkbox — tick it to have Tap Launcher open automatically when you log in, untick it to stop. It simply adds or removes a shortcut in your Startup folder.

Both use a generated `tap_launcher.ico` next to the script, which is also the window/taskbar icon.

Only one copy runs at a time: launching it again (double-clicking the icon a second time) just brings the existing window to the front — restoring it from the tray if it was hidden — instead of opening a duplicate. So a tap never fires an action several times over.

## First run

1. Pick your microphone from the dropdown if the default isn't the one near your tapping surface.
2. Click **Calibrate for this surface**. Stay quiet for about a second (it measures the room's noise floor), then tap five times on the surface you'll actually use. The app finds the frequency band where *your* taps on *that* surface put their energy, builds a band-pass filter around it, and picks a starting sensitivity between the noise floor and your tap loudness.
3. Try a double tap and a triple tap. The log at the bottom narrates everything it hears and how it interpreted it, the meter shows the live level against the yellow threshold line, and the big text flashes the recognized pattern.
4. When detection feels right, choose what each pattern does from its dropdown — **Launch an app or file** (use **Pick app** for a searchable list of installed programs with their icons, or **Browse** for any file) or one of the quick actions. App launches have an optional **then open** field: put a website or file there and it's handed to the app on launch, so Opera + `netflix.com` opens the browser straight on Netflix (bare domains get `https://` added automatically) — then tick **Run actions on tap patterns**. It's off by default so testing doesn't fire things constantly. Each pattern has a **Test** button that fires its action immediately (even while the checkbox is off), so you can check an assignment without tapping.

One caution on the mic-mute quick action: while the microphone is muted at the system level the app can't hear you either, so unmute from the taskbar or Sound settings rather than by tapping.

The Tuning tab is all preset buttons rather than fiddly sliders: **Tap sensitivity** (More / Normal / Less) on top of what calibration measured; **Gap between taps** (Tight / Medium / Relaxed) for how far apart taps can land and still group; **Max tap length** (Short / Medium / Long) for the longest burst still counted as a tap — bump this up if the log keeps saying "too long to be a tap. Ignored"; and **Close / Far distance** (Small / Medium / Large) for the loudness gap a distance move needs (Small = easiest to trigger). Only the fingerprint **Tap match strictness** stays a slider. The calibrated **listening band** is editable on the Calibration tab: widen it (or hit **Full range**) if taps go unnoticed.

Two checkboxes under the microphone stop the mic reacting to your computer's own audio (music, videos, game sound), from bluntest to smartest:

- **Ignore ALL taps while my PC speakers are playing** — watches the speakers' output level (Windows Core Audio) and pauses tap detection while sound is coming out. Simple and never misfires, but you can't tap while audio plays.
- **Ignore only my PC's own audio** (experimental, needs `soundcard`) — captures a loopback of what the speakers are actually playing and only drops mic transients that line up with a *fresh* sound from the PC. A real finger tap makes no sound in that loopback, so you can still tap over steady music; only sounds that coincide with the computer's own audio get rejected. It's approximate — a tap landing exactly on a drum hit may be dropped, and a mic with aggressive "enhancements" (AGC/noise suppression) distorts the echo and hurts accuracy — so it's a toggle you can switch off if it misbehaves.

The two can be combined, and both are off by default. Settings persist in `tap_launcher_config.json` next to the script.

## How it works

Audio comes in as 512-sample blocks (~12 ms each). Each block is band-pass filtered (after calibration) and reduced to an RMS energy value. A slowly adapting noise floor tracks the room, and the trigger threshold sits at *noise floor x sensitivity*, so it rides along if your environment gets louder or quieter.

When the level jumps above threshold, the detector watches how long it stays up. A burst that dies within the **Max tap length** (default ~120 ms) is a tap; anything longer is ignored and logged so you can see it happened. Taps arriving within the tap gap window are grouped, and when the window closes the group fires as a single/double/triple/quadruple pattern. With the speaker gate on, a tap-length burst heard while the speakers are outputting sound is dropped instead, logged as ignored.

There's also an **onset-shape gate** ("Only accept sharp, tap-like sounds" in Tuning, on by default): a finger tap snaps up to full volume in a couple of milliseconds, while speech, music and other sustained sounds swell in over tens of milliseconds. The detector measures the rising edge of each candidate on the raw waveform and rejects anything that ramps up too slowly to be a tap — surface- and mic-agnostic, so it complements the spectral fingerprint. If a real tap gets rejected for shape, the log says so (with the measured rise time) and you can switch the gate off.

Distance moves add one more layer: every tap's peak loudness is kept with its group, and when a two-tap group lands, the two peaks are compared. If the first tap is louder than the second by at least the **Close / Far distance** gap (Medium = 8 dB) it fires **Close · Far** (equivalently **Hard · Soft**); the reverse fires **Far · Close** / **Soft · Hard**; anything in between stays a plain double tap. Because it's judged purely by loudness, moving the second tap farther from the mic and simply tapping it softer read the same — do whichever is comfortable. The radar shows what each tap read as, and while a distance move is bound the log prints a "Distance read" line for every two-tap group. Patterns with no action bound don't divert: an uneven double tap still counts as a double unless you've assigned the matching move.

## Telling taps apart from keyboards and other noises

Loudness alone can't distinguish a finger tap from a keystroke — both are short transients that thump the desk, and typing naturally lands two or three sounds inside the multi-tap window. So on top of the energy detector, every tap candidate is *fingerprinted*: the ~93 ms around its loudest point is reduced to 48 log-spaced frequency-band energies, and that signature is correlated against a template of your real taps.

Calibration builds the template automatically from your five sample taps, and sets the starting **Tap match strictness** from how consistent those five were with each other. It also quality-checks the samples as it captures them: taps that clip (too loud), are too faint to fingerprint, or disagree with the others (an outlier — a stray cough or chair creak mixed into the five) are dropped so one bad sample can't poison the template. The log notes how many were skipped.

One template is deliberately specific — a hard tap sounds different from a soft one and can get rejected. **Add a tap type** fixes that: click it and tap five times the *other* way (harder, knuckle, a different spot) and a second fingerprint joins the first; a sound counts as a tap if it matches *any* learned type. Adding a type also widens the listening band to cover both and relaxes strictness to the pickier of the two calibrations, while leaving your sensitivity alone. Learn as many types as you need; **Calibrate for this surface** starts the list over from scratch and Reset clears it. A dull fingertip thud and a clicky keystroke have very different signatures, so most typing gets rejected outright — the log shows every rejection with its match percentage, e.g. "Rejected — only 34% similar to your calibrated taps".

For stubborn cases there's **Learn a sound to ignore**: click it, then type normally (or click your mouse, or whatever keeps sneaking through) for about six seconds. It fingerprints those transients into an ignore-template, and from then on any sound that resembles the ignored noise at least as much as it resembles your taps is dropped, logged as "Ignored". You can learn several different ignore-sounds; Clear ignored forgets them all. **Guided: keyboard + voice** runs that capture twice back-to-back — type for six seconds, then talk for six seconds — so the two most common false triggers get their own ignore-templates in one pass.

Tuning by log-watching works well: if real taps get rejected, lower the strictness toward their logged percentages; if junk gets through, raise it or teach the junk as an ignore-sound. Recalibrating replaces the template and resets strictness from the fresh taps.

## Tuning tips

If real taps are missed, switch sensitivity to **More sensitive**, widen the listening band, or lower the match strictness; if random noise triggers it, go **Less sensitive**, raise the strictness, teach the noise via Learn a sound to ignore, or recalibrate. Windows "audio enhancements" (noise suppression, AGC) on the mic can fight the detector by squashing exactly the transients it's looking for — if detection is flaky, try disabling enhancements for that device in Sound settings. A mic physically resting near or on the tapping surface works far better than one across the room, since the sound conducts through the desk.

A synthetic-audio test suite (`python test_detector.py`) covers the detector, calibrator, and reject-learner without needing a mic.

## Credit & how this differs from Holo

The overall concept — listening to a microphone for taps on a desk and turning them into actions — and several of the noise-rejection ideas came from **[Holo](https://github.com/JustinGamer191/Holo)** by JustinGamer191. Credit for that groundwork, and for being the jumping-off point for this project, goes there.

**Ideas borrowed from Holo:**

- **Sharp-onset detection.** Holo keys on a "short, high-contrast onset" and gates out sustained speech. That's the seed of the onset-shape gate here, which rejects sounds whose rising edge is too slow to be a real tap.
- **Calibration quality gates.** Holo rejects "weak, masked, or clipped" calibration taps and runs a leave-one-out agreement check across its samples. This app likewise skips clipped/faint samples during calibration and drops an outlier that disagrees with the rest, so one bad sample can't poison the template.
- **Negative-example collection.** Holo trains on negatives like typing and talking. Here that idea became the **Learn a sound to ignore** flow and the guided **keyboard + voice** capture.

**What's original to this project (and how it differs):**

- **Any microphone, any surface.** Holo is built for a single MacBook and forces its built-in mic — external inputs are rejected. This app was made specifically to *not* be laptop-specific: it has a microphone picker and is meant to work with whatever mic you point at whatever surface. That difference is the whole reason it exists as its own project.
- **No spatial zones — distance and count instead.** Holo classifies *which of four desk zones* a tap lands in, using a trained zone model (MFCC-style features, nearest-example novelty) and optional ultrasonic chirps for active sensing. This app doesn't attempt location at all. It distinguishes taps by **count** (double / triple / quadruple) and by loudness-based **distance moves** (Close·Far / Far·Close), and it identifies taps with a simpler spectral-fingerprint correlation plus multiple learnable tap types — passive only, no chirp, no per-zone training.
- **A whole application wrapped around it.** The launcher side is all original: the searchable app picker with real program icons and the optional "then open a website/file" field; the library of quick actions (mute mic/speakers, minimize/maximize, media and volume keys, lock, screenshot, Windows voice typing, and launch-app-then-dictate); named surface profiles; the dark/light theme; minimize-to-tray; the preset-button tuning; and the live **tap radar** that shows how hard and how near each tap read.
- **Handling the PC's own audio.** The two ways to stop the mic reacting to computer sound — the simple speaker gate (Windows Core Audio peak meter) and the experimental loopback-based echo rejection that still lets you tap while audio plays — are specific to this project.

In short: Holo is a laptop-specific, location-aware tap *classifier*; this is a mic-agnostic, action-focused tap *launcher* that borrowed Holo's instincts for telling taps from noise and went its own way on everything else.
