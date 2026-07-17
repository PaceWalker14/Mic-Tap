# Tap Launcher

A small desktop app that listens to your microphone for finger taps on a surface and turns them into actions. Double-tap to launch one app, triple-tap for another. As an experimental extra, dragging your finger along the surface can scroll: away from the mic scrolls down, toward the mic jumps to the top of the page.

All audio is processed live in memory. Nothing is recorded or written to disk.

## Setup

Python 3.9+ with tkinter (the standard python.org Windows installer includes it), then:

```
pip install numpy scipy sounddevice pyautogui
```

`pyautogui` is only needed for the drag-scrolling feature; everything else works without it. On Windows, if the mic doesn't pick anything up, check Settings > Privacy & security > Microphone and allow desktop apps to use it.

Run it with:

```
python tap_launcher.py
```

## First run

1. Pick your microphone from the dropdown if the default isn't the one near your tapping surface.
2. Click **Calibrate for this surface**. Stay quiet for about a second (it measures the room's noise floor), then tap five times on the surface you'll actually use. The app finds the frequency band where *your* taps on *that* surface put their energy, builds a band-pass filter around it, and picks a starting sensitivity between the noise floor and your tap loudness.
3. Try a double tap and a triple tap. The log at the bottom narrates everything it hears and how it interpreted it, the meter shows the live level against the yellow threshold line, and the big text flashes the recognized pattern.
4. When detection feels right, browse for the .exe you want on each pattern and tick **Launch apps on tap patterns**. It's off by default so testing doesn't open things constantly.

The multi-tap window slider controls how close together taps must be to count as one pattern (default 400 ms). Settings persist in `tap_launcher_config.json` next to the script.

## How it works

Audio comes in as 512-sample blocks (~12 ms each). Each block is band-pass filtered (after calibration) and reduced to an RMS energy value. A slowly adapting noise floor tracks the room, and the trigger threshold sits at *noise floor x sensitivity*, so it rides along if your environment gets louder or quieter.

When the level jumps above threshold, the detector watches how long it stays up. A burst that dies within ~120 ms is a tap; sounds lasting longer than ~250 ms are treated as a drag; anything in between is ignored and logged so you can see it happened. Taps arriving within the multi-tap window are grouped, and when the window closes the group fires as a single/double/triple/quadruple pattern.

Drag direction comes from the volume trend of the friction noise: getting louder means your finger is approaching the mic, getting quieter means it's moving away. That's a genuinely rough heuristic — one microphone can't measure distance, only loudness — so expect misfires and keep it off unless you're playing with it. `_run_drag` in the code is where you'd swap the scroll amount or the "jump to top" key.

## Telling taps apart from keyboards and other noises

Loudness alone can't distinguish a finger tap from a keystroke — both are short transients that thump the desk, and typing naturally lands two or three sounds inside the multi-tap window. So on top of the energy detector, every tap candidate is *fingerprinted*: the ~93 ms around its loudest point is reduced to 48 log-spaced frequency-band energies, and that signature is correlated against a template of your real taps.

Calibration builds the template automatically from your five sample taps, and sets the starting **Tap match strictness** from how consistent those five were with each other. A dull fingertip thud and a clicky keystroke have very different signatures, so most typing gets rejected outright — the log shows every rejection with its match percentage, e.g. "Rejected — only 34% similar to your calibrated taps".

For stubborn cases there's **Learn a sound to ignore**: click it, then type normally (or click your mouse, or whatever keeps sneaking through) for about six seconds. It fingerprints those transients into an ignore-template, and from then on any sound that resembles the ignored noise at least as much as it resembles your taps is dropped, logged as "Ignored". You can learn several different ignore-sounds; Clear ignored forgets them all.

Tuning by log-watching works well: if real taps get rejected, lower the strictness toward their logged percentages; if junk gets through, raise it or teach the junk as an ignore-sound. Recalibrating replaces the template and resets strictness from the fresh taps.

## Tuning tips

If real taps are missed, drag sensitivity toward 2x or lower the match strictness; if random noise triggers it, push them higher, teach the noise via Learn a sound to ignore, or recalibrate. Windows "audio enhancements" (noise suppression, AGC) on the mic can fight the detector by squashing exactly the transients it's looking for — if detection is flaky, try disabling enhancements for that device in Sound settings. A mic physically resting near or on the tapping surface works far better than one across the room, since the sound conducts through the desk.

## Testing without a mic

`python test_detector.py` feeds synthetic audio through the detector, calibrator, and reject-learner: it checks that double/triple/single taps and both drag directions are recognized, that clicky "keystrokes" (which also thump the desk) are rejected by the tap fingerprint, and that a learned ignore-sound drops them with the right reason. Useful for checking logic changes without tapping at your desk like a woodpecker.
