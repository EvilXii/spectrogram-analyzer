# SPECTRA

A desktop spectrogram analyzer, audio player and spectrum-watermark editor, built with PySide6.

## Features

- Rainbow spectrogram (violet = quiet … red = loud) with a dBFS colour legend
- Linear or logarithmic frequency axis
- Tools: **Select** (area), **Zoom box**, **Pan**, **Marker**
- File info panel (format, bitrate, sample rate, channels, peak/RMS, tags)
- Originality check: estimates the lowpass cutoff to flag likely lossy transcodes
- Player for the whole file or just the selected time range, with loop and seek
- **Spectrum watermark**: draw text (or a solid area) into the spectrogram by
  - *Add tones*: synthesises new tones, works even in silent areas
  - *Boost existing*: raises the music's own spectrum inside the letters
  - *Cut existing*: lowers it inside the letters
  - optional invert, and edit / move / resize / delete afterwards
- ID3 tags editor and text markers stored as ID3 chapters (MP3)
- Export audio (MP3 320k, FLAC, WAV) and spectrogram PNG

## Requirements

- Python 3.9+
- Packages in `requirements.txt`: PySide6, numpy, soundfile, mutagen
- **Optional:** [ffmpeg](https://ffmpeg.org/) on your `PATH`
  - decodes formats libsndfile can't read (e.g. M4A/AAC)
  - enables 320 kbps MP3 export (without it, MP3 export uses libsndfile's encoder)

## Installation

```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

## Usage

```bash
python spectra.py                # then open or drag & drop a file
python spectra.py song.mp3       # open a file directly
```

Supported input: `.mp3 .wav .flac .ogg .m4a .aac .opus`

### Controls

| Action | Input |
| --- | --- |
| Zoom time | Mouse wheel |
| Zoom frequency | Shift + wheel |
| Zoom both | Ctrl + wheel |
| Zoom back out | Right-click |
| Pan | Middle-drag, or the Pan tool |
| Play / pause | Space |
| Clear selection | Esc |

### Making a watermark

1. Pick the **Select** tool and drag an area on the spectrogram.
2. Open the **Watermark** tab, type your text (or choose *Solid area*) and pick a mode.
3. Click **Add watermark from selection**.
4. Select it in the list to edit, move, resize or delete it.
5. Click **Export audio…** to save the result.

> **Tip:** Low-bitrate MP3 encoders discard content above ~16 kHz. For
> high-frequency watermarks, export as 320k MP3, FLAC or WAV.

## Notes

- Watermark processing runs in a background thread, so playback continues while you edit.
- If the watermarked audio would clip, the level is lowered automatically.
- The originality check is a heuristic only; judge by ear and with other tools too.
- If Qt Multimedia isn't available, playback is disabled but all analysis and editing still works.
- "Write into MP3" and "Save tags" modify the original MP3's ID3 tag (the audio data is untouched).

## Troubleshooting

| Problem | Fix |
| --- | --- |
| "Cannot decode file" | Install ffmpeg and make sure it's on your `PATH` |
| No sound | Check your system's default audio output; ensure PySide6 includes QtMultimedia |
| MP3 export sounds dull | Install ffmpeg for 320 kbps encoding |
