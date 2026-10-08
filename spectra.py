#!/usr/bin/env python3
"""
SPECTRA - spectrogram analyzer, player and spectrum-watermark editor.

  * Rainbow spectrogram palette (violet = quiet ... red = loud)
  * Linear / logarithmic frequency axis
  * Tools: Select (area), Zoom box (drag an area to zoom in), Pan, Marker
    Wheel = zoom time, Shift+wheel = zoom frequency, Ctrl+wheel = both,
    Right-click = zoom back out, middle-drag = pan
  * File info panel + player (whole file or just the selected part)
  * Spectrum watermark: select an area, type text (or fill the whole area) and
    blend it into the audio as   Add tones / Boost existing / Cut existing,
    optionally inverted, resizable, editable, removable
  * ID3 tags + text markers (ID3 chapters), PNG export, audio export

Install:  pip install PySide6 numpy soundfile mutagen
Optional: ffmpeg on PATH (320 kbps MP3 export, decode fallback)
Run:      python spectra.py [file]
"""
import html as _html
import os
import shutil
import subprocess
import sys
import tempfile

import numpy as np
import soundfile as sf
from mutagen import File as MutagenFile
from mutagen.id3 import (CHAP, COMM, CTOC, ID3, TALB, TCOP, TIT2, TPE1, TXXX,
                         CTOCFlags, ID3NoHeaderError)
from mutagen.mp3 import MP3
from PySide6.QtCore import (QBuffer, QByteArray, QIODevice, QObject, QPointF,
                            QRectF, Qt, QThread, QTimer, Signal)
from PySide6.QtGui import (QColor, QFont, QFontMetrics, QImage, QKeySequence,
                           QLinearGradient, QPainter, QPen, QShortcut)
from PySide6.QtWidgets import (QApplication, QButtonGroup, QCheckBox, QComboBox,
                               QDoubleSpinBox, QFileDialog, QFormLayout,
                               QHBoxLayout, QInputDialog, QLabel, QLineEdit,
                               QListWidget, QMainWindow, QMessageBox,
                               QPushButton, QScrollArea, QTabWidget, QVBoxLayout,
                               QWidget)

try:
    from PySide6.QtMultimedia import QAudio, QAudioFormat, QAudioSink, QMediaDevices
    HAVE_AUDIO = True
except Exception:                      # multimedia backend not available
    HAVE_AUDIO = False

DB_MIN, DB_MAX = -120.0, 0.0
FMIN_LOG = 20.0
AUDIO_EXTS = (".mp3", ".wav", ".flac", ".ogg", ".m4a", ".aac", ".opus")

# Visible-light spectrum palette (matches the reference image):
# quiet = black/violet -> indigo -> blue -> cyan -> green -> yellow -> orange -> red = loud
RAINBOW_STOPS = [(0.00, (0, 0, 0)), (0.07, (45, 0, 90)), (0.18, (125, 0, 205)),
                 (0.30, (45, 60, 255)), (0.43, (0, 190, 255)), (0.56, (0, 225, 70)),
                 (0.70, (255, 235, 0)), (0.84, (255, 130, 0)), (1.00, (255, 0, 0))]


# --------------------------------------------------------------------------- #
# DSP helpers
# --------------------------------------------------------------------------- #
def build_lut(stops):
    xs = np.array([s[0] for s in stops])
    lut = np.zeros((256, 3), np.uint8)
    for c in range(3):
        lut[:, c] = np.interp(np.linspace(0, 1, 256), xs, [s[1][c] for s in stops])
    return lut


LUT = build_lut(RAINBOW_STOPS)


def decode_audio(path):
    """Return (samples[n, ch] float32, samplerate). Falls back to ffmpeg."""
    try:
        data, sr = sf.read(path, dtype="float32", always_2d=True)
        return data, sr
    except Exception as first_err:
        if not shutil.which("ffmpeg"):
            raise RuntimeError(f"Cannot decode file ({first_err}). "
                               "Install ffmpeg for wider format support.")
        with tempfile.TemporaryDirectory() as td:
            tmp = os.path.join(td, "d.wav")
            subprocess.run(["ffmpeg", "-y", "-i", path, tmp], check=True,
                           stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            return sf.read(tmp, dtype="float32", always_2d=True)


def compute_spectrogram(x, sr, n_fft=2048, max_cols=2000):
    """Magnitude spectrogram in dB (full-scale sine = 0 dB). Shape: (freq, time)."""
    if len(x) < n_fft:
        x = np.pad(x, (0, n_fft - len(x)))
    hop = max(n_fft // 4, len(x) // max_cols)
    n_frames = 1 + (len(x) - n_fft) // hop
    win = np.hanning(n_fft).astype(np.float32)
    ref = n_fft / 4.0
    out = np.empty((n_fft // 2 + 1, n_frames), np.float32)
    offs = np.arange(n_fft)
    for a in range(0, n_frames, 128):
        b = min(a + 128, n_frames)
        starts = np.arange(a, b) * hop
        frames = x[starts[:, None] + offs] * win
        mag = np.abs(np.fft.rfft(frames, axis=1)) / ref
        out[:, a:b] = (20 * np.log10(mag + 1e-12)).T
    return out


def spectrum_to_index(db):
    norm = np.clip((db - DB_MIN) / (DB_MAX - DB_MIN), 0, 1)
    return (norm * 255).astype(np.uint8)


def estimate_cutoff(db, sr, floor_db=-88.0):
    p = np.percentile(db, 90, axis=1)
    k = np.convolve(p, np.ones(5) / 5, mode="same")
    above = np.where(k > floor_db)[0]
    return 0.0 if len(above) == 0 else above[-1] / (len(p) - 1) * (sr / 2)


def verdict_for(cutoff, nyq):
    khz = cutoff / 1000
    if cutoff >= nyq - 400:
        return (f"Content reaches ~{khz:.1f} kHz (near Nyquist). "
                "No obvious lossy lowpass - consistent with a lossless source.")
    if khz < 11:
        return f"Sharp cutoff at ~{khz:.1f} kHz: very low-bitrate lossy source."
    if khz < 16.6:
        return (f"Cutoff at ~{khz:.1f} kHz: typical of 128 kbps (or lower) lossy "
                "encodes. If sold as lossless/high-quality, it is likely a transcode.")
    if khz < 19.0:
        return f"Cutoff at ~{khz:.1f} kHz: typical of ~160-192 kbps lossy encodes."
    if khz < 20.6:
        return (f"Cutoff at ~{khz:.1f} kHz: typical of 256-320 kbps MP3 or good AAC. "
                "Fine for MP3, suspicious if claimed to be lossless.")
    return f"Cutoff at ~{khz:.1f} kHz: very high; likely high-bitrate or lossless."


def build_info(path, data, sr):
    """List of (label, value) rows describing the loaded file."""
    rows = []
    size = os.path.getsize(path)
    dur = len(data) / sr
    ext = os.path.splitext(path)[1].lstrip(".").upper()
    fmt, bitrate, mode, encoder, bits = ext, 0, "", "", None
    try:
        mf = MutagenFile(path)
        info = mf.info if mf is not None else None
        if info is not None:
            bitrate = getattr(info, "bitrate", 0) or 0
            bits = getattr(info, "bits_per_sample", None)
            if isinstance(mf, MP3):
                fmt = f"MP3 · MPEG {info.version:g} Layer {info.layer}"
                mode = str(info.bitrate_mode).split(".")[-1]
                encoder = getattr(info, "encoder_info", "") or ""
    except Exception:
        pass
    if not bitrate and dur > 0:
        bitrate = int(size * 8 / dur)
    tags = {}
    try:
        easy = MutagenFile(path, easy=True)
        if easy is not None:
            for k in ("title", "artist", "album", "date", "genre"):
                if easy.get(k):
                    tags[k] = str(easy[k][0])
    except Exception:
        pass
    for k in ("title", "artist", "album", "date", "genre"):
        if k in tags:
            rows.append((k.capitalize(), tags[k]))
    rows.append(("File", os.path.basename(path)))
    rows.append(("Format", fmt))
    br = f"{bitrate // 1000} kbps" + (f" {mode}" if mode and mode != "UNKNOWN" else "")
    rows.append(("Bitrate", br))
    if encoder:
        rows.append(("Encoder", encoder))
    rows.append(("Sample rate", f"{sr:,} Hz"))
    rows.append(("Channels", {1: "Mono", 2: "Stereo"}.get(data.shape[1], str(data.shape[1]))))
    if bits:
        rows.append(("Bit depth", f"{bits}-bit"))
    m, s = divmod(dur, 60)
    rows.append(("Duration", f"{int(m)}:{s:04.1f}"))
    rows.append(("Size", f"{size / 1e6:.2f} MB"))
    peak = float(np.abs(data).max()) if data.size else 0.0
    rms = float(np.sqrt(np.mean(data.astype(np.float64) ** 2))) if data.size else 0.0
    rows.append(("Peak", f"{20 * np.log10(peak + 1e-9):.1f} dBFS"))
    rows.append(("RMS", f"{20 * np.log10(rms + 1e-9):.1f} dBFS"))
    return rows


# ---- watermark engine ------------------------------------------------------ #
def render_text_image(text, w, h, bold=True, size=0.8):
    """Rasterise text to float image (h x w), row 0 = top."""
    img = QImage(w, h, QImage.Format_Grayscale8)
    img.fill(0)
    p = QPainter(img)
    p.setRenderHint(QPainter.TextAntialiasing, True)
    p.setPen(QColor(255, 255, 255))
    px = max(int(h * size), 4)
    while True:
        f = QFont("Arial")
        f.setBold(bold)
        f.setPixelSize(px)
        if QFontMetrics(f).horizontalAdvance(text) <= w * 0.96 or px <= 4:
            break
        px -= 1
    p.setFont(f)
    p.drawText(QRectF(0, 0, w, h), Qt.AlignCenter, text)
    p.end()
    buf = np.frombuffer(img.constBits(), np.uint8).reshape(h, img.bytesPerLine())
    return buf[:, :w].astype(np.float32) / 255.0


def sample_mask(wm, freqs, times):
    """Sample the watermark shape on a (len(freqs) x len(times)) grid.
    Returns (mask 0..1, inside-rectangle boolean)."""
    t0, t1, f0, f1 = wm["t0"], wm["t1"], wm["f0"], wm["f1"]
    fw = (lambda f: np.log(np.maximum(f, 1.0))) if wm["log"] else (lambda f: f)
    yf = (fw(np.asarray(freqs, float)) - fw(f0)) / (fw(f1) - fw(f0))
    xf = (np.asarray(times, float) - t0) / (t1 - t0)
    inside = ((yf >= 0) & (yf <= 1))[:, None] & ((xf >= 0) & (xf <= 1))[None, :]
    if wm["shape"] == "solid":
        m = np.ones(inside.shape, np.float32)
    else:
        img = wm.get("_img")
        if img is None:
            H = 160
            W = int(np.clip(H * wm["aspect"], 32, 4000))
            img = render_text_image(wm["text"] or " ", W, H, wm["bold"], wm["size"] / 100.0)
        H, W = img.shape
        rows = np.clip(np.rint((1 - yf) * (H - 1)).astype(int), 0, H - 1)
        cols = np.clip(np.rint(xf * (W - 1)).astype(int), 0, W - 1)
        m = img[rows[:, None], cols[None, :]]
    m = m * inside
    if wm["invert"]:
        m = np.where(inside, 1.0 - m, 0.0).astype(np.float32)
    return m.astype(np.float32), inside


def prepare_wm(wm):
    """Pre-render the text bitmap on the GUI thread (workers must not touch fonts)."""
    if wm["shape"] != "text":
        wm.pop("_img", None)
        wm.pop("_key", None)
        return
    key = (wm["text"], wm["bold"], wm["size"], round(wm["aspect"], 3))
    if wm.get("_key") != key:
        H = 160
        W = int(np.clip(H * wm["aspect"], 32, 4000))
        wm["_img"] = render_text_image(wm["text"] or " ", W, H, wm["bold"], wm["size"] / 100.0)
        wm["_key"] = key


def synth_tones(sr, wm):
    """Mono tone signal (length of the watermark's time span) drawing the shape."""
    t0, t1 = wm["t0"], wm["t1"]
    dur = t1 - t0
    f0, f1 = wm["f0"], min(wm["f1"], sr / 2 - 100)
    if f1 <= f0 or dur <= 0:
        return np.zeros(0, np.float32)
    rows = int(np.clip((f1 - f0) / 45, 16, 200))
    if wm["log"]:
        freqs = np.geomspace(max(f0, 20.0), f1, rows)[::-1]
    else:
        freqs = np.linspace(f1, f0, rows)
    cols = max(int(dur * 80), 16)
    ts = np.linspace(t0, t1, cols)
    m, _ = sample_mask(wm, freqs, ts)
    n = int(dur * sr)
    tloc = np.arange(n) / sr
    colt = ts - t0
    rng = np.random.default_rng(1)
    sig = np.zeros(n, np.float32)
    for r in range(rows):
        if m[r].max() < 0.02:
            continue
        env = np.interp(tloc, colt, m[r]).astype(np.float32)
        ph = rng.uniform(0, 2 * np.pi)
        sig += env * np.sin(2 * np.pi * freqs[r] * tloc + ph).astype(np.float32)
    sig *= 10 ** (wm["amount"] / 20)
    fade = min(int(0.01 * sr), n // 2)
    if fade > 1:
        ramp = np.linspace(0, 1, fade, dtype=np.float32)
        sig[:fade] *= ramp
        sig[-fade:] *= ramp[::-1]
    return sig


def smooth_mask(m, passes=3):
    """Separable [1,2,1]/4 smoothing along frequency and time."""
    k = np.array([1, 2, 1], np.float32) / 4.0
    for _ in range(passes):
        for ax in (0, 1):
            p = np.pad(m, [(1, 1) if a == ax else (0, 0) for a in (0, 1)], mode="edge")
            sl = lambda o: tuple(slice(o, o + m.shape[a]) if a == ax else slice(None) for a in (0, 1))
            m = k[0] * p[sl(0)] + k[1] * p[sl(1)] + k[2] * p[sl(2)]
    return m.astype(np.float32)


N_FFT, HOP = 2048, 512


def apply_spec_gain(x, s0, s1, starts, gain):
    """Multiply the STFT of x[s0:s1] by gain (frames x bins), resynthesise."""
    seg = np.pad(x[s0:s1], (N_FFT, N_FFT)).astype(np.float32)
    win = np.hanning(N_FFT).astype(np.float32)
    idx = starts[:, None] + np.arange(N_FFT)
    spec = np.fft.rfft(seg[idx] * win, axis=1) * gain
    frames = (np.fft.irfft(spec, N_FFT, axis=1) * win).astype(np.float32)
    y = np.zeros(len(seg), np.float32)
    ws = np.zeros(len(seg), np.float32)
    w2 = win * win
    for k, s in enumerate(starts):
        y[s:s + N_FFT] += frames[k]
        ws[s:s + N_FFT] += w2
    y /= np.maximum(ws, 1e-3)
    out = x.copy()
    out[s0:s1] = y[N_FFT:N_FFT + (s1 - s0)]
    return out


def apply_watermark(data, sr, wm):
    out = data.copy()
    n = len(out)
    if wm["mode"] == "add":
        sig = synth_tones(sr, wm)
        s0 = int(wm["t0"] * sr)
        end = min(s0 + len(sig), n)
        if len(sig) and s0 < n:
            out[s0:end] += sig[:end - s0, None]
        return out
    s0 = max(int(wm["t0"] * sr) - N_FFT, 0)
    s1 = min(int(wm["t1"] * sr) + N_FFT, n)
    if s1 - s0 < N_FFT:
        return out
    nfr = 1 + (s1 - s0 + N_FFT) // HOP
    starts = np.arange(nfr) * HOP
    times = (s0 - N_FFT + starts + N_FFT / 2) / sr
    freqs = np.fft.rfftfreq(N_FFT, 1 / sr)
    m, _ = sample_mask(wm, freqs, times)                 # (bins, frames)
    m = smooth_mask(m)                                   # avoid clicks / splatter at edges
    sign = 1.0 if wm["mode"] == "boost" else -1.0
    gain = (10 ** (sign * wm["amount"] * m / 20.0)).T.astype(np.float32)
    for c in range(out.shape[1]):
        out[:, c] = apply_spec_gain(out[:, c], s0, s1, starts, gain)
    return out


def write_audio(path, data, sr):
    ext = os.path.splitext(path)[1].lower()
    if ext == ".mp3":
        with tempfile.TemporaryDirectory() as td:
            wav = os.path.join(td, "t.wav")
            sf.write(wav, data, sr, subtype="PCM_24")
            if shutil.which("ffmpeg"):
                subprocess.run(["ffmpeg", "-y", "-i", wav, "-codec:a", "libmp3lame",
                                "-b:a", "320k", path], check=True,
                               stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            else:
                sf.write(path, data, sr, format="MP3", compression_level=0.0)
    else:
        sf.write(path, data, sr, subtype="PCM_24")


# --------------------------------------------------------------------------- #
# Player
# --------------------------------------------------------------------------- #
class Player(QObject):
    """Audio player that can seek, change its end point and hot-swap the audio
    while playing (it restarts the sink from the current position)."""
    position = Signal(float)
    ended = Signal()
    state_changed = Signal(str)          # "playing" | "paused" | "stopped"

    def __init__(self):
        super().__init__()
        self.sink = self.qbuf = self.qbytes = None
        self.paused = False
        self.loop = False
        self.data = None
        self.sr = 44100
        self.base = 0.0
        self.end = 0.0
        self.region = (0.0, 0.0)
        self.timer = QTimer(self)
        self.timer.setInterval(40)
        self.timer.timeout.connect(self._tick)

    @property
    def active(self):
        return self.sink is not None

    def _release(self):
        self.timer.stop()
        if self.sink is not None:
            self.sink.stop()
            self.sink = None
        if self.qbuf is not None:
            self.qbuf.close()
            self.qbuf = None
        self.paused = False

    def _begin(self, from_t, paused=False):
        if not HAVE_AUDIO or self.data is None:
            return False
        self._release()
        sr = self.sr
        s0, s1 = max(int(from_t * sr), 0), min(int(self.end * sr), len(self.data))
        if s1 - s0 < 256:
            return False
        seg = self.data[s0:s1]
        if seg.shape[1] == 1:
            seg = np.repeat(seg, 2, axis=1)
        elif seg.shape[1] > 2:
            seg = seg[:, :2]
        dev = QMediaDevices.defaultAudioOutput()
        fmt = QAudioFormat()
        fmt.setSampleRate(sr)
        fmt.setChannelCount(2)
        fmt.setSampleFormat(QAudioFormat.SampleFormat.Int16)
        if not dev.isFormatSupported(fmt):
            sr2 = dev.preferredFormat().sampleRate() or 48000
            n2 = int(len(seg) * sr2 / sr)
            xs = np.linspace(0, len(seg) - 1, n2)
            seg = np.stack([np.interp(xs, np.arange(len(seg)), seg[:, c]) for c in range(2)], 1)
            fmt.setSampleRate(sr2)
        pcm = (np.clip(seg, -1, 1) * 32767).astype("<i2").tobytes()
        self.qbytes = QByteArray(pcm)
        self.qbuf = QBuffer(self.qbytes)
        self.qbuf.open(QIODevice.ReadOnly)
        self.sink = QAudioSink(dev, fmt)
        self.sink.start(self.qbuf)
        self.base = from_t
        self.timer.start()
        if paused:
            self.sink.suspend()
            self.paused = True
        return True

    def _restart(self, pos):
        if not self._begin(pos, paused=self.paused):
            self.stop()

    def play(self, data, sr, from_t, end_t, region):
        self.data, self.sr, self.end, self.region = data, sr, end_t, region
        ok = self._begin(from_t)
        self.state_changed.emit("playing" if ok else "stopped")
        return ok

    def current_pos(self):
        if self.sink is None:
            return self.base
        return self.base + self.sink.processedUSecs() / 1e6

    def seek(self, t, end_t=None, region=None):
        if end_t is not None:
            self.end = end_t
        if region is not None:
            self.region = region
        if self.active:
            self._restart(t)

    def set_bounds(self, end_t, region):
        pos = self.current_pos()
        self.end, self.region = end_t, region
        if self.active:
            self._restart(pos)

    def replace_audio(self, data):
        """Swap in edited audio and carry on from the same position."""
        self.data = data
        if self.active:
            self._restart(self.current_pos())

    def toggle_pause(self):
        if self.sink is None:
            return
        if self.paused:
            self.sink.resume()
        else:
            self.sink.suspend()
        self.paused = not self.paused
        self.state_changed.emit("paused" if self.paused else "playing")

    def stop(self):
        self._release()
        self.ended.emit()
        self.state_changed.emit("stopped")

    def _tick(self):
        if self.sink is None:
            return
        pos = self.current_pos()
        self.position.emit(min(pos, self.end))
        if self.paused:
            return
        done = pos >= self.end - 0.02 or (
            self.sink.state() == QAudio.State.IdleState and self.qbuf.atEnd())
        if done:
            if self.loop and self.region[1] - self.region[0] > 0.1:
                self.end = self.region[1]
                if not self._begin(self.region[0]):
                    self.stop()
            else:
                self.stop()


# --------------------------------------------------------------------------- #
# Axis helpers
# --------------------------------------------------------------------------- #
def nice_step(span, target):
    raw = span / max(target, 1)
    mag = 10 ** np.floor(np.log10(raw))
    for m in (1, 2, 5, 10):
        if raw <= m * mag:
            return m * mag
    return 10 * mag


def fmt_time(s, dec=None):
    s = max(s, 0.0)
    if dec is None:
        dec = 1 if abs(s - round(s)) > 1e-6 else 0
    m = int(s // 60)
    sec = s - 60 * m
    if round(sec, dec) >= 60:
        m, sec = m + 1, 0.0
    width = 2 if dec == 0 else 3 + dec
    return f"{m}:{sec:0{width}.{dec}f}"


def fmt_freq(f):
    return f"{f:g} Hz" if f < 1000 else f"{f / 1000:g} kHz"


def zoom_axis(lo, hi, anchor, factor, lim_lo, lim_hi, min_span):
    span = min(max((hi - lo) * factor, min_span), lim_hi - lim_lo)
    frac = (anchor - lo) / (hi - lo) if hi > lo else 0.5
    nlo = anchor - frac * span
    nhi = nlo + span
    if nlo < lim_lo:
        nlo, nhi = lim_lo, lim_lo + span
    if nhi > lim_hi:
        nlo, nhi = lim_hi - span, lim_hi
    return nlo, nhi


def pan_axis(lo, hi, delta, lim_lo, lim_hi):
    span = hi - lo
    nlo = min(max(lo + delta, lim_lo), lim_hi - span)
    return nlo, nlo + span


# --------------------------------------------------------------------------- #
# Spectrogram widget
# --------------------------------------------------------------------------- #
class SpectrogramView(QWidget):
    clicked = Signal(float, float)          # marker tool click: time, freq
    hovered = Signal(str)
    selectionChanged = Signal(object)       # (t0, t1, f0, f1) or None
    ML, MR, MT, MB = 70, 92, 16, 36

    def __init__(self):
        super().__init__()
        self.setMouseTracking(True)
        self.setMinimumSize(520, 320)
        self.idx = None
        self.db = None
        self.duration = 0.0
        self.nyq = 22050.0
        self.markers = []
        self.cutoff = None
        self.log = False
        self.t0 = self.t1 = 0.0
        self.f0, self.f1 = 0.0, 22050.0
        self.tool = "select"
        self.selection = None
        self.highlight = None
        self.playhead = None
        self.history = []
        self._cache_key = self._cache_img = None
        self._press = self._last = self._rubber = None
        self._dragging = self._panning = False
        self.set_tool("select")

    # ---- state
    def set_tool(self, tool):
        self.tool = tool
        self.setCursor({"pan": Qt.OpenHandCursor, "marker": Qt.PointingHandCursor}
                       .get(tool, Qt.CrossCursor))

    def set_data(self, db, duration, nyq, cutoff, reset=True):
        self.db, self.duration, self.nyq, self.cutoff = db, duration, nyq, cutoff
        self.idx = spectrum_to_index(db)
        self._cache_key = None
        if reset:
            self.history.clear()
            self.selection = self.highlight = self.playhead = None
            self.selectionChanged.emit(None)
            self.reset_view(push=False)
        else:
            self.update()

    def set_selection(self, sel):
        self.selection = sel
        self.selectionChanged.emit(sel)
        self.update()

    def set_playhead(self, t):
        self.playhead = t
        self.update()

    def fwd(self, f):
        return float(np.log(max(f, 1.0))) if self.log else float(f)

    def inv(self, v):
        return float(np.exp(v)) if self.log else float(v)

    def _flimits(self):
        return self.fwd(FMIN_LOG if self.log else 0.0), self.fwd(self.nyq)

    def _fmin_span(self):
        return float(np.log(1.05)) if self.log else 50.0

    def _state(self):
        return (self.t0, self.t1, self.f0, self.f1)

    def push_history(self):
        if not self.history or self.history[-1] != self._state():
            self.history.append(self._state())
            del self.history[:-50]

    def back(self):
        if self.idx is None:
            return
        if self.history:
            self.t0, self.t1, self.f0, self.f1 = self.history.pop()
            if self.log:
                self.f0 = max(self.f0, FMIN_LOG)
            self.update()
        else:
            self.reset_view(push=False)

    def reset_view(self, push=True):
        if push:
            self.push_history()
        self.t0, self.t1 = 0.0, self.duration
        self.f0 = FMIN_LOG if self.log else 0.0
        self.f1 = self.nyq
        self.update()

    def set_log(self, on):
        self.log = on
        if on:
            self.f0 = max(self.f0, FMIN_LOG)
        elif self.f0 <= FMIN_LOG:
            self.f0 = 0.0
        self.update()

    def zoom(self, factor, mode="both", anchor=None):
        if self.idx is None:
            return
        if anchor is None:
            anchor = ((self.t0 + self.t1) / 2,
                      self.inv((self.fwd(self.f0) + self.fwd(self.f1)) / 2))
        if mode in ("t", "both"):
            self.t0, self.t1 = zoom_axis(self.t0, self.t1, anchor[0], factor,
                                         0.0, self.duration, min(0.2, self.duration))
        if mode in ("f", "both"):
            lo, hi = zoom_axis(self.fwd(self.f0), self.fwd(self.f1), self.fwd(anchor[1]),
                               factor, *self._flimits(), self._fmin_span())
            self.f0, self.f1 = self.inv(lo), self.inv(hi)
        self.update()

    def zoom_to(self, t0, t1, f0, f1):
        if self.idx is None:
            return
        self.push_history()
        min_t = min(0.2, self.duration)
        t0, t1 = max(t0, 0.0), min(t1, self.duration)
        if t1 - t0 < min_t:
            c = (t0 + t1) / 2
            t0, t1 = c - min_t / 2, c + min_t / 2
        lo_lim, hi_lim = self._flimits()
        lo, hi = max(self.fwd(f0), lo_lim), min(self.fwd(f1), hi_lim)
        if hi - lo < self._fmin_span():
            c = (lo + hi) / 2
            lo, hi = c - self._fmin_span() / 2, c + self._fmin_span() / 2
        self.t0, self.t1 = t0, t1
        self.f0, self.f1 = self.inv(lo), self.inv(hi)
        self.update()

    # ---- mapping
    def plot_rect(self, w=None, h=None):
        w = self.width() if w is None else w
        h = self.height() if h is None else h
        return QRectF(self.ML, self.MT, w - self.ML - self.MR, h - self.MT - self.MB)

    def t_to_x(self, t, r):
        return r.left() + (t - self.t0) / (self.t1 - self.t0) * r.width()

    def f_to_y(self, f, r):
        a, b = self.fwd(self.f0), self.fwd(self.f1)
        return r.bottom() - (self.fwd(f) - a) / (b - a) * r.height()

    def pos_to_tf(self, pos, r):
        t = self.t0 + (pos.x() - r.left()) / r.width() * (self.t1 - self.t0)
        a, b = self.fwd(self.f0), self.fwd(self.f1)
        f = self.inv(a + (r.bottom() - pos.y()) / r.height() * (b - a))
        return t, f

    def to_tf(self, pos):
        r = self.plot_rect()
        if self.idx is None or not r.contains(pos):
            return None
        return self.pos_to_tf(pos, r)

    def rect_aspect(self, sel):
        """Displayed pixel aspect (w/h) of a (t0,t1,f0,f1) area in the current view."""
        r = self.plot_rect()
        t0, t1, f0, f1 = sel
        w = (t1 - t0) / (self.t1 - self.t0) * r.width()
        h = abs(self.fwd(f1) - self.fwd(f0)) / abs(self.fwd(self.f1) - self.fwd(self.f0)) * r.height()
        return float(np.clip(w / max(h, 1.0), 0.05, 60.0))

    # ---- rendering
    def _render_plot(self, pw, ph):
        key = (pw, ph, self.t0, self.t1, self.f0, self.f1, self.log)
        if key == self._cache_key:
            return self._cache_img
        nb, nc = self.idx.shape
        ys = 1 - (np.arange(ph) + 0.5) / ph
        if self.log:
            lo, hi = np.log(self.f0), np.log(self.f1)
            f = np.exp(lo + ys * (hi - lo))
        else:
            f = self.f0 + ys * (self.f1 - self.f0)
        rows = np.clip(np.rint(f / self.nyq * (nb - 1)).astype(int), 0, nb - 1)
        t = self.t0 + (np.arange(pw) + 0.5) / pw * (self.t1 - self.t0)
        cols = np.clip((t / self.duration * nc).astype(int), 0, nc - 1)
        rgb = np.ascontiguousarray(LUT[self.idx[rows[:, None], cols[None, :]]])
        img = QImage(rgb.data, pw, ph, pw * 3, QImage.Format_RGB888).copy()
        self._cache_key, self._cache_img = key, img
        return img

    def _freq_ticks(self):
        f0, f1 = self.f0, self.f1
        ticks = []
        if self.log:
            k = int(np.floor(np.log10(f0))) - 1
            while 10 ** k <= f1:
                for m in (1, 2, 5):
                    v = m * 10 ** k
                    if f0 <= v <= f1:
                        ticks.append(v)
                k += 1
        if len(ticks) < 3:
            step = nice_step(f1 - f0, 9)
            v = np.ceil(f0 / step) * step
            ticks = []
            while v <= f1 + 1e-9:
                ticks.append(round(float(v), 6))
                v += step
        return ticks

    def _tf_rect(self, rect, r):
        t0, t1, f0, f1 = rect
        return QRectF(QPointF(self.t_to_x(t0, r), self.f_to_y(f1, r)),
                      QPointF(self.t_to_x(t1, r), self.f_to_y(f0, r))).normalized()

    def paintEvent(self, _):
        p = QPainter(self)
        self.draw(p, self.width(), self.height(), overlays=True)

    def draw(self, p, w, h, overlays=True):
        p.setRenderHint(QPainter.Antialiasing, True)
        p.fillRect(0, 0, w, h, QColor(0, 0, 0))
        r = self.plot_rect(w, h)
        p.setFont(QFont("Helvetica", 9))
        if self.idx is None:
            p.setPen(QColor(150, 150, 150))
            p.drawText(QRectF(0, 0, w, h), Qt.AlignCenter, "Open or drop an audio file")
            return
        pw, ph = int(r.width()), int(r.height())
        if pw < 4 or ph < 4:
            return
        p.drawImage(QPointF(r.left(), r.top()), self._render_plot(pw, ph))
        white, grey = QColor(255, 255, 255), QColor(110, 110, 110)
        p.setPen(QPen(white, 1))
        p.setBrush(Qt.NoBrush)
        p.drawRect(r)

        for f in self._freq_ticks():
            y = self.f_to_y(f, r)
            p.setPen(white)
            p.drawLine(QPointF(r.left() - 4, y), QPointF(r.left(), y))
            p.drawText(QRectF(0, y - 8, r.left() - 8, 16),
                       Qt.AlignRight | Qt.AlignVCenter, fmt_freq(f))
            if self.log:
                p.setPen(QPen(QColor(255, 255, 255, 28), 1))
                p.drawLine(QPointF(r.left(), y), QPointF(r.right(), y))

        step = nice_step(self.t1 - self.t0, r.width() / 80)
        dec = 0 if step >= 1 else (1 if step >= 0.1 else 2)
        t = np.ceil(self.t0 / step) * step
        while t <= self.t1 + 1e-9:
            x = self.t_to_x(t, r)
            p.setPen(white)
            p.drawLine(QPointF(x, r.bottom()), QPointF(x, r.bottom() + 4))
            p.drawText(QRectF(x - 32, r.bottom() + 6, 64, 16), Qt.AlignCenter,
                       fmt_time(float(t), dec))
            t += step

        if self.cutoff and self.f0 <= self.cutoff <= self.f1:
            y = self.f_to_y(self.cutoff, r)
            p.setPen(QPen(QColor(255, 255, 255, 170), 1, Qt.DashLine))
            p.drawLine(QPointF(r.left(), y), QPointF(r.right(), y))
            p.setPen(white)
            p.drawText(QPointF(r.left() + 6, y - 4), f"cutoff ~{self.cutoff / 1000:.1f} kHz")

        # colour legend
        lg = QRectF(r.right() + 18, r.top(), 14, r.height())
        grad = QLinearGradient(lg.topLeft(), lg.bottomLeft())
        for pos, (cr, cg, cb) in RAINBOW_STOPS:
            grad.setColorAt(1 - pos, QColor(cr, cg, cb))
        p.fillRect(lg, grad)
        p.setPen(white)
        p.drawRect(lg)
        for db in range(0, int(DB_MIN) - 1, -20):
            y = lg.top() + (DB_MAX - db) / (DB_MAX - DB_MIN) * lg.height()
            p.drawLine(QPointF(lg.right(), y), QPointF(lg.right() + 4, y))
            p.drawText(QRectF(lg.right() + 6, y - 8, 40, 16),
                       Qt.AlignLeft | Qt.AlignVCenter, f"{db}")
        p.setPen(grey)
        p.drawText(QRectF(lg.left() - 4, lg.bottom() + 4, 60, 14), Qt.AlignLeft, "dBFS")

        # markers
        fm = QFontMetrics(p.font())
        for m in self.markers:
            if not (self.t0 <= m["t"] <= self.t1 and self.f0 <= m["f"] <= self.f1):
                continue
            x, y = self.t_to_x(m["t"], r), self.f_to_y(m["f"], r)
            p.setPen(QPen(white, 1))
            p.setBrush(QColor(0, 0, 0))
            p.drawEllipse(QPointF(x, y), 4, 4)
            tw = fm.horizontalAdvance(m["text"]) + 12
            box = QRectF(min(x + 8, r.right() - tw), y - 11, tw, 22)
            p.setBrush(QColor(0, 0, 0, 215))
            p.drawRect(box)
            p.drawText(box, Qt.AlignCenter, m["text"])
        p.setBrush(Qt.NoBrush)

        if not overlays:
            return
        p.save()
        p.setClipRect(r)
        if self.highlight:
            p.setPen(QPen(white, 1, Qt.DotLine))
            p.drawRect(self._tf_rect(self.highlight, r))
        if self.selection:
            sr = self._tf_rect(self.selection, r)
            p.setBrush(QColor(255, 255, 255, 40))
            p.setPen(QPen(white, 1.5, Qt.DashLine))
            p.drawRect(sr)
        if self._rubber is not None:
            p.setBrush(QColor(255, 255, 255, 45))
            p.setPen(QPen(white, 1, Qt.DashLine))
            p.drawRect(self._rubber)
        if self.playhead is not None and self.t0 <= self.playhead <= self.t1:
            x = self.t_to_x(self.playhead, r)
            p.setPen(QPen(white, 2))
            p.drawLine(QPointF(x, r.top()), QPointF(x, r.bottom()))
        p.restore()

    # ---- interaction
    def wheelEvent(self, e):
        if self.idx is None:
            return
        d = e.angleDelta().y() or e.angleDelta().x()
        if d == 0:
            return
        mods = e.modifiers()
        mode = "f" if mods & Qt.ShiftModifier else ("both" if mods & Qt.ControlModifier else "t")
        self.zoom(0.8 ** (d / 120.0), mode, self.to_tf(e.position()))
        e.accept()

    @staticmethod
    def _clamp(pt, r):
        return QPointF(min(max(pt.x(), r.left()), r.right()),
                       min(max(pt.y(), r.top()), r.bottom()))

    def mousePressEvent(self, e):
        if self.idx is None:
            return
        r = self.plot_rect()
        pos = e.position()
        if e.button() == Qt.RightButton:
            self.back()
        elif e.button() == Qt.MiddleButton or (e.button() == Qt.LeftButton and self.tool == "pan"):
            if r.contains(pos):
                self._panning = True
                self._last = pos
                self.setCursor(Qt.ClosedHandCursor)
        elif e.button() == Qt.LeftButton and r.contains(pos):
            self._press = pos
            self._dragging = False

    def mouseMoveEvent(self, e):
        pos = e.position()
        r = self.plot_rect()
        if self._panning:
            dx, dy = pos.x() - self._last.x(), pos.y() - self._last.y()
            dt = -dx / r.width() * (self.t1 - self.t0)
            self.t0, self.t1 = pan_axis(self.t0, self.t1, dt, 0.0, self.duration)
            a, b = self.fwd(self.f0), self.fwd(self.f1)
            lo, hi = pan_axis(a, b, dy / r.height() * (b - a), *self._flimits())
            self.f0, self.f1 = self.inv(lo), self.inv(hi)
            self._last = pos
            self.update()
            return
        if self._press is not None and (e.buttons() & Qt.LeftButton) \
                and self.tool in ("select", "zoom"):
            if (pos - self._press).manhattanLength() > 5:
                self._dragging = True
            if self._dragging:
                self._rubber = QRectF(self._clamp(self._press, r), self._clamp(pos, r)).normalized()
                self.update()
            return
        tf = self.to_tf(pos)
        if tf and self.db is not None:
            t, f = tf
            nb, nc = self.db.shape
            col = min(max(int(t / self.duration * nc), 0), nc - 1)
            row = min(max(int(round(f / self.nyq * (nb - 1))), 0), nb - 1)
            self.hovered.emit(f"{fmt_time(t, 2)}   {f:,.0f} Hz   {self.db[row, col]:.1f} dB")

    def mouseReleaseEvent(self, e):
        if self._panning and e.button() in (Qt.MiddleButton, Qt.LeftButton):
            self._panning = False
            self.set_tool(self.tool)
            return
        if e.button() != Qt.LeftButton or self._press is None:
            return
        press, rub, was = self._press, self._rubber, self._dragging
        self._press = self._rubber = None
        self._dragging = False
        r = self.plot_rect()
        if was and rub is not None and rub.width() > 6 and rub.height() > 6:
            ta, fa = self.pos_to_tf(QPointF(rub.left(), rub.bottom()), r)
            tb, fb = self.pos_to_tf(QPointF(rub.right(), rub.top()), r)
            if self.tool == "select":
                self.set_selection((max(ta, 0.0), min(tb, self.duration),
                                    max(fa, 0.0), min(fb, self.nyq)))
            elif self.tool == "zoom":
                self.zoom_to(ta, tb, fa, fb)
        elif not was:
            tf = self.pos_to_tf(press, r)
            if self.tool == "marker":
                self.clicked.emit(*tf)
            elif self.tool == "zoom":
                self.push_history()
                self.zoom(0.5, "both", tf)
            elif self.tool == "select" and self.selection is not None:
                self.set_selection(None)
        self.update()

    def export_png(self, path, w=1600, h=800):
        img = QImage(w, h, QImage.Format_RGB32)
        p = QPainter(img)
        self.draw(p, w, h, overlays=False)
        p.end()
        img.save(path)


# --------------------------------------------------------------------------- #
# Timeline (progress + play-range selector) and background worker
# --------------------------------------------------------------------------- #
class TimelineBar(QWidget):
    seekRequested = Signal(float)
    rangeChanged = Signal(object)        # live while dragging: (a, b) or None
    rangeCommitted = Signal(object)      # on mouse release
    PAD = 14

    def __init__(self):
        super().__init__()
        self.setFixedHeight(56)
        self.setMouseTracking(True)
        self.duration = 0.0
        self.range = None
        self.pos = None                  # playhead while playing
        self.cursor_t = 0.0              # parked start position
        self.view = None                 # visible time window of the spectrogram
        self._drag = None
        self._press_x = 0.0
        self._moved = False
        self.setToolTip("Drag on the bar to choose the part to play · drag the white "
                        "handles to adjust · click to jump · double-click to clear")

    def track(self):
        return QRectF(self.PAD, 10, self.width() - 2 * self.PAD, 20)

    def t_to_x(self, t):
        r = self.track()
        return r.left() + t / max(self.duration, 1e-9) * r.width()

    def x_to_t(self, x):
        r = self.track()
        return min(max((x - r.left()) / r.width(), 0.0), 1.0) * self.duration

    def paintEvent(self, _):
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing, True)
        p.fillRect(self.rect(), QColor(0, 0, 0))
        tr = self.track()
        white = QColor(255, 255, 255)
        if self.duration <= 0:
            p.setPen(QColor(90, 90, 90))
            p.drawRect(tr)
            return
        p.fillRect(tr, QColor(26, 26, 26))
        p.setFont(QFont("Helvetica", 8))
        step = nice_step(self.duration, tr.width() / 90)
        dec = 0 if step >= 1 else 1
        t = 0.0
        while t <= self.duration + 1e-9:
            x = self.t_to_x(t)
            p.setPen(QColor(120, 120, 120))
            p.drawLine(QPointF(x, tr.bottom()), QPointF(x, tr.bottom() + 4))
            p.setPen(QColor(170, 170, 170))
            p.drawText(QRectF(x - 30, tr.bottom() + 6, 60, 12), Qt.AlignCenter,
                       fmt_time(float(t), dec))
            t += step
        start = self.range[0] if self.range else 0.0
        if self.range:
            a, b = self.t_to_x(self.range[0]), self.t_to_x(self.range[1])
            p.fillRect(QRectF(a, tr.top(), b - a, tr.height()), QColor(255, 255, 255, 60))
        if self.pos is not None:                       # progress fill
            x0, x1 = self.t_to_x(start), self.t_to_x(max(self.pos, start))
            p.fillRect(QRectF(x0, tr.top(), x1 - x0, tr.height()), QColor(255, 255, 255, 150))
        if self.range:
            p.setPen(Qt.NoPen)
            p.setBrush(white)
            for hx in (self.t_to_x(self.range[0]), self.t_to_x(self.range[1])):
                p.drawRect(QRectF(hx - 3, tr.top() - 5, 6, tr.height() + 10))
            p.setBrush(Qt.NoBrush)
        if self.view and self.view[1] - self.view[0] < self.duration - 0.01:
            x0, x1 = self.t_to_x(self.view[0]), self.t_to_x(self.view[1])
            p.fillRect(QRectF(x0, 2, x1 - x0, 3), QColor(255, 255, 255, 170))
        if self.pos is None:                           # parked cursor
            x = self.t_to_x(self.cursor_t)
            p.setPen(QPen(QColor(255, 255, 255, 190), 1, Qt.DashLine))
            p.drawLine(QPointF(x, tr.top() - 3), QPointF(x, tr.bottom() + 3))
        else:                                          # playhead
            x = self.t_to_x(self.pos)
            p.setPen(QPen(white, 2))
            p.drawLine(QPointF(x, tr.top() - 4), QPointF(x, tr.bottom() + 4))
        p.setPen(QPen(white, 1))
        p.setBrush(Qt.NoBrush)
        p.drawRect(tr)

    def _near_handle(self, x):
        if not self.range:
            return None
        for name, t in (("a", self.range[0]), ("b", self.range[1])):
            if abs(x - self.t_to_x(t)) <= 8:
                return name
        return None

    def mousePressEvent(self, e):
        if self.duration <= 0 or e.button() != Qt.LeftButton:
            return
        x = e.position().x()
        h = self._near_handle(x)
        self._moved = False
        if h:
            self._drag = ("handle", h)
        else:
            self._drag = ("new", x)
            self._press_x = x

    def mouseMoveEvent(self, e):
        x = e.position().x()
        if self._drag is None:
            self.setCursor(Qt.SizeHorCursor if self._near_handle(x) else Qt.PointingHandCursor)
            return
        t = self.x_to_t(x)
        if self._drag[0] == "handle":
            a, b = self.range
            if self._drag[1] == "a":
                a = min(t, b - 0.05)
            else:
                b = max(t, a + 0.05)
            self.range = (a, b)
            self._moved = True
            self.rangeChanged.emit(self.range)
        elif abs(x - self._press_x) > 4:
            self._moved = True
            a, b = sorted((self.x_to_t(self._press_x), t))
            self.range = (a, b)
            self.rangeChanged.emit(self.range)
        self.update()

    def mouseReleaseEvent(self, e):
        if self._drag is None:
            return
        kind, _ = self._drag
        self._drag = None
        if self._moved:
            if self.range and self.range[1] - self.range[0] < 0.05:
                self.range = None
                self.rangeChanged.emit(None)
            self.rangeCommitted.emit(self.range)
        elif kind == "new":
            self.seekRequested.emit(self.x_to_t(self._press_x))
        self.update()

    def mouseDoubleClickEvent(self, e):
        self.range = None
        self.rangeChanged.emit(None)
        self.rangeCommitted.emit(None)
        self.update()


class RebuildWorker(QThread):
    """Applies watermarks and recomputes the spectrogram off the GUI thread, so
    playback and the UI keep running while you edit."""
    done = Signal(object)

    def __init__(self, orig, base, sr, wms, req, channel, n_fft):
        super().__init__()
        self.orig, self.base, self.sr = orig, base, sr
        self.wms, self.req, self.channel, self.n_fft = wms, req, channel, n_fft

    def run(self):
        res = {"req": self.req}
        try:
            msg = ""
            if self.req["rebuild"]:
                d = self.orig
                for w in self.wms:
                    d = apply_watermark(d, self.sr, w)
                if self.wms:
                    peak = float(np.abs(d).max())
                    if peak > 0.98:                       # avoid clipping distortion
                        d = d * (0.98 / peak)
                        msg = f"Level lowered by {20 * np.log10(peak / 0.98):.1f} dB to avoid clipping."
            else:
                d = self.base
            c = self.channel
            x = d.mean(axis=1) if c <= 0 else d[:, min(c, d.shape[1]) - 1]
            db = compute_spectrogram(x, self.sr, self.n_fft)
            res.update(data=d, db=db, cutoff=estimate_cutoff(db, self.sr),
                       dur=len(x) / self.sr, msg=msg)
        except Exception as e:
            res["error"] = str(e)
        self.done.emit(res)


# --------------------------------------------------------------------------- #
# Main window
# --------------------------------------------------------------------------- #
STYLE = """
* { font-family: Helvetica, Arial, sans-serif; font-size: 12px; }
QMainWindow, QDialog, QWidget#page, QScrollArea, QScrollArea > QWidget > QWidget { background:#000; color:#fff; }
QLabel { color:#fff; background:transparent; }
QLabel#h { font-weight:bold; letter-spacing:2px; padding-top:10px; border-top:1px solid #444; }
QLabel#dim { color:#9a9a9a; }
QPushButton { background:#000; color:#fff; border:1px solid #fff; padding:6px 10px; }
QPushButton:hover, QPushButton:checked { background:#fff; color:#000; }
QPushButton:disabled { color:#555; border-color:#444; }
QLineEdit, QComboBox, QDoubleSpinBox, QListWidget { background:#000; color:#fff;
    border:1px solid #666; padding:3px; selection-background-color:#fff; selection-color:#000; }
QComboBox QAbstractItemView { background:#000; color:#fff; selection-background-color:#fff;
    selection-color:#000; }
QCheckBox { color:#fff; }
QTabWidget::pane { border:1px solid #444; top:-1px; }
QTabBar::tab { background:#000; color:#9a9a9a; padding:7px 10px; border:1px solid #444; }
QTabBar::tab:selected { background:#fff; color:#000; }
QScrollBar:vertical { background:#000; width:8px; }
QScrollBar::handle:vertical { background:#666; min-height:20px; }
QScrollBar::add-line, QScrollBar::sub-line { height:0; width:0; }
QToolTip { background:#000; color:#fff; border:1px solid #fff; }
"""


def scrolled(widget):
    s = QScrollArea()
    s.setWidget(widget)
    s.setWidgetResizable(True)
    s.setFrameShape(QScrollArea.NoFrame)
    return s


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("SPECTRA")
        self.resize(1360, 800)
        self.setAcceptDrops(True)
        self.path = None
        self.orig = None          # decoded original audio
        self.data = None          # working audio (original + watermarks)
        self.sr = 44100
        self.wms = []
        self.player = Player()
        self.range = None         # chosen play range (a, b) or None = whole file
        self.cursor = 0.0         # parked start position
        self.worker = None
        self._pending = None
        self._gen = 0
        self._sync = False

        self.view = SpectrogramView()
        self.view.clicked.connect(self.on_marker_click)
        self.view.hovered.connect(lambda s: self.status.setText(s))
        self.view.selectionChanged.connect(self.on_selection)
        self.status = QLabel("")
        self.status.setObjectName("dim")

        # ---------------- toolbar
        tb = QHBoxLayout()
        self.tool_group = QButtonGroup(self)
        self.tool_group.setExclusive(True)
        for key, label, tip in (
                ("select", "Select", "Drag an area (for playing / watermark). Click to clear."),
                ("zoom", "Zoom box", "Drag an area to zoom into it. Click = zoom in x2."),
                ("pan", "Pan", "Drag to move the view."),
                ("marker", "Marker", "Click to drop a text marker.")):
            b = QPushButton(label)
            b.setCheckable(True)
            b.setToolTip(tip)
            b.setChecked(key == "select")
            b.clicked.connect(lambda _=False, k=key: self.view.set_tool(k))
            self.tool_group.addButton(b)
            tb.addWidget(b)
        tb.addSpacing(12)
        b_zsel = QPushButton("Zoom to selection")
        b_zsel.clicked.connect(self.zoom_to_selection)
        b_in, b_out = QPushButton("+"), QPushButton("−")
        b_in.setFixedWidth(34)
        b_out.setFixedWidth(34)
        b_in.clicked.connect(lambda: (self.view.push_history(), self.view.zoom(0.6)))
        b_out.clicked.connect(lambda: (self.view.push_history(), self.view.zoom(1 / 0.6)))
        b_back, b_reset = QPushButton("Back"), QPushButton("Reset")
        b_back.setToolTip("Undo the last zoom (also: right-click)")
        b_back.clicked.connect(self.view.back)
        b_reset.clicked.connect(self.view.reset_view)
        self.b_log = QPushButton("Frequency: Linear")
        self.b_log.setCheckable(True)
        self.b_log.toggled.connect(self.toggle_log)
        for w in (b_zsel, b_in, b_out, b_back, b_reset):
            tb.addWidget(w)
        tb.addSpacing(12)
        tb.addWidget(self.b_log)
        tb.addStretch(1)
        hint = QLabel("Wheel: zoom time · Shift+wheel: zoom freq · Ctrl+wheel: both · "
                      "Right-click: zoom back · Middle-drag: pan · Space: play/pause")
        hint.setObjectName("dim")

        # ---------------- tab 1: info + player
        t1 = QWidget()
        t1.setObjectName("page")
        l1 = QVBoxLayout(t1)
        l1.setSpacing(6)
        btn_open = QPushButton("Open audio file…")
        btn_open.clicked.connect(self.open_dialog)
        l1.addWidget(btn_open)
        l1.addWidget(self._h("INFO"))
        self.info = QLabel("No file loaded")
        self.info.setWordWrap(True)
        self.info.setTextFormat(Qt.RichText)
        l1.addWidget(self.info)
        l1.addWidget(self._h("ORIGINALITY CHECK"))
        self.verdict = QLabel("")
        self.verdict.setWordWrap(True)
        l1.addWidget(self.verdict)
        form = QFormLayout()
        self.cb_channel = QComboBox()
        self.cb_channel.currentIndexChanged.connect(lambda *_: self.request_update())
        self.cb_fft = QComboBox()
        self.cb_fft.addItems(["1024", "2048", "4096", "8192"])
        self.cb_fft.setCurrentText("2048")
        self.cb_fft.currentIndexChanged.connect(lambda *_: self.request_update())
        form.addRow("Channel", self.cb_channel)
        form.addRow("FFT size", self.cb_fft)
        l1.addLayout(form)

        l1.addStretch(1)

        # ---------------- tab 2: watermark
        t2 = QWidget()
        t2.setObjectName("page")
        l2 = QVBoxLayout(t2)
        l2.setSpacing(6)
        how = QLabel("1. Select tool: drag an area on the spectrogram.\n"
                     "2. Type text (or fill the whole area), choose how it blends.\n"
                     "3. Add watermark. Select it in the list to edit, resize or remove it.")
        how.setWordWrap(True)
        how.setObjectName("dim")
        l2.addWidget(how)
        wf = QFormLayout()
        self.w_shape = QComboBox()
        self.w_shape.addItem("Text / letters", "text")
        self.w_shape.addItem("Solid area", "solid")
        self.w_text = QLineEdit("© YOUR NAME")
        self.w_bold = QCheckBox("Bold")
        self.w_bold.setChecked(True)
        self.w_size = self._spin(20, 100, 80, " %", 0)
        self.w_mode = QComboBox()
        self.w_mode.addItem("Add tones (increment)", "add")
        self.w_mode.addItem("Boost existing audio (+dB)", "boost")
        self.w_mode.addItem("Cut existing audio (−dB)", "cut")
        self.w_mode.currentIndexChanged.connect(self.on_mode_changed)
        self.w_amount = self._spin(-90, -6, -35, " dBFS", 0)
        self.w_invert = QCheckBox("Invert (affect everything except the letters)")
        wf.addRow("Shape", self.w_shape)
        wf.addRow("Text", self.w_text)
        wf.addRow("", self.w_bold)
        wf.addRow("Text size", self.w_size)
        wf.addRow("Mode", self.w_mode)
        wf.addRow("Amount", self.w_amount)
        l2.addLayout(wf)
        l2.addWidget(self.w_invert)
        mode_note = QLabel("Add = draws new tones (works even in silent areas, e.g. above the "
                           "lowpass cutoff). Boost/Cut = reshapes the music's own spectrum, "
                           "so the letters follow it (needs audio in that area).")
        mode_note.setWordWrap(True)
        mode_note.setObjectName("dim")
        l2.addWidget(mode_note)
        b_add = QPushButton("Add watermark from selection")
        b_add.clicked.connect(self.add_wm)
        l2.addWidget(b_add)
        self.wm_list = QListWidget()
        self.wm_list.setFixedHeight(120)
        self.wm_list.currentRowChanged.connect(self.on_wm_selected)
        l2.addWidget(self.wm_list)
        r = QHBoxLayout()
        b_upd, b_mv = QPushButton("Apply settings"), QPushButton("Move to selection")
        b_upd.setToolTip("Apply the settings above to the selected watermark")
        b_mv.setToolTip("Move / resize the selected watermark to the current selection")
        b_upd.clicked.connect(self.update_wm)
        b_mv.clicked.connect(self.move_wm)
        r.addWidget(b_upd)
        r.addWidget(b_mv)
        l2.addLayout(r)
        r = QHBoxLayout()
        b_big, b_small, b_del = QPushButton("Bigger"), QPushButton("Smaller"), QPushButton("Delete")
        b_big.clicked.connect(lambda: self.scale_wm(1.2))
        b_small.clicked.connect(lambda: self.scale_wm(1 / 1.2))
        b_del.clicked.connect(self.delete_wm)
        for b in (b_big, b_small, b_del):
            r.addWidget(b)
        l2.addLayout(r)
        l2.addWidget(self._h("EXPORT"))
        warn = QLabel("Low-bitrate MP3 encoders remove content above ~16 kHz. For high-frequency "
                      "watermarks export at 320k MP3, FLAC or WAV.")
        warn.setWordWrap(True)
        warn.setObjectName("dim")
        l2.addWidget(warn)
        b_exp = QPushButton("Export audio…")
        b_exp.clicked.connect(self.export_audio)
        b_png = QPushButton("Export spectrogram PNG")
        b_png.clicked.connect(self.export_png)
        l2.addWidget(b_exp)
        l2.addWidget(b_png)
        l2.addStretch(1)

        # ---------------- tab 3: tags + markers
        t3 = QWidget()
        t3.setObjectName("page")
        l3 = QVBoxLayout(t3)
        l3.setSpacing(6)
        l3.addWidget(self._h("ID3 TAGS"))
        tf = QFormLayout()
        self.t_title, self.t_artist, self.t_album = QLineEdit(), QLineEdit(), QLineEdit()
        self.t_copy, self.t_comm = QLineEdit(), QLineEdit()
        for lbl, w in (("Title", self.t_title), ("Artist", self.t_artist),
                       ("Album", self.t_album), ("Copyright", self.t_copy),
                       ("Comment", self.t_comm)):
            tf.addRow(lbl, w)
        l3.addLayout(tf)
        b4 = QPushButton("Save tags to MP3")
        b4.clicked.connect(self.save_tags)
        l3.addWidget(b4)
        l3.addWidget(self._h("TEXT MARKERS"))
        mh = QLabel("Choose the Marker tool and click the spectrogram. Markers can be written "
                    "into the MP3 as ID3 chapters (audio untouched).")
        mh.setWordWrap(True)
        mh.setObjectName("dim")
        l3.addWidget(mh)
        self.mlist = QListWidget()
        self.mlist.setFixedHeight(120)
        self.mlist.itemDoubleClicked.connect(self.edit_marker)
        l3.addWidget(self.mlist)
        r = QHBoxLayout()
        bd = QPushButton("Delete marker")
        bd.clicked.connect(self.del_marker)
        bw = QPushButton("Write into MP3")
        bw.clicked.connect(self.write_markers)
        r.addWidget(bd)
        r.addWidget(bw)
        l3.addLayout(r)
        l3.addStretch(1)

        self.tabs = QTabWidget()
        self.tabs.setDocumentMode(True)
        self.tabs.addTab(scrolled(t1), "Info && Play")
        self.tabs.addTab(scrolled(t2), "Watermark")
        self.tabs.addTab(scrolled(t3), "Tags")
        self.tabs.setFixedWidth(350)

        # ---------------- transport (under the graph)
        self.timeline = TimelineBar()
        self.b_play = QPushButton("▶ Play")
        self.b_play.setFixedWidth(120)
        self.b_stop = QPushButton("■ Stop")
        self.b_stop.setFixedWidth(90)
        self.chk_loop = QCheckBox("Loop")
        self.lbl_time = QLabel("0:00.0 / 0:00.0")
        self.lbl_time.setMinimumWidth(130)
        self.lbl_range = QLabel("Play range: whole file")
        self.lbl_range.setObjectName("dim")
        b_clear = QPushButton("Clear range")
        trow = QHBoxLayout()
        for w_ in (self.b_play, self.b_stop, self.chk_loop, self.lbl_time):
            trow.addWidget(w_)
        trow.addStretch(1)
        trow.addWidget(self.lbl_range)
        trow.addWidget(b_clear)
        self.b_play.clicked.connect(self.on_play)
        self.b_stop.clicked.connect(self.player.stop)
        self.chk_loop.toggled.connect(lambda on: setattr(self.player, "loop", on))
        b_clear.clicked.connect(lambda: self.set_range(None, commit=True))
        self.timeline.seekRequested.connect(self.on_seek)
        self.timeline.rangeChanged.connect(lambda r: self.set_range(r))
        self.timeline.rangeCommitted.connect(self.on_timeline_committed)
        self.player.position.connect(self.on_player_position)
        self.player.ended.connect(self.on_player_ended)
        self.player.state_changed.connect(self.on_player_state)
        if not HAVE_AUDIO:
            for w_ in (self.b_play, self.b_stop):
                w_.setEnabled(False)
                w_.setToolTip("Audio playback unavailable (Qt Multimedia not found).")
        self._sync_timer = QTimer(self)
        self._sync_timer.setInterval(120)
        self._sync_timer.timeout.connect(self._sync_timeline)
        self._sync_timer.start()

        right = QVBoxLayout()
        right.setContentsMargins(0, 0, 0, 4)
        right.addLayout(tb)
        right.addWidget(hint)
        right.addWidget(self.view, 1)
        right.addWidget(self.timeline)
        right.addLayout(trow)
        right.addWidget(self.status)
        rw = QWidget()
        rw.setLayout(right)
        central = QWidget()
        h = QHBoxLayout(central)
        h.setContentsMargins(8, 8, 8, 8)
        h.addWidget(self.tabs)
        h.addWidget(rw, 1)
        self.setCentralWidget(central)

        QShortcut(QKeySequence(Qt.Key_Space), self, activated=self.on_play)
        QShortcut(QKeySequence(Qt.Key_Escape), self, activated=lambda: self.view.set_selection(None))

    # ---------------- helpers
    def _h(self, text):
        l = QLabel(text)
        l.setObjectName("h")
        return l

    def _spin(self, lo, hi, val, suffix, dec):
        s = QDoubleSpinBox()
        s.setRange(lo, hi)
        s.setValue(val)
        s.setSuffix(suffix)
        s.setDecimals(dec)
        return s

    def warn(self, msg):
        QMessageBox.warning(self, "SPECTRA", msg)

    def is_mp3(self):
        return bool(self.path) and self.path.lower().endswith(".mp3")

    def toggle_log(self, on):
        self.b_log.setText("Frequency: Log" if on else "Frequency: Linear")
        self.view.set_log(on)

    def dragEnterEvent(self, e):
        if e.mimeData().hasUrls():
            e.acceptProposedAction()

    def dropEvent(self, e):
        for u in e.mimeData().urls():
            if u.toLocalFile().lower().endswith(AUDIO_EXTS):
                self.load(u.toLocalFile())
                break

    # ---------------- loading / analysis
    def open_dialog(self):
        p, _ = QFileDialog.getOpenFileName(
            self, "Open audio", "", "Audio (*.mp3 *.wav *.flac *.ogg *.m4a *.aac *.opus)")
        if p:
            self.load(p)

    def load(self, path):
        QApplication.setOverrideCursor(Qt.WaitCursor)
        try:
            data, sr = decode_audio(path)
        except Exception as e:
            QApplication.restoreOverrideCursor()
            self.warn(str(e))
            return
        QApplication.restoreOverrideCursor()
        self.player.stop()
        self.path, self.orig, self.data, self.sr = path, data, data, sr
        self._gen += 1
        self._pending = None
        self.cursor = 0.0
        self.timeline.duration = len(data) / sr
        self.timeline.cursor_t = 0.0
        self.timeline.pos = None
        self.set_range(None)
        self.on_player_position(None)
        self.wms = []
        self.refresh_wm_list()
        self.setWindowTitle(f"SPECTRA - {os.path.basename(path)}")
        self.cb_channel.blockSignals(True)
        self.cb_channel.clear()
        self.cb_channel.addItem("Mix")
        for i in range(data.shape[1]):
            self.cb_channel.addItem(f"Ch {i + 1}")
        self.cb_channel.blockSignals(False)
        rows = build_info(path, data, sr)
        self.info.setText("<table cellspacing='3'>" + "".join(
            f"<tr><td style='color:#9a9a9a' valign='top'>{_html.escape(k)}&nbsp;&nbsp;</td>"
            f"<td>{_html.escape(v)}</td></tr>" for k, v in rows) + "</table>")
        self.view.markers = []
        self.load_tags()
        self.request_update(reset=True, rebuild=False)

    def request_update(self, reset=False, rebuild=False):
        """Recompute audio (watermarks) and/or spectrogram in a worker thread."""
        if self.orig is None:
            return
        old = self._pending
        if old:
            reset, rebuild = reset or old["reset"], rebuild or old["rebuild"]
        self._pending = dict(reset=reset, rebuild=rebuild, gen=self._gen)
        if self.worker is None or not self.worker.isRunning():
            self._start_worker()

    def _start_worker(self):
        req, self._pending = self._pending, None
        if req is None:
            return
        for w in self.wms:
            prepare_wm(w)                       # font work stays on the GUI thread
        self.status.setText("Processing… (playback continues)")
        wk = RebuildWorker(self.orig, self.data, self.sr, [dict(w) for w in self.wms], req,
                           self.cb_channel.currentIndex(), int(self.cb_fft.currentText()))
        wk.done.connect(self._on_update)
        wk.finished.connect(self._after_worker)
        self.worker = wk
        wk.start()

    def _after_worker(self):
        if self._pending:
            self._start_worker()

    def _on_update(self, res):
        req = res["req"]
        if req["gen"] != self._gen:             # a different file was opened meanwhile
            return
        if "error" in res:
            return self.warn(f"Processing failed: {res['error']}")
        changed = res["data"] is not self.data
        self.data = res["data"]
        nyq = self.sr / 2
        self.view.set_data(res["db"], res["dur"], nyq, res["cutoff"], reset=req["reset"])
        self.timeline.duration = res["dur"]
        if changed and not req["reset"]:
            self.player.replace_audio(self.data)   # keeps playing from the same spot
        note = "\n(Includes your watermarks.)" if self.wms else ""
        self.verdict.setText(verdict_for(res["cutoff"], nyq) + note +
                             "\nHeuristic only - judge by ear and other tools too.")
        self.status.setText(res["msg"] or "Ready.")
        self.refresh_markers()
        self.timeline.update()

    # ---------------- selection / zoom / transport
    def duration(self):
        return len(self.data) / self.sr if self.data is not None else 0.0

    def region(self):
        return self.range if self.range else (0.0, self.duration())

    def update_range_label(self):
        if self.range:
            a, b = self.range
            txt = f"Play range {fmt_time(a, 1)} – {fmt_time(b, 1)} ({b - a:.1f} s)"
        else:
            txt = "Play range: whole file"
        sel = self.view.selection
        if sel:
            txt += f" · area {fmt_freq(round(sel[2]))} – {fmt_freq(round(sel[3]))}"
        self.lbl_range.setText(txt)

    def set_range(self, rng, commit=False):
        if rng is not None:
            a, b = max(0.0, rng[0]), min(self.duration() or rng[1], rng[1])
            rng = (a, b) if b - a >= 0.05 else None
        self.range = rng
        self.timeline.range = rng
        self.timeline.update()
        self.update_range_label()
        if commit:
            self.apply_range_to_player()

    def apply_range_to_player(self):
        if not self.player.active:
            return
        r0, r1 = self.region()
        pos = self.player.current_pos()
        if r0 <= pos < r1 - 0.05:
            self.player.set_bounds(r1, (r0, r1))        # keep playing, new end point
        else:
            self.player.seek(r0, r1, (r0, r1))          # jump to the newly chosen part

    def on_timeline_committed(self, rng):
        self.set_range(rng, commit=True)
        sel = self.view.selection
        if rng and sel:                                  # keep the area's time span in step
            self._sync = True
            self.view.set_selection((rng[0], rng[1], sel[2], sel[3]))
            self._sync = False
            self.update_range_label()

    def on_selection(self, sel):
        if sel and not self._sync:
            self.set_range((sel[0], sel[1]), commit=True)
        self.update_range_label()

    def zoom_to_selection(self):
        if self.view.selection:
            self.view.zoom_to(*self.view.selection)
        else:
            self.warn("Make a selection first (Select tool, then drag).")

    def on_play(self):
        if self.data is None:
            return
        if not HAVE_AUDIO:
            return self.warn("Audio playback is unavailable (Qt Multimedia not found).")
        if self.player.active:
            self.player.toggle_pause()
            return
        r0, r1 = self.region()
        start = self.cursor if r0 <= self.cursor < r1 - 0.05 else r0
        self.player.loop = self.chk_loop.isChecked()
        if not self.player.play(self.data, self.sr, start, r1, (r0, r1)):
            self.warn("Could not start playback (no audio device, or range too short).")

    def on_seek(self, t):
        self.cursor = t
        self.timeline.cursor_t = t
        self.timeline.update()
        if self.player.active:
            r0, r1 = self.region()
            end = r1 if r0 <= t < r1 - 0.05 else self.duration()
            self.player.seek(t, end, (r0, r1))
        else:
            self.on_player_position(None)

    def on_player_position(self, t):
        self.view.set_playhead(t)
        self.timeline.pos = t
        self.timeline.update()
        shown = self.cursor if t is None else t
        self.lbl_time.setText(f"{fmt_time(shown, 1)} / {fmt_time(self.duration(), 1)}")

    def on_player_ended(self):
        self.on_player_position(None)

    def on_player_state(self, state):
        self.b_play.setText({"playing": "❚❚ Pause", "paused": "▶ Resume"}.get(state, "▶ Play"))

    def _sync_timeline(self):
        v = (self.view.t0, self.view.t1) if self.view.idx is not None else None
        if v != self.timeline.view:
            self.timeline.view = v
            self.timeline.update()

    # ---------------- watermark
    def on_mode_changed(self, _=None):
        mode = self.w_mode.currentData()
        if mode == "add":
            self.w_amount.setRange(-90, -6)
            self.w_amount.setValue(-35)
            self.w_amount.setSuffix(" dBFS")
        elif mode == "boost":
            self.w_amount.setRange(1, 60)
            self.w_amount.setValue(15)
            self.w_amount.setSuffix(" dB")
        else:
            self.w_amount.setRange(1, 90)
            self.w_amount.setValue(40)
            self.w_amount.setSuffix(" dB")

    def wm_params(self):
        return dict(text=self.w_text.text(), shape=self.w_shape.currentData(),
                    bold=self.w_bold.isChecked(), size=self.w_size.value(),
                    mode=self.w_mode.currentData(), amount=self.w_amount.value(),
                    invert=self.w_invert.isChecked())

    def _check_wm_ready(self, p):
        if self.orig is None:
            self.warn("Open a file first.")
            return False
        if p["shape"] == "text" and not p["text"].strip():
            self.warn("Type some text, or switch Shape to 'Solid area'.")
            return False
        return True

    def _clamp_rect(self, t0, t1, f0, f1):
        dur = len(self.orig) / self.sr
        t0, t1 = max(t0, 0.0), min(t1, dur)
        f0, f1 = max(f0, 0.0), min(f1, self.sr / 2 - 100)
        return t0, t1, f0, f1

    def add_wm(self):
        p = self.wm_params()
        if not self._check_wm_ready(p):
            return
        sel = self.view.selection
        if not sel:
            return self.warn("Select an area first: choose the Select tool and drag on "
                             "the spectrogram.")
        t0, t1, f0, f1 = self._clamp_rect(*sel)
        if t1 - t0 < 0.2 or f1 - f0 < 100:
            return self.warn("That area is too small. Select a larger area.")
        p.update(t0=t0, t1=t1, f0=f0, f1=f1,
                 aspect=self.view.rect_aspect((t0, t1, f0, f1)), log=self.view.log)
        self.wms.append(p)
        self.rebuild_audio(select=len(self.wms) - 1)

    def update_wm(self):
        i = self.wm_list.currentRow()
        if i < 0:
            return self.warn("Select a watermark in the list first.")
        p = self.wm_params()
        if not self._check_wm_ready(p):
            return
        self.wms[i].update(p)
        self.rebuild_audio(select=i)

    def move_wm(self):
        i = self.wm_list.currentRow()
        if i < 0 or not self.view.selection:
            return self.warn("Select a watermark in the list and make a new selection first.")
        t0, t1, f0, f1 = self._clamp_rect(*self.view.selection)
        if t1 - t0 < 0.2 or f1 - f0 < 100:
            return self.warn("That area is too small.")
        self.wms[i].update(t0=t0, t1=t1, f0=f0, f1=f1,
                           aspect=self.view.rect_aspect((t0, t1, f0, f1)), log=self.view.log)
        self.rebuild_audio(select=i)

    def scale_wm(self, factor):
        i = self.wm_list.currentRow()
        if i < 0:
            return self.warn("Select a watermark in the list first.")
        w = self.wms[i]
        fw = (lambda f: float(np.log(max(f, 1.0)))) if w["log"] else (lambda f: float(f))
        iv = (lambda v: float(np.exp(v))) if w["log"] else (lambda v: float(v))
        tc, th = (w["t0"] + w["t1"]) / 2, (w["t1"] - w["t0"]) / 2 * factor
        fc, fh = (fw(w["f0"]) + fw(w["f1"])) / 2, (fw(w["f1"]) - fw(w["f0"])) / 2 * factor
        lo = iv(fc - fh)
        t0, t1, f0, f1 = self._clamp_rect(tc - th, tc + th, lo if not w["log"] else max(lo, 20.0),
                                          iv(fc + fh))
        w.update(t0=t0, t1=t1, f0=f0, f1=f1)
        self.rebuild_audio(select=i)

    def delete_wm(self):
        i = self.wm_list.currentRow()
        if i < 0:
            return
        del self.wms[i]
        self.rebuild_audio(select=min(i, len(self.wms) - 1))

    def refresh_wm_list(self, select=-1):
        self.wm_list.blockSignals(True)
        self.wm_list.clear()
        names = {"add": "add", "boost": "boost", "cut": "cut"}
        for k, w in enumerate(self.wms):
            label = f"“{w['text']}”" if w["shape"] == "text" else "solid area"
            self.wm_list.addItem(f"{k + 1}. {label} · {names[w['mode']]} · "
                                 f"{w['t0']:.1f}-{w['t1']:.1f}s · "
                                 f"{w['f0'] / 1000:.1f}-{w['f1'] / 1000:.1f}k")
        if 0 <= select < len(self.wms):
            self.wm_list.setCurrentRow(select)
        self.wm_list.blockSignals(False)
        self.on_wm_selected(self.wm_list.currentRow())

    def on_wm_selected(self, i):
        if i < 0 or i >= len(self.wms):
            self.view.highlight = None
            self.view.update()
            return
        w = self.wms[i]
        self.view.highlight = (w["t0"], w["t1"], w["f0"], w["f1"])
        self.w_shape.setCurrentIndex(self.w_shape.findData(w["shape"]))
        self.w_text.setText(w["text"])
        self.w_bold.setChecked(w["bold"])
        self.w_size.setValue(w["size"])
        self.w_mode.blockSignals(True)
        self.w_mode.setCurrentIndex(self.w_mode.findData(w["mode"]))
        self.w_mode.blockSignals(False)
        self.on_mode_changed()
        self.w_amount.setValue(w["amount"])
        self.w_invert.setChecked(w["invert"])
        self.view.update()

    def rebuild_audio(self, select=-1):
        """Queue a background rebuild; playback, view, selection and tabs stay as they are."""
        self.refresh_wm_list(select)
        self.request_update(rebuild=True)

    def export_audio(self):
        if self.data is None:
            return self.warn("Open a file first.")
        base = os.path.splitext(self.path)[0] + ("_watermarked" if self.wms else "_copy") + ".mp3"
        out, _ = QFileDialog.getSaveFileName(self, "Export audio", base,
                                             "MP3 (*.mp3);;FLAC (*.flac);;WAV (*.wav)")
        if not out:
            return
        QApplication.setOverrideCursor(Qt.WaitCursor)
        try:
            write_audio(out, self.data, self.sr)
            if out.lower().endswith(".mp3"):
                try:
                    tags = ID3(self.path) if self.is_mp3() else ID3()
                except ID3NoHeaderError:
                    tags = ID3()
                self._apply_text_tags(tags)
                tags.save(out)
        except Exception as e:
            QApplication.restoreOverrideCursor()
            return self.warn(f"Export failed: {e}")
        QApplication.restoreOverrideCursor()
        self.status.setText(f"Exported {out}")

    def export_png(self):
        if self.view.idx is None:
            return
        base = os.path.splitext(self.path)[0] + "_spectrogram.png"
        p, _ = QFileDialog.getSaveFileName(self, "Export current view as PNG", base, "PNG (*.png)")
        if p:
            self.view.export_png(p)
            self.status.setText(f"Saved {p}")

    # ---------------- markers
    def refresh_markers(self):
        self.view.markers.sort(key=lambda m: m["t"])
        self.mlist.clear()
        for m in self.view.markers:
            self.mlist.addItem(f"{fmt_time(m['t'])} | {m['f'] / 1000:.1f} kHz | {m['text']}")
        self.view.update()

    def on_marker_click(self, t, f):
        text, ok = QInputDialog.getText(self, "Marker", "Marker text:", text="© copyright")
        if ok and text.strip():
            self.view.markers.append({"t": t, "f": f, "text": text.strip()})
            self.refresh_markers()

    def edit_marker(self, item):
        m = self.view.markers[self.mlist.row(item)]
        text, ok = QInputDialog.getText(self, "Edit marker", "Text:", text=m["text"])
        if ok and text.strip():
            m["text"] = text.strip()
            self.refresh_markers()

    def del_marker(self):
        i = self.mlist.currentRow()
        if i >= 0:
            del self.view.markers[i]
            self.refresh_markers()

    # ---------------- ID3
    def load_tags(self):
        for w in (self.t_title, self.t_artist, self.t_album, self.t_copy, self.t_comm):
            w.clear()
        if not self.is_mp3():
            return
        try:
            tags = ID3(self.path)
        except ID3NoHeaderError:
            return

        def g(key):
            f = tags.get(key)
            return str(f.text[0]) if f and f.text else ""
        self.t_title.setText(g("TIT2"))
        self.t_artist.setText(g("TPE1"))
        self.t_album.setText(g("TALB"))
        self.t_copy.setText(g("TCOP"))
        comm = tags.getall("COMM")
        if comm and comm[0].text:
            self.t_comm.setText(str(comm[0].text[0]))
        for ch in tags.getall("CHAP"):
            title = ch.sub_frames.get("TIT2")
            fr = ch.sub_frames.getall("TXXX:freq_hz") if hasattr(ch.sub_frames, "getall") else []
            freq = float(fr[0].text[0]) if fr else self.sr / 4
            if title:
                self.view.markers.append({"t": ch.start_time / 1000.0, "f": freq,
                                          "text": str(title.text[0])})

    def _apply_text_tags(self, tags):
        for key, cls, w in (("TIT2", TIT2, self.t_title), ("TPE1", TPE1, self.t_artist),
                            ("TALB", TALB, self.t_album), ("TCOP", TCOP, self.t_copy)):
            tags.delall(key)
            if w.text().strip():
                tags.add(cls(encoding=3, text=[w.text().strip()]))
        tags.delall("COMM")
        if self.t_comm.text().strip():
            tags.add(COMM(encoding=3, lang="eng", desc="", text=[self.t_comm.text().strip()]))

    def save_tags(self):
        if not self.is_mp3():
            return self.warn("Tag saving works on MP3 files.")
        try:
            tags = ID3(self.path)
        except ID3NoHeaderError:
            tags = ID3()
        self._apply_text_tags(tags)
        tags.save(self.path)
        self.status.setText("Tags saved.")

    def write_markers(self):
        if not self.is_mp3():
            return self.warn("Writing ID3 chapters works on MP3 files.")
        if not self.view.markers:
            return self.warn("Add at least one marker first.")
        if QMessageBox.question(self, "SPECTRA", "Write markers into this MP3's ID3 tag?\n"
                                "(The audio data is not touched.)") != QMessageBox.Yes:
            return
        try:
            tags = ID3(self.path)
        except ID3NoHeaderError:
            tags = ID3()
        tags.delall("CHAP")
        tags.delall("CTOC")
        ms = sorted(self.view.markers, key=lambda m: m["t"])
        ids = []
        for i, m in enumerate(ms):
            eid = f"mk{i}"
            ids.append(eid)
            start = int(m["t"] * 1000)
            tags.add(CHAP(element_id=eid, start_time=start, end_time=start + 1,
                          start_offset=0xFFFFFFFF, end_offset=0xFFFFFFFF,
                          sub_frames=[TIT2(encoding=3, text=[m["text"]]),
                                      TXXX(encoding=3, desc="freq_hz", text=[f"{m['f']:.0f}"])]))
        tags.add(CTOC(element_id="toc", flags=CTOCFlags.TOP_LEVEL | CTOCFlags.ORDERED,
                      child_element_ids=ids, sub_frames=[TIT2(encoding=3, text=["Markers"])]))
        self._apply_text_tags(tags)
        tags.save(self.path)
        self.status.setText(f"Wrote {len(ms)} marker(s) to {os.path.basename(self.path)}.")

    def closeEvent(self, e):
        self.player.stop()
        if self.worker is not None:
            self.worker.wait(5000)
        super().closeEvent(e)


def main():
    app = QApplication(sys.argv)
    app.setStyle("Fusion")
    app.setStyleSheet(STYLE)
    w = MainWindow()
    w.show()
    if len(sys.argv) > 1 and os.path.isfile(sys.argv[1]):
        w.load(sys.argv[1])
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
