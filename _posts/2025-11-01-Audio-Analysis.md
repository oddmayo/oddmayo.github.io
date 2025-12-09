---
layout: post
toc: true
title: Audio Analysis - How Porter Robinson Rewired my Brain
author: oddmayo
feature-img: "assets/img/feature-img/hawk.jpg"
thumbnail: "assets/img/thumbnails/feature-img/hawk.jpg"
color: rgba(210, 115, 19, 1)
tags: [Audio, Music]
categories: audio
favicon: assets/brain.ico
mathjax: true

---

Work in progress...

In this post I'll explain the fundamentals of audio signal processing from what I've learned about [Music Information Retrieval (MIR)](https://musicinformationretrieval.com/content/1_introduction/why_mir.html). Audio analysis is something I have always wanted to do but never really got around to it. First because it looked hard, and second I've never really come across data in audio format during my data scientist career.

I can't say I 'grew up' listening to Porter Robinson, but it's been a while. First time I heard his music was while playing the first Forza Horizon in Xbox 360 back in 2012 and thought it was pretty cool but didn't think much about it. Later on with the years, I noticed drastic change of styles through the albums, from your typical electro music to more emotional and melodic compositions.

We'll compare two tracks from Porter Robinson discography:

- "Sea of Voices" from the album "Worlds" (2014).

<div style="max-width:300px; margin:auto;">
  {% include aligner.html images="posts/audio/worlds.jpg" column=1 caption="Worlds album cover" %}
</div>

- "Wind Tempos" from the album "Nurture" (2021).

<div style="max-width:300px; margin:auto;">
  {% include aligner.html images="posts/audio/nurture.png" column=1 caption="Nurture album cover" %}
</div>

There is a great library called [librosa](https://librosa.org/doc/latest/index.html) for this sort of analysis, and while it would be easy to just use it calling just calling functions, I wanted to understand the math behind it. That's why we'll implement the algorithms from scratch using NumPy where possible, with mathematical explanations, while using librosa for loading and validation.

**CONTENTS:**

* TOC 
{:toc}

You can download the code from this post here: [oddmayo/audio-analysis](https://github.com/oddmayo/audio-analysis)

**languages:** 

![Python](https://img.shields.io/badge/python-3670A0?style=for-the-badge&logo=python&logoColor=white)


**requirements:** \

# Setup and Loading

``` python
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from IPython.display import Audio
import librosa
import librosa.display
```

## Load Audio Files

We use librosa to load the data, in my case I have FLAC files from my purchased albums but you can load MP3 or WAV files as well. Setting `sr=None` preserves the native sample rate (typically 44100 Hz for CD-quality audio).

``` python
# Load audio files
# Load Sea of Voices
filename_sov = str(Path('flacs') / 'sea-of-voices.flac')
y_sov, sr_sov = librosa.load(filename_sov, sr=None)

# Load Wind Tempos
filename_wt = str(Path('flacs') / 'wind-tempos.flac')
y_wt, sr_wt = librosa.load(filename_wt, sr=None)

# Quick summary
{
    "Sea of Voices": {
        "duration_sec": len(y_sov) / sr_sov,
        "sample_rate": sr_sov
    },
    "Wind Tempos": {
        "duration_sec": len(y_wt) / sr_wt,
        "sample_rate": sr_wt
    }
}
```

output:

``` 
{'Sea of Voices': {'duration_sec': 298.9808843537415,
  'sample_rate': 44100},
 'Wind Tempos': {'duration_sec': 364.11496598639457,
  'sample_rate': 44100} }
```

The durations are approximately 5 minutes for "Sea of Voices" and just over 6 minutes for "Wind Tempos".


# First Listen: Opening Atmospheres

Before diving into analysis, let's hear how each track opens. Notices the contrast:

``` python
# Sea of Voices: Opening 10 seconds — ethereal pad entrance
print("🌊 Sea of Voices — Opening atmosphere (0:00-0:10)")
audio_excerpt(y_sov, sr_sov, start_sec=0, duration_sec=10)
```

<audio controls>
  <source src="/assets/audio/porter-robinson/sea-opening.wav" type="audio/wav">
  Your browser does not support the audio element.
</audio>


``` python
# Wind Tempos: First piano entrance around 0:08
print("🍃 Wind Tempos — First piano notes (0:05-0:15)")
audio_excerpt(y_wt, sr_wt, start_sec=5, duration_sec=10)
```

<audio controls>
  <source src="/assets/audio/porter-robinson/wind-opening.wav" type="audio/wav">
  Your browser does not support the audio element.
</audio>

The opening of "Sea of Voices" is characterized by a lush, evolving pad sound that creates an immersive atmosphere, while "Wind Tempos" introduces a more immediate and melodic piano motif that sets a different emotional tone.

# Understanding Digital Audio

Digital audio represents continuous sound waves as discrete samples.

## Sample Rate (fs)

The **sample rate** determines how many times per second the audio signal is measured. By the **Nyquist theorem**, to accurately capture a frequency $$f$$, we need a sample rate of at least $$2f$$:

$$f_s \geq 2f_{max}$$

Since human hearing ranges up to ~20 kHz, CD-quality audio uses 44.1 kHz (or 48 kHz for video).

## Amplitude

Each sample is a floating-point value (typically normalized to [-1, 1]) representing the air pressure displacement at that instant.

## Duration

Total duration in seconds:

$$T = \frac{N}{f_s}$$

where $$N$$ is the total number of samples.



``` python
# Examining the raw sample data
# Audio is just a 1D array of amplitude values

# First 100 samples of Sea of Voices
sample_view = y_sov[:100]

fig, ax = plt.subplots(figsize=(12, 3))
ax.stem(sample_view, linefmt='C0-', markerfmt='C0o', basefmt='k-')
ax.set_xlabel('Sample Index')
ax.set_ylabel('Amplitude')
ax.set_title(f'First 100 samples of Sea of Voices (sr={sr_sov} Hz)')
ax.axhline(y=0, color='k', linewidth=0.5)
plt.tight_layout()
```

<div style="max-width:1000px; margin:auto;">
  {% include aligner.html images="posts/audio/sea-samples.png" column=1 caption="raw sample data" %}
</div>

The plot shows the first 100 samples of "Sea of Voices". Each stem represents the amplitude at that sample index. The waveform oscillates around zero, indicating positive and negative air pressure variations.

# Waveform Visualization

The waveform shows amplitude over time. It reveals the overall dynamic structure of a track:
- **Sea of Voices**: Gradual builds with atmospheric, sustained layers
- **Wind Tempos**: More percussive attacks with piano transients

``` python
# Compare full waveforms side by side
fig, axes = plt.subplots(2, 1, figsize=(14, 6), sharex=False)

# Sea of Voices
time_sov = np.arange(len(y_sov)) / sr_sov
axes[0].plot(time_sov, y_sov, linewidth=0.3, color='#4A90D9')
axes[0].set_ylabel('Amplitude')
axes[0].set_title('Sea of Voices — Ethereal layers, gradual dynamics')
axes[0].set_xlim(0, time_sov[-1])

# Wind Tempos
time_wt = np.arange(len(y_wt)) / sr_wt
axes[1].plot(time_wt, y_wt, linewidth=0.3, color='#7AC36A')
axes[1].set_xlabel('Time (seconds)')
axes[1].set_ylabel('Amplitude')
axes[1].set_title('Wind Tempos — Piano attacks, dynamic expression')
axes[1].set_xlim(0, time_wt[-1])

plt.tight_layout()
```

<div style="max-width:1000px; margin:auto;">
  {% include aligner.html images="posts/audio/waveform.png" column=1 caption="waveform comparison" %}
</div>

The waveform plots illustrate the contrasting dynamics of the two tracks. "Sea of Voices" shows smooth, flowing amplitude changes, while "Wind Tempos" has sharper peaks corresponding to piano notes.

``` python
# Listen to excerpts
# Sea of Voices - the vocal build around 30 seconds
audio_excerpt(y_sov, sr_sov, start_sec=25, duration_sec=10)
```

<audio controls>
  <source src="/assets/audio/porter-robinson/sea-wave.wav" type="audio/wav">
  Your browser does not support the audio element.
</audio>

``` python
# Wind Tempos - opening piano
audio_excerpt(y_wt, sr_wt, start_sec=0, duration_sec=15)
```

<audio controls>
  <source src="/assets/audio/porter-robinson/wind-wave.wav" type="audio/wav">
  Your browser does not support the audio element.
</audio>

# The Fourier Transform

The **Discrete Fourier Transform (DFT)** decomposes a signal into its constituent frequencies. For a signal $$x[n]$$ of length $$N$$:

$$X[k] = \sum_{n=0}^{N-1} x[n] \cdot e^{-i2\pi kn/N}$$

where:
- $$k$$ is the frequency bin index.

- $$X[k]$$ is a complex number whose magnitude $$\lvert X[k] \rvert$$ gives the amplitude at that frequency.

- The phase $$\angle X[k]$$ gives the phase offset.

## Implementing the DFT

Below we implement the DFT using pure NumPy, then compare to the Fast Fourier Transform (FFT).

``` python
def dft_naive(x):
    """
    Compute the Discrete Fourier Transform (naive O(N²) implementation).
    
    DFT formula: X[k] = Σ x[n] * e^(-i*2π*k*n/N) for n=0..N-1
    
    Parameters:
        x: Input signal (1D array)
    Returns:
        X: Complex DFT coefficients
    """
    N = len(x)
    n = np.arange(N)
    k = n.reshape((N, 1))
    
    # Create the DFT matrix: each entry is e^(-i*2π*k*n/N)
    W = np.exp(-2j * np.pi * k * n / N)
    
    # Matrix-vector multiplication gives us all DFT coefficients
    return np.dot(W, x)

# Test on a small segment (FFT is O(N log N), DFT is O(N²))
test_segment = y_sov[:1024]

# Compare our implementation to NumPy's FFT
dft_result = dft_naive(test_segment)
fft_result = np.fft.fft(test_segment)

# Verify they match
np.allclose(dft_result, fft_result)
```

output:

``` 
True
```

## Frequency Spectrum Visualization

The magnitude spectrum shows which frequencies are present in the audio. We'll analyze a section from each track.

``` python
def plot_magnitude_spectrum(y, sr, start_sec, duration_sec, title):
    """Plot the magnitude spectrum of an audio segment."""
    start_sample = int(start_sec * sr)
    end_sample = int((start_sec + duration_sec) * sr)
    segment = y[start_sample:end_sample]
    
    # Apply FFT
    N = len(segment)
    fft_result = np.fft.fft(segment)
    
    # Only positive frequencies (up to Nyquist)
    freqs = np.fft.fftfreq(N, 1/sr)[:N//2]
    magnitudes = np.abs(fft_result)[:N//2]
    
    # Convert to dB scale
    magnitudes_db = 20 * np.log10(magnitudes + 1e-10)
    
    return freqs, magnitudes_db, title

# Analyze 5-second segments from each track
fig, axes = plt.subplots(2, 1, figsize=(14, 8))

# Sea of Voices - during the ethereal build
freqs_sov, mag_sov, _ = plot_magnitude_spectrum(y_sov, sr_sov, 30, 5, "Sea of Voices")
axes[0].plot(freqs_sov, mag_sov, linewidth=0.5, color='#4A90D9')
axes[0].set_xlim(0, 10000)
axes[0].set_xlabel('Frequency (Hz)')
axes[0].set_ylabel('Magnitude (dB)')
axes[0].set_title('Sea of Voices (30-35s) — Rich harmonic content from layered synths')
axes[0].grid(True, alpha=0.3)

# Wind Tempos - piano section
freqs_wt, mag_wt, _ = plot_magnitude_spectrum(y_wt, sr_wt, 30, 5, "Wind Tempos")
axes[1].plot(freqs_wt, mag_wt, linewidth=0.5, color='#7AC36A')
axes[1].set_xlim(0, 10000)
axes[1].set_xlabel('Frequency (Hz)')
axes[1].set_ylabel('Magnitude (dB)')
axes[1].set_title('Wind Tempos (30-35s) — Distinct piano harmonic partials')
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
```

<div style="max-width:1000px; margin:auto;">
  {% include aligner.html images="posts/audio/frequency-spectrum.png" column=1 caption="frequency spectrum viz" %}
</div>

# Spectrograms: Time-Frequency Analysis

The **Short-Time Fourier Transform (STFT)** applies the DFT to overlapping windows of the signal, producing a time-frequency representation.

For a window function $$w[n]$$ of length $$L$$:

$$X[m, k] = \sum_{n=0}^{L-1} x[n + mH] \cdot w[n] \cdot e^{-i2\pi kn/L}$$

where:
- $$m$$ is the frame (time) index
- $$H$$ is the hop length (samples between frames)
- $$k$$ is the frequency bin

## Implementation from scratch

``` python
def stft_numpy(x, n_fft=2048, hop_length=512, window='hann'):
    """
    Compute the Short-Time Fourier Transform from scratch.
    
    Parameters:
        x: Input signal
        n_fft: FFT size (window length)
        hop_length: Number of samples between frames
        window: Window function type
    
    Returns:
        S: Complex STFT matrix (n_fft//2+1, n_frames)
    """
    # Create window function
    if window == 'hann':
        win = 0.5 - 0.5 * np.cos(2 * np.pi * np.arange(n_fft) / n_fft)
    else:
        win = np.ones(n_fft)
    
    # Pad signal to center the first window
    x_padded = np.pad(x, (n_fft // 2, n_fft // 2), mode='reflect')
    
    # Number of frames
    n_frames = 1 + (len(x_padded) - n_fft) // hop_length
    
    # Pre-allocate STFT matrix
    S = np.zeros((n_fft // 2 + 1, n_frames), dtype=np.complex128)
    
    for m in range(n_frames):
        # Extract windowed frame
        start = m * hop_length
        frame = x_padded[start:start + n_fft] * win
        
        # Compute FFT and keep positive frequencies
        fft_frame = np.fft.fft(frame)
        S[:, m] = fft_frame[:n_fft // 2 + 1]
    
    return S

# Compare our implementation to librosa
test_seg = y_sov[sr_sov*30:sr_sov*31]  # 1 second
S_ours = stft_numpy(test_seg, n_fft=2048, hop_length=512)
S_librosa = librosa.stft(test_seg, n_fft=2048, hop_length=512)

# Check correlation (won't be exact due to edge handling)
correlation = np.corrcoef(np.abs(S_ours).flatten(), np.abs(S_librosa).flatten())[0, 1]
f"Correlation with librosa STFT: {correlation:.6f}"
```

output:

```
Correlation with librosa STFT: 0.995549'
```

## Spectrogram Comparison

A spectrogram is the magnitude of the STFT, typically displayed in decibels on a log-frequency scale.

``` python
# Generate spectrograms using librosa for both tracks
fig, axes = plt.subplots(2, 1, figsize=(14, 10))

# Sea of Voices - first 60 seconds
S_sov = librosa.stft(y_sov[:60*sr_sov])
S_db_sov = librosa.amplitude_to_db(np.abs(S_sov), ref=np.max)

img1 = librosa.display.specshow(S_db_sov, sr=sr_sov, hop_length=512, 
                                 x_axis='time', y_axis='log', ax=axes[0], cmap='magma')
axes[0].set_title('Sea of Voices — Dense harmonic textures, reverberant layers')
axes[0].set_ylabel('Frequency (Hz)')
fig.colorbar(img1, ax=axes[0], format='%+2.0f dB')

# Wind Tempos - first 60 seconds
S_wt = librosa.stft(y_wt[:60*sr_wt])
S_db_wt = librosa.amplitude_to_db(np.abs(S_wt), ref=np.max)

img2 = librosa.display.specshow(S_db_wt, sr=sr_wt, hop_length=512,
                                 x_axis='time', y_axis='log', ax=axes[1], cmap='magma')
axes[1].set_title('Wind Tempos — Clear piano transients, defined attacks')
axes[1].set_ylabel('Frequency (Hz)')
axes[1].set_xlabel('Time (seconds)')
fig.colorbar(img2, ax=axes[1], format='%+2.0f dB')

plt.tight_layout()
```

<div style="max-width:1000px; margin:auto;">
  {% include aligner.html images="posts/audio/spectogram.png" column=1 caption="spectograms comparison" %}
</div>

## Hearing the Spectogram: Key Moments

The spectrogram reveals structure we can now *hear*:
- **Sea of Voices @ 0:30** — Notice the dense harmonic buildup visible as bright horizontal bands
- **Wind Tempos @ 0:40** — Clear piano transients appear as sharp vertical lines

``` python
# Sea of Voices: Dense harmonic section @ 0:25-0:35
print("🌊 Sea of Voices — Harmonic buildup (0:25-0:35)")
print("   Listen for the layered synth textures creating those bright spectral bands")
audio_excerpt(y_sov, sr_sov, start_sec=25, duration_sec=10)
```

<audio controls>
  <source src="/assets/audio/porter-robinson/sea-spectogram.wav" type="audio/wav">
  Your browser does not support the audio element.
</audio>


``` python
# Wind Tempos: Piano transients section @ 0:35-0:45
print("🍃 Wind Tempos — Piano transients (0:35-0:45)")
print("   Notice the sharp attacks — these create the vertical lines in the spectrogram")
audio_excerpt(y_wt, sr_wt, start_sec=35, duration_sec=10)
```

<audio controls>
  <source src="/assets/audio/porter-robinson/wind-spectogram.wav" type="audio/wav">
  Your browser does not support the audio element.
</audio>

# Spectral Features

Spectral features summarize properties of the frequency content. These are essential for music information retrieval.
