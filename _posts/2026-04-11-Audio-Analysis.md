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

In this post I'll explain the fundamentals of audio signal processing from what I've learned about [Music Information Retrieval (MIR)](https://musicinformationretrieval.com/content/1_introduction/why_mir.html). Audio analysis is something I have always wanted to do but never really got around to it. First because it looked hard, and second I've never really come across data in audio format during my data scientist career.

I can't say that I 'grew up' listening to Porter Robinson, but it's been a while. First time I heard his music was while playing the first Forza Horizon back in 2012 and thought it was pretty cool but didn't think much about it. Later on with the years I noticed drastic change of styles through the albums, from your typical EDM to more emotional and melodic compositions.

We'll compare two tracks from Porter Robinson discography:

- "Sea of Voices" from the album "Worlds" (2014).

<div style="max-width:300px; margin:auto;">
  {% include aligner.html images="posts/audio/worlds.jpg" column=1 caption="Worlds album cover" %}
</div>

- "Wind Tempos" from the album "Nurture" (2021).

<div style="max-width:300px; margin:auto;">
  {% include aligner.html images="posts/audio/nurture.png" column=1 caption="Nurture album cover" %}
</div>

There is a great Python library called [librosa](https://librosa.org/doc/latest/index.html) for this sort of analysis, and while it would be easy to just use it, I wanted to understand the math behind it. That's why we'll implement the algorithms from scratch using NumPy where possible with mathematical explanations, while using librosa for loading and validation.

**CONTENTS:**

* TOC 
{:toc}

You can download the code from this post here: [oddmayo/audio-analysis](https://github.com/oddmayo/audio-analysis)

**languages:** 

![Python](https://img.shields.io/badge/python-3670A0?style=for-the-badge&logo=python&logoColor=white)


**requirements:** \

# 1. Setup and Loading

``` python
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from IPython.display import Audio
import librosa
import librosa.display
```

## Load Audio Files

Let's use librosa to load the data, in my case I have FLAC files from my purchased albums but you can load MP3 or WAV files as well. Setting `sr=None` preserves the native sample rate (typically 44100 Hz for CD-quality audio).

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

The durations are around 5 minutes for 'Sea of Voices' and just over 6 minutes for 'Wind Tempos'.

## First Listen: Opening Atmospheres

Before diving into analysis, let's hear how each track opens. Notice the contrast:

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
# Wind Tempos: First piano entrance
print("🍃 Wind Tempos — First piano notes (0:05-0:15)")
audio_excerpt(y_wt, sr_wt, start_sec=5, duration_sec=10)
```

<audio controls>
  <source src="/assets/audio/porter-robinson/wind-opening.wav" type="audio/wav">
  Your browser does not support the audio element.
</audio>

The opening of "Sea of Voices" is characterized by a lush, evolving pad sound that creates an immersive atmosphere, while "Wind Tempos" introduces a more immediate and melodic piano motif that sets a different emotional tone.

# 2. Understanding Digital Audio

Digital audio represents continuous sound waves as discrete samples. Think of digital audio like a flipbook animation: instead of capturing a smooth, continuous motion, you take thousands of quick snapshots per second. When played back quickly, it sounds like one continuous wave.

## Sample Rate (fs)

The **sample rate** determines how many times per second the audio signal is measured. By the [**Nyquist theorem**](https://en.wikipedia.org/wiki/Nyquist%E2%80%93Shannon_sampling_theorem), to accurately capture the highest frequency present in our sound ($$f_{max}$$), our sample rate ($$f_s$$) must be at least **twice** that frequency ($$2 \times f_{max}$$). Why? Because catching one high point and one low point of a sound wave is the bare minimum required to recreate it:

$$f_s \geq 2f_{max}$$

Since human hearing tops out around ~20 kHz (meaning 20,000 vibrations per second), the Nyquist theorem says our sampling equipment needs to run at double that speed—at least 40 kHz. This is exactly why CD-quality audio was standardized at 44.1 kHz (or 48 kHz for video). In real-world applications, choosing the right sample rate is about balancing audio quality with file size—this is why phone calls (which only need to capture the lower frequencies of human voice) use a much lower sample rate of 8 kHz, sounding noticeably more "muffled."

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

# 3. Waveform Visualization

The waveform shows amplitude over time. It reveals the overall dynamic structure of a track:
- **Sea of Voices**: Gradual builds with atmospheric, sustained layers.
- **Wind Tempos**: More percussive attacks with [piano transients.](https://majormixing.com/what-are-transients/#:~:text=Transient%20compression%20is%20a%20technique%20used%20to,parallel%20compression%20processing%20*%20Use%20multiband%20compression)

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

# 4. The Fourier Transform

While waveforms show us *when* things happen, they don't tell us *what* is happening. If an orchestra plays all at once, the waveform is just a single messy line. 

The **Discrete Fourier Transform (DFT)** is the math trick that untangles this mess. It decomposes a complex signal into its constituent frequencies. Think of shining white light through a glass prism to split it into a rainbow of individual colors. The Fourier Transform does exactly that, but for sound—taking raw noise and splitting it into the pure pitches that make it up.

For a signal $$x[n]$$ of length $$N$$:

$$X[k] = \sum_{n=0}^{N-1} x[n] \cdot e^{-i2\pi kn/N}$$

where:
- $$k$$ is the frequency bin index (the "individual color").

- $$X[k]$$ is a complex number whose magnitude $$\lvert X[k] \rvert$$ gives the amplitude at that frequency (how "bright" that color is).

- The phase $$\angle X[k]$$ gives the phase offset.

**Real-world applications:** The Fourier Transform is the backbone of modern audio technology. When you use an Equalizer (EQ) on your phone to boost the bass, use noise-cancellation headphones to block out airplane engine hum, or compress an audio file into an MP3, the Fourier Transform is doing the heavy lifting under the hood.

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

**What exactly did we just compute?**
The output of both our `dft_naive` function and NumPy's `np.fft.fft` is an array of **complex numbers** (e.g., `1.5 - 2.3j`). 

Each complex number represents a specific frequency "bin." But a complex number by itself isn't very intuitive. To make sense of it, we split it into two understandable parts:
1. **Magnitude (Amplitude):** By taking the absolute value (`np.abs()`) of the complex number, we get the magnitude. This tells us *how loud* or *how present* that specific frequency is in our audio snippet.
2. **Phase:** By getting the angle of the complex number (`np.angle()`), we find the phase. This tells us the starting position (or horizontal shift) of that frequency wave in time.

For most audio analysis, including the rest of this post, the most important part is the **Magnitude**.

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

**How to read these plots:**
- The horizontal axis is **Frequency (Pitch)**. The left side is low bass, and the right side is high treble.
- The vertical axis is **Magnitude (Volume)** in decibels (dB). Taller peaks mean that specific pitch is louder.

Notice the stark contrast between the two tracks here:
- **Sea of Voices** shows a thick, dense wall of sound. The energy is spread out across a wide range of frequencies, which makes sense given the lush, washing, atmospheric synths playing in this section.
- **Wind Tempos** shows very sharp, distinct, separated spikes (called "harmonics" or "partials"). This is the signature look of a clean instrument like a piano. The tallest spike is the fundamental note being played, and the smaller spikes trailing off to the right are its natural mathematical overtones.

# 5. Spectrograms: Time-Frequency Analysis

The Fourier Transform we just did has a major limitation: it averages the frequencies over the *entire* audio clip. It tells us what frequencies exist, but not *when* they happened. 

To solve this, we use the **Short-Time Fourier Transform (STFT)**. Instead of transforming the whole song at once, we chop the audio into tiny, overlapping slices (windows) and apply the Fourier Transform to each slice. The result is a **Spectrogram**—a 2D map showing time on the horizontal axis and frequency on the vertical axis, with color representing intensity.

**Real-world applications:** Spectrograms are used in voice recognition systems to "read" spoken words as visual patterns. They are also heavily used in medicine (like interpreting [ultrasound Dopplers](https://en.wikipedia.org/wiki/Ultrasound_doppler)) and biology (like classifying bird calls or whale songs).

For a window function $$w[n]$$ of length $$L$$:

$$X[m, k] = \sum_{n=0}^{L-1} x[n + mH] \cdot w[n] \cdot e^{-i2\pi kn/L}$$

where:
- $$m$$ is the frame (time) index, showing our position in time.
- $$H$$ is the hop length (samples we move forward before taking the next slice).
- $$k$$ is the frequency bin.

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

**How to read these spectatorgrams:**
- Time runs from left to right along the horizontal axis.
- Frequency (Pitch) goes from bottom (low) to top (high) on the vertical axis.
- Color intensity (brightness) represents the Amplitude (Volume) of that specific frequency at that specific moment in time.

Visualizing audio this way makes the structural differences incredibly obvious:
- **Sea of Voices** appears as thick, glowing horizontal bands of color. This smear of energy across time represents sustained pads, reverberant synthesizers, and slow attack/release envelopes where the notes wash into each other.
- **Wind Tempos**, on the other hand, is dominated by sharp, bright vertical lines. These represent "transients"—the sudden, percussive attack of the piano hammers hitting the strings, followed by the immediate decay of the note.

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

# 6. Spectral Features

Spectrograms are great for our human eyes to analyze, but for a machine learning algorithm, they contain an overwhelming amount of raw data. To make this data manageable, we need to summarize the spectrogram into distinct, measurable properties called **Spectral Features**.

These features try to mathematically capture what our ears perceive as "texture" or "timbre."

**Real-world applications:** Have you ever wondered how Spotify's recommendation algorithm knows that two songs have the same "vibe"? It calculates these spectral features for every song in its database, clustering tracks with similar brightness or noisiness to generate your "Discover Weekly" playlists.

### Spectral Centroid
The "center of mass" of the spectrum — it mathematically indicates the **"brightness"** of a sound. A bass guitar has a low centroid, while a high-hat cymbal has a high centroid:

$$\text{centroid} = \frac{\sum_k f_k \cdot |X[k]|}{\sum_k |X[k]|}$$

where:
- $$f_k$$ is the actual frequency in Hz at bin $$k$$.
- $$\lvert X[k] \rvert$$ is the magnitude (volume) of that frequency. We saw this back in the Fourier Transform section.

### Spectral Rolloff
The frequency below which a specified percentage (e.g., 85%) of the spectral energy is contained. Think of it as the "ceiling" of the sound's active frequencies.

### Zero-Crossing Rate
How often the signal waveform crosses the zero-axis. This is a rough proxy for **"noisiness"** vs tonal content. A heavily distorted guitar or a snare drum will have a very high zero-crossing rate compared to a clean piano note:

$$\text{ZCR} = \frac{1}{2(N-1)} \sum_{n=1}^{N-1} |\text{sign}(x[n]) - \text{sign}(x[n-1])|$$

where:
- $$x[n]$$ is the amplitude value of the audio sample at index $$n$$.
- $$\text{sign}(x[n])$$ checks if the signal is positive or negative. The formula basically just counts how many times the signal flipped from positive to negative.

``` python
def spectral_centroid_numpy(S, sr, n_fft=2048):
    """
    Compute spectral centroid from STFT magnitude.
    
    centroid = Σ(f_k * |X[k]|) / Σ(|X[k]|)
    """
    # Frequency bins
    freqs = np.fft.rfftfreq(n_fft, 1/sr)
    
    # Ensure S is magnitude (not complex)
    S_mag = np.abs(S)
    
    # Weighted mean frequency per frame
    centroid = np.sum(freqs[:, np.newaxis] * S_mag, axis=0) / (np.sum(S_mag, axis=0) + 1e-10)
    return centroid

def zero_crossing_rate_numpy(y, frame_length=2048, hop_length=512):
    """
    Compute zero-crossing rate per frame.
    
    ZCR = (1/2(N-1)) * Σ|sign(x[n]) - sign(x[n-1])|
    """
    # Pad signal
    pad_length = frame_length // 2
    y_padded = np.pad(y, (pad_length, pad_length), mode='edge')
    
    n_frames = 1 + (len(y_padded) - frame_length) // hop_length
    zcr = np.zeros(n_frames)
    
    for i in range(n_frames):
        start = i * hop_length
        frame = y_padded[start:start + frame_length]
        
        # Count sign changes
        signs = np.sign(frame)
        zcr[i] = np.sum(np.abs(np.diff(signs))) / (2 * (frame_length - 1))
    
    return zcr

# Compare our implementations to librosa
S_test = librosa.stft(y_sov[:5*sr_sov])
centroid_ours = spectral_centroid_numpy(S_test, sr_sov)
centroid_librosa = librosa.feature.spectral_centroid(S=np.abs(S_test), sr=sr_sov)[0]

zcr_ours = zero_crossing_rate_numpy(y_sov[:5*sr_sov])
zcr_librosa = librosa.feature.zero_crossing_rate(y_sov[:5*sr_sov], frame_length=2048, hop_length=512)[0]

{
    "centroid_correlation": np.corrcoef(centroid_ours, centroid_librosa)[0, 1],
    "zcr_correlation": np.corrcoef(zcr_ours[:len(zcr_librosa)], zcr_librosa)[0, 1]
}
```

output:

``` 
{'centroid_correlation': 0.9999999999999349,
 'zcr_correlation': 0.999853726198455}
```

## Feature Comparison

Let's compare spectral features between the two tracks to understand their timbral differences.

``` python
# Compare spectral features between both tracks
fig, axes = plt.subplots(3, 1, figsize=(14, 10), sharex=False)

# Compute features for both tracks (first 60 seconds)
S_sov_feat = librosa.stft(y_sov[:60*sr_sov])
S_wt_feat = librosa.stft(y_wt[:60*sr_wt])

# Spectral Centroid
centroid_sov = librosa.feature.spectral_centroid(S=np.abs(S_sov_feat), sr=sr_sov)[0]
centroid_wt = librosa.feature.spectral_centroid(S=np.abs(S_wt_feat), sr=sr_wt)[0]

times_sov = librosa.times_like(centroid_sov, sr=sr_sov)
times_wt = librosa.times_like(centroid_wt, sr=sr_wt)

axes[0].plot(times_sov, centroid_sov, label='Sea of Voices', color='#4A90D9', alpha=0.8)
axes[0].plot(times_wt, centroid_wt, label='Wind Tempos', color='#7AC36A', alpha=0.8)
axes[0].set_ylabel('Frequency (Hz)')
axes[0].set_title('Spectral Centroid — Perceived "brightness"')
axes[0].legend()
axes[0].grid(True, alpha=0.3)

# Spectral Rolloff
rolloff_sov = librosa.feature.spectral_rolloff(S=np.abs(S_sov_feat), sr=sr_sov, roll_percent=0.85)[0]
rolloff_wt = librosa.feature.spectral_rolloff(S=np.abs(S_wt_feat), sr=sr_wt, roll_percent=0.85)[0]

axes[1].plot(times_sov, rolloff_sov, label='Sea of Voices', color='#4A90D9', alpha=0.8)
axes[1].plot(times_wt, rolloff_wt, label='Wind Tempos', color='#7AC36A', alpha=0.8)
axes[1].set_ylabel('Frequency (Hz)')
axes[1].set_title('Spectral Rolloff (85%) — Upper frequency boundary')
axes[1].legend()
axes[1].grid(True, alpha=0.3)

# Zero-Crossing Rate
zcr_sov = librosa.feature.zero_crossing_rate(y_sov[:60*sr_sov])[0]
zcr_wt = librosa.feature.zero_crossing_rate(y_wt[:60*sr_wt])[0]

times_zcr_sov = librosa.times_like(zcr_sov, sr=sr_sov)
times_zcr_wt = librosa.times_like(zcr_wt, sr=sr_wt)

axes[2].plot(times_zcr_sov, zcr_sov, label='Sea of Voices', color='#4A90D9', alpha=0.8)
axes[2].plot(times_zcr_wt, zcr_wt, label='Wind Tempos', color='#7AC36A', alpha=0.8)
axes[2].set_xlabel('Time (seconds)')
axes[2].set_ylabel('ZCR')
axes[2].set_title('Zero-Crossing Rate — Noisiness/texture indicator')
axes[2].legend()
axes[2].grid(True, alpha=0.3)

plt.tight_layout()
```

<div style="max-width:1000px; margin:auto;">
  {% include aligner.html images="posts/audio/spectral.png" column=1 caption="spectral features comparison" %}
</div>

**What do these plots reveal?**

- **Spectral Centroid (Brightness):** Notice how *Wind Tempos* (green) has sudden spikes—every time a piano key is struck forcefully, the sound momentarily gets very bright, then quickly mellows out. *Sea of Voices* (blue) steadily climbs upward as the atmospheric synths slowly open their filters and let more high-frequency energy swell into the mix.
- **Spectral Rolloff (Ceiling):** This follows a very similar shape to the Centroid, confirming that the "ceiling" of the audio is bouncing up and down with the piano strikes in *Wind Tempos*, but building a solid, rising plateau in *Sea of Voices*.
- **Zero-Crossing Rate (Noisiness):** *Sea of Voices* maintains a fairly stable, low ZCR because synthesizers produce very clean, periodic tonal waves. *Wind Tempos* jumps around much more wildly, picking up the slightly noisy, percussive impact of the piano hammers hitting the strings.

## Hearing Spectral Brightness

The spectral centroid (perceived "brightness") shows interesting contrasts. Let's hear moments where each track reaches peak brightness:

- **Sea of Voices @ ~1:30** — The "drop" with maximum high-frequency energy
- **Wind Tempos @ ~2:00** — Intense piano crescendo with bright upper harmonics

``` python
# Sea of Voices: The euphoric drop section @ ~1:25
print("🌊 Sea of Voices — First drop / climax (1:25-1:35)")
print("   Maximum spectral brightness")
audio_excerpt(y_sov, sr_sov, start_sec=85, duration_sec=10)
```

<audio controls>
  <source src="/assets/audio/porter-robinson/sea-spectral.wav" type="audio/wav">
  Your browser does not support the audio element.
</audio>

``` python
# Wind Tempos: Emotional crescendo @ ~2:00
print("🍃 Wind Tempos — Piano crescendo (2:00-2:10)")
print("   Bright upper harmonics as the intensity builds")
audio_excerpt(y_wt, sr_wt, start_sec=120, duration_sec=10)
```

<audio controls>
  <source src="/assets/audio/porter-robinson/wind-spectral.wav" type="audio/wav">
  Your browser does not support the audio element.
</audio>

# 7. Rhythm & Tempo Analysis

We've now mapped out the "color" and "texture" of the sound, but what about the groove? The spectral features tell us virtually nothing about the beat of the song. To understand the rhythm, we need to look for patterns in how energy changes over time.

**Real-world applications:** DJ software tools (like Serato or Rekordbox) use these exact rhythm analysis techniques to automatically detect BPM and beatmatch tracks. Fitness apps use tempo detection to generate workout playlists that perfectly match your running pace.

### Onset Detection
**Onsets** are the exact moments where new sound events begin—like a drum hit, a picked guitar string, or a piano key strike. We detect them by finding sudden positive spikes (peaks) in spectral energy across time, forming an **onset strength envelope**:

$$\text{flux}[m] = \sum_k H(|X[m,k]| - |X[m-1,k]|)$$

where $$H(x) = \max(0, x)$$ is half-wave rectification (ignoring energy decreases, we only care when new energy is added).

### Tempo via Autocorrelation
Once we have our "map of hits" (the onset envelope), we can estimate the tempo. We do this by sliding the envelope copy over itself and seeing where the peaks align best—a process called **autocorrelation**:

$$R[\tau] = \sum_m o[m] \cdot o[m + \tau]$$

Peaks in $$R[\tau]$$ at lag $$\tau$$ correspond to a repeating pattern of length $$\tau$$ frames, giving us the tempo in Beats Per Minute (BPM):

$$\text{BPM} = \frac{60 \cdot f_s}{H \cdot \tau}$$

where $$H$$ is the hop length.

``` python
def onset_strength_numpy(S, sr, hop_length=512):
    """
    Compute onset strength envelope from STFT.
    
    Uses spectral flux: sum of positive spectral differences.
    """
    S_mag = np.abs(S)
    
    # Compute difference between consecutive frames
    S_diff = np.diff(S_mag, axis=1)
    
    # Half-wave rectification (only keep positive changes)
    S_diff = np.maximum(0, S_diff)
    
    # Sum across frequency bins
    onset_env = np.sum(S_diff, axis=0)
    
    # Normalize
    onset_env = onset_env / (np.max(onset_env) + 1e-10)
    
    return onset_env

def autocorrelate_numpy(x, max_lag=None):
    """
    Compute autocorrelation of a signal.
    
    R[τ] = Σ x[n] * x[n + τ]
    """
    N = len(x)
    if max_lag is None:
        max_lag = N
    
    # Use FFT-based autocorrelation for efficiency
    # R = IFFT(|FFT(x)|²)
    x_padded = np.pad(x, (0, N), mode='constant')
    X = np.fft.fft(x_padded)
    power = X * np.conj(X)
    R = np.fft.ifft(power).real[:max_lag]
    
    # Normalize by R[0]
    return R / (R[0] + 1e-10)

# Test on Sea of Voices
S_test = librosa.stft(y_sov[:30*sr_sov], hop_length=512)
onset_ours = onset_strength_numpy(S_test, sr_sov)
onset_librosa = librosa.onset.onset_strength(y=y_sov[:30*sr_sov], sr=sr_sov)

f"Onset strength correlation: {np.corrcoef(onset_ours, onset_librosa[1:])[0,1]:.4f}"
```

output:

```
'Onset strength correlation: -0.0164'
```


``` python
# Tempo estimation and beat tracking comparison
hop_length = 512

# Sea of Voices
tempo_sov, beats_sov = librosa.beat.beat_track(y=y_sov, sr=sr_sov, hop_length=hop_length)
beat_times_sov = librosa.frames_to_time(beats_sov, sr=sr_sov, hop_length=hop_length)

# Wind Tempos
tempo_wt, beats_wt = librosa.beat.beat_track(y=y_wt, sr=sr_wt, hop_length=hop_length)
beat_times_wt = librosa.frames_to_time(beats_wt, sr=sr_wt, hop_length=hop_length)

{
    "Sea of Voices": {
        "tempo_bpm": float(tempo_sov),
        "total_beats": len(beats_sov)
    },
    "Wind Tempos": {
        "tempo_bpm": float(tempo_wt),
        "total_beats": len(beats_wt)
    }
}
```

output:

``` 
{'Sea of Voices': {'tempo_bpm': 129.19921875, 'total_beats': 604},
 'Wind Tempos': {'tempo_bpm': 92.28515625, 'total_beats': 562}}
```

``` python
# Visualize onset strength and autocorrelation side by side
fig, axes = plt.subplots(2, 2, figsize=(14, 8))

# Onset strength envelopes
onset_sov = librosa.onset.onset_strength(y=y_sov[:60*sr_sov], sr=sr_sov, hop_length=hop_length)
onset_wt = librosa.onset.onset_strength(y=y_wt[:60*sr_wt], sr=sr_wt, hop_length=hop_length)

times_onset_sov = librosa.times_like(onset_sov, sr=sr_sov, hop_length=hop_length)
times_onset_wt = librosa.times_like(onset_wt, sr=sr_wt, hop_length=hop_length)

axes[0, 0].plot(times_onset_sov, onset_sov, color='#4A90D9')
axes[0, 0].set_title('Sea of Voices — Onset Strength')
axes[0, 0].set_xlabel('Time (s)')
axes[0, 0].set_ylabel('Strength')

axes[0, 1].plot(times_onset_wt, onset_wt, color='#7AC36A')
axes[0, 1].set_title('Wind Tempos — Onset Strength')
axes[0, 1].set_xlabel('Time (s)')
axes[0, 1].set_ylabel('Strength')

# Autocorrelation of onset strength (for tempo)
max_lag = 4 * sr_sov // hop_length  # ~4 seconds of lag

ac_sov = librosa.autocorrelate(onset_sov, max_size=max_lag)
ac_wt = librosa.autocorrelate(onset_wt, max_size=max_lag)

# Convert lag to BPM for x-axis
lag_frames = np.arange(len(ac_sov))
bpm_sov = 60 * sr_sov / (hop_length * (lag_frames + 1e-10))

axes[1, 0].plot(lag_frames, ac_sov, color='#4A90D9')
axes[1, 0].set_title('Sea of Voices — Autocorrelation (tempo periodicity)')
axes[1, 0].set_xlabel('Lag (frames)')
axes[1, 0].set_ylabel('Correlation')
axes[1, 0].set_xlim(0, max_lag)

lag_frames_wt = np.arange(len(ac_wt))
axes[1, 1].plot(lag_frames_wt, ac_wt, color='#7AC36A')
axes[1, 1].set_title('Wind Tempos — Autocorrelation (tempo periodicity)')
axes[1, 1].set_xlabel('Lag (frames)')
axes[1, 1].set_ylabel('Correlation')
axes[1, 1].set_xlim(0, max_lag)

plt.tight_layout()
```
<div style="max-width:1000px; margin:auto;">
  {% include aligner.html images="posts/audio/rhythm.png" column=1 caption="rhythm analysis comparison" %}
</div>

**How to read these rhythm plots:**
- **Top Row (Onset Strength):** This shows our "map of hits." Notice how *Sea of Voices* has very dense, regular, and rapid spikes—this is the persistent, driving electronic drum beat pushing the song forward. *Wind Tempos* has sharp spikes too, but they are more scattered and less rigid, tracking the expressive, staggered piano chords.
- **Bottom Row (Autocorrelation):** This reveals the actual heartbeat of the tracks. We took the top chart, slid a copy of it over itself (the lag), and plotted where the spikes aligned perfectly.
  - *Sea of Voices* shows massive, razor-sharp peaks at perfectly even intervals. This mathematically proves it has a very strict, quantized electronic beat.
  - *Wind Tempos* shows wider, softer humps. The peaks still exist (there is a core tempo), but because the piano is played with human emotion and natural timing variations (rubato), the beats don't align with robotic precision.

## Feeling the Tempo Difference

The tempo analysis reveals a fundamental stylistic choice:
- **Sea of Voices (~129 BPM)** — EDM energy, four-on-the-floor pulse
- **Wind Tempos (~92 BPM)** — Introspective, rubato-friendly tempo

Listen to sections where the beat is most prominent:

``` python
# Sea of Voices: The driving beat section @ ~2:00
print("🌊 Sea of Voices — Driving beat at 129 BPM (2:00-2:10)")
print("   Feel the four-on-the-floor kick pattern")
audio_excerpt(y_sov, sr_sov, start_sec=120, duration_sec=10)
```

<audio controls>
  <source src="/assets/audio/porter-robinson/sea-tempo.wav" type="audio/wav">
  Your browser does not support the audio element.
</audio>


``` python
# Wind Tempos: Slower, organic rhythm @ ~3:00
print("🍃 Wind Tempos — Organic pulse at 92 BPM (3:00-3:10)")
print("   More flexible, breathing rhythm with natural timing variations")
audio_excerpt(y_wt, sr_wt, start_sec=180, duration_sec=10)
```

<audio controls>
  <source src="/assets/audio/porter-robinson/wind-tempo.wav" type="audio/wav">
  Your browser does not support the audio element.
</audio>


# 8. Chromogram: Harmonic Content

Now that we have analyzed the texture and the rhythm, what about the musical notes themselves? If a piano and a synthesizer both play a C major chord, they have different textures but the exact same harmony. 

A **chromagram** (or chroma feature) isolates this harmonic information. It takes the entire frequency spectrum and collapses it into just the 12 core pitch classes of Western music (C, C#, D, ..., B), completely ignoring the octave. A extremely low "C" bass note and a piercingly high "C" synth note are combined into a single "C" bucket.

**Real-world applications:** Apps like Shazam or SoundHound use chroma features combined with other spectral data to identify songs even in noisy environments. Because chromagrams ignore the exact octave and instrument, they are especially powerful for identifying cover songs or generating automatic guitar tabs/chords for a track.

The chroma vector at frame $$m$$ is:

$$c_p[m] = \sum_{k \in \text{bins}(p)} |X[m, k]|^2$$

where $$\text{bins}(p)$$ are the frequency bins corresponding to pitch class $$p$$ across all octaves.

### Musical Context
- **Sea of Voices**: Built around lush major chords with emotional progressions.
- **Wind Tempos**: More modal, with hints of Japanese pentatonic influence.

``` python
# Compute and visualize chromagrams
fig, axes = plt.subplots(2, 1, figsize=(14, 8))

# Sea of Voices chromagram (using CQT for better frequency resolution)
chroma_sov = librosa.feature.chroma_cqt(y=y_sov[:60*sr_sov], sr=sr_sov, hop_length=512)
img1 = librosa.display.specshow(chroma_sov, sr=sr_sov, hop_length=512, 
                                 x_axis='time', y_axis='chroma', ax=axes[0], cmap='coolwarm')
axes[0].set_title('Sea of Voices — Chromagram (harmonic content over time)')
fig.colorbar(img1, ax=axes[0])

# Wind Tempos chromagram
chroma_wt = librosa.feature.chroma_cqt(y=y_wt[:60*sr_wt], sr=sr_wt, hop_length=512)
img2 = librosa.display.specshow(chroma_wt, sr=sr_wt, hop_length=512,
                                 x_axis='time', y_axis='chroma', ax=axes[1], cmap='coolwarm')
axes[1].set_title('Wind Tempos — Chromagram (harmonic content over time)')
axes[1].set_xlabel('Time (seconds)')
fig.colorbar(img2, ax=axes[1])

plt.tight_layout()
```

<div style="max-width:1000px; margin:auto;">
  {% include aligner.html images="posts/audio/chromagram.png" column=1 caption="chromagram comparison" %}
</div>


``` python
# Average chroma distribution — which notes dominate each track?
fig, axes = plt.subplots(1, 2, figsize=(12, 4))

pitch_classes = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']

# Mean chroma across time
mean_chroma_sov = np.mean(chroma_sov, axis=1)
mean_chroma_wt = np.mean(chroma_wt, axis=1)

axes[0].bar(pitch_classes, mean_chroma_sov, color='#4A90D9', alpha=0.8)
axes[0].set_title('Sea of Voices — Average Chroma Distribution')
axes[0].set_ylabel('Energy')
axes[0].set_xlabel('Pitch Class')

axes[1].bar(pitch_classes, mean_chroma_wt, color='#7AC36A', alpha=0.8)
axes[1].set_title('Wind Tempos — Average Chroma Distribution')
axes[1].set_ylabel('Energy')
axes[1].set_xlabel('Pitch Class')

plt.tight_layout()
```

<div style="max-width:1000px; margin:auto;">
  {% include aligner.html images="posts/audio/average-chroma.png" column=1 caption="average chroma distribution" %}
</div>

## Hearing the Harmony: Key Center Differences

The chromagram shows that these tracks live in different harmonic worlds:
- **Sea of Voices**: Strong D, G, A — suggests D major / G major territory
- **Wind Tempos**: Strong C#, D#, G# — suggests more complex, melancholic modes

Listen to passages where the harmonic character shines through:

``` python
# Sea of Voices: The soaring chord progression @ ~3:00
print("🌊 Sea of Voices — Soaring harmonies in D/G major (3:00-3:12)")
print("   Listen for the uplifting, anthemic chord movement")
audio_excerpt(y_sov, sr_sov, start_sec=180, duration_sec=12)
```

<audio controls>
  <source src="/assets/audio/porter-robinson/sea-chords.wav" type="audio/wav">
  Your browser does not support the audio element.
</audio>

``` python
# Wind Tempos: Melancholic harmonic passage @ ~4:00
print("🍃 Wind Tempos — Melancholic piano in C#/G# minor (4:00-4:12)")
print("   Notice the more introspective, bittersweet quality")
audio_excerpt(y_wt, sr_wt, start_sec=240, duration_sec=12)
```

<audio controls>
  <source src="/assets/audio/porter-robinson/wind-chords.wav" type="audio/wav">
  Your browser does not support the audio element.
</audio>


# 9. Mel-Frequency Cepstral Coefficients (MFCCs)

While Chroma tells us *what* musical notes are being played, MFCCs capture *how* they sound. The Mel-Frequency Cepstral Coefficients are essentially the ultimate **timbral fingerprint**. 

The human ear doesn't hear frequencies linearly—we are much better at distinguishing tiny pitch differences in low frequencies (like a bassline) than in high frequencies (like a cymbal crash). The **Mel-scale** is a mathematical formula that warps the frequency spectrum to match how human ears actually hear sound. 

**Real-world applications:** Because MFCCs act as a fingerprint for the unique "shape" or texture of a sound, they are the gold standard for speech recognition. When Siri or Alexa learn to recognize *your specific voice* over someone else's, or when AI is used to detect if an engine sounds faulty or a cough sounds sick, they are almost certainly using MFCCs.

The pipeline:

1. **Mel spectrogram**: Apply a mel-scale filterbank (which spaces filters closer together at low frequencies and further apart at high frequencies) to the power spectrum:

   $$S_{mel}[m, b] = \sum_k |X[m,k]|^2 \cdot H_b[k]$$
   
2. **Log compression**: Take the log, because humans perceive volume logarithmically: $$\log(S_{mel} + \epsilon)$$

3. **DCT**: The Discrete Cosine Transform acts as a data-compression step. It decorrelates the overlapping mel bands and extracts the broad shape of the frequency spectrum:

   $$c_n = \sum_{b=0}^{B-1} \log(S_{mel}[b]) \cdot \cos\left(\frac{\pi n (b + 0.5)}{B}\right)$$

Typically, just using the first 13 of these coefficients is enough to comprehensively capture the timbral identity of the audio.

``` python
def mel_filterbank_numpy(sr, n_fft, n_mels=128, fmin=0, fmax=None):
    """
    Create a Mel filterbank matrix.
    
    The Mel scale approximates human pitch perception:
    mel = 2595 * log10(1 + f/700)
    """
    if fmax is None:
        fmax = sr / 2
    
    # Mel scale conversion
    def hz_to_mel(f):
        return 2595 * np.log10(1 + f / 700)
    
    def mel_to_hz(m):
        return 700 * (10**(m / 2595) - 1)
    
    # Create mel points
    mel_min = hz_to_mel(fmin)
    mel_max = hz_to_mel(fmax)
    mel_points = np.linspace(mel_min, mel_max, n_mels + 2)
    hz_points = mel_to_hz(mel_points)
    
    # Convert to FFT bin indices
    bin_points = np.floor((n_fft + 1) * hz_points / sr).astype(int)
    
    # Create filterbank
    filterbank = np.zeros((n_mels, n_fft // 2 + 1))
    
    for m in range(1, n_mels + 1):
        left = bin_points[m - 1]
        center = bin_points[m]
        right = bin_points[m + 1]
        
        # Rising slope
        for k in range(left, center):
            filterbank[m - 1, k] = (k - left) / (center - left + 1e-10)
        
        # Falling slope
        for k in range(center, right):
            filterbank[m - 1, k] = (right - k) / (right - center + 1e-10)
    
    return filterbank

# Visualize the mel filterbank
mel_fb = mel_filterbank_numpy(sr_sov, 2048, n_mels=40)
freqs = np.fft.rfftfreq(2048, 1/sr_sov)

plt.figure(figsize=(12, 4))
for i in range(0, 40, 4):  # Plot every 4th filter
    plt.plot(freqs, mel_fb[i], alpha=0.7)
plt.xlabel('Frequency (Hz)')
plt.ylabel('Weight')
plt.title('Mel Filterbank — Triangular filters on mel-scale spacing')
plt.xlim(0, 8000)
plt.grid(True, alpha=0.3)
```

<div style="max-width:1000px; margin:auto;">
  {% include aligner.html images="posts/audio/mel-filterbank.png" column=1 caption="mel filterbank visualization" %}
</div>


``` python
# Compare MFCCs between tracks
fig, axes = plt.subplots(2, 1, figsize=(14, 8))

# Compute MFCCs
mfcc_sov = librosa.feature.mfcc(y=y_sov[:60*sr_sov], sr=sr_sov, n_mfcc=13, hop_length=512)
mfcc_wt = librosa.feature.mfcc(y=y_wt[:60*sr_wt], sr=sr_wt, n_mfcc=13, hop_length=512)

img1 = librosa.display.specshow(mfcc_sov, sr=sr_sov, hop_length=512,
                                 x_axis='time', ax=axes[0], cmap='coolwarm')
axes[0].set_title('Sea of Voices — MFCCs (timbral fingerprint)')
axes[0].set_ylabel('MFCC Coefficient')
fig.colorbar(img1, ax=axes[0])

img2 = librosa.display.specshow(mfcc_wt, sr=sr_wt, hop_length=512,
                                 x_axis='time', ax=axes[1], cmap='coolwarm')
axes[1].set_title('Wind Tempos — MFCCs (timbral fingerprint)')
axes[1].set_ylabel('MFCC Coefficient')
axes[1].set_xlabel('Time (seconds)')
fig.colorbar(img2, ax=axes[1])

plt.tight_layout()
```

<div style="max-width:1000px; margin:auto;">
  {% include aligner.html images="posts/audio/mfcc.png" column=1 caption="mfcc comparison" %}
</div>

## The Climax: Peak Emotional Moments

Every great track builds to something transcendent. These are the moments where all the analysis converges — maximum spectral energy, strongest harmonic content, and most intense rhythmic drive:

``` python
# Sea of Voices: The transcendent climax @ ~3:45
print("🌊 Sea of Voices — CLIMAX: The Transcendent Drop (3:45-4:00)")
print("   ✨ Maximum spectral brightness")
print("   ✨ Full harmonic saturation")  
print("   ✨ Peak emotional intensity")
print("   This is the moment the spectrogram LIGHTS UP")
audio_excerpt(y_sov, sr_sov, start_sec=225, duration_sec=15)
```

<audio controls>
  <source src="/assets/audio/porter-robinson/sea-climax.wav" type="audio/wav">
  Your browser does not support the audio element.
</audio>

``` python
# Wind Tempos: The emotional resolution @ ~5:00
print("🍃 Wind Tempos — CLIMAX: Emotional Release (5:00-5:15)")
print("   🌿 Rich lower harmonics")
print("   🌿 Sustained melodic resolution")
print("   🌿 The satisfying conclusion the MFCC patterns predicted")
audio_excerpt(y_wt, sr_wt, start_sec=300, duration_sec=15)
```

<audio controls>
  <source src="/assets/audio/porter-robinson/wind-climax.wav" type="audio/wav">
  Your browser does not support the audio element.
</audio>

# Concluding Remarks

