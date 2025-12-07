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

