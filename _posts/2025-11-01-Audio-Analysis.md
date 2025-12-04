---
layout: post
toc: true
title: Audio Analysis - How Porter Robinson rewired my brain
author: oddmayo
feature-img: "assets/img/feature-img/hawk.jpg"
thumbnail: "assets/img/thumbnails/feature-img/hawk.jpg"
color: rgba(210, 115, 19, 1)
tags: [Audio, Music]
categories: audio
favicon: assets/brain.ico

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

# Setup and loading

``` python
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from IPython.display import Audio
import librosa
import librosa.display
```

## Load audio files

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


# First listen: Opening atmospheres

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
