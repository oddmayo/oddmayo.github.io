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

