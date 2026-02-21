# 🔊 Generic Audio Classifier

A powerful audio classification application using state-of-the-art deep learning models

## 🎯 What Can You Do?

This application allows you to classify audio files into various categories and subcategories using advanced machine learning models.

* **Upload** your audio files for instant classification
* **Record** audio directly through your microphone
* **Visualize** classification results with detailed analytics
* **Contribute** to the dataset by adding new labeled audio files
* **Explore** the existing dataset structure and examples

## 🧠 Powered by Advanced Models

| Model | Description | Accuracy |
|-------|-------------|----------|
| **NASNet Mobile** | Neural Architecture Search Network optimized for mobile | 95% |
| **EfficientNet V2 B0** | Optimized CNN with balanced performance | 87% |
| **DualNet CX** | Dual-pathway network for contextual features | 99% |
| **DualNet Xpert** | Expert system with dual feature extraction | 98% |

## 📊 Dataset Overview

| Metric | Count |
|--------|-------|
| Audio Files | 23,303 |
| Categories | 4 |
| Subcategories | 23 |

## 📈 Classification Visualization

The application provides detailed visualizations of classification results, including confidence scores for each category.

## Dataset Structure

```
GENERIC_AUDIO_CLASSIFIER
├── Animals
│   ├── CATS
│   ├── DOGS
│   ├── ELEPHANT
│   ├── HORSE
│   └── LIONS
├── Birds
│   ├── CROWS
│   ├── PARROT
│   ├── PEACOCK
│   └── SPARROW
├── Environment
│   ├── CROWD
│   ├── MILITARY
│   ├── OFFICE
│   ├── RAINFALL
│   ├── TRAFFIC
│   └── WIND
└── Vehicles
    ├── airplane
    ├── bicycle
    ├── bike
    ├── bus
    ├── car
    ├── helicopter
    ├── train
    └── truck
```

## 🔑 Key Features

| Feature | Description |
|---------|-------------|
| 🎙️ **Audio Processing** | Process various audio formats with intelligent feature extraction |
| 🔄 **Real-time Classification** | Get instant predictions with high accuracy and precision |
| 📊 **Advanced Visualization** | See detailed analytics and confidence scores for each prediction |
| 🔍 **Dynamic Dataset** | Flexible system that grows and improves with new data |

## Dataset Sources

The dataset includes audio samples from various sources:

- [Kaggle Generic Audio Samples Dataset](https://www.kaggle.com/datasets/lokeshbhaskarnr/generic-audio-samples) - Collected and Organized by me🤘
- Various YouTube videos (see Acknowledgements section)
- Vehicle sounds from [Kaggle Vehicle Sounds Dataset](https://www.kaggle.com/datasets/janboubiabderrahim/vehicle-sounds-dataset)

## Model Training Notebooks

- [Audio Classification - ConvNextTiny (95% accuracy)](https://www.kaggle.com/code/lokeshbhaskarnr/audio-classification-convnexttiny-95)
- [Audio Classification - EfficientNetV2B0 (87% accuracy)](https://www.kaggle.com/code/lokeshbhaskarnr/audio-classification-efficientnetv2b0-87)
- [Audio Classification - DualNet CX (99% accuracy)](https://www.kaggle.com/code/lokeshbhaskarnr/audio-classification-dualnet-cx-accuracy-99)
- [Audio Classification - NASNet (95% accuracy)](https://www.kaggle.com/code/lokeshbhaskarnr/audio-classification-nasnet-95)
- [Audio Classification - DualNet Xpert (99% accuracy)](https://www.kaggle.com/code/lokeshbhaskarnr/audio-classification-dualnet-xpert-99)
- [Audio Dataset Visualization](https://www.kaggle.com/code/lokeshbhaskarnr/audio-dataset-visualization)

---

## 🛠️ Setup Instructions

### 1️⃣ Install Required Dependencies

Ensure you have Python **3.8 or later** installed.

Run the following command to install all required Python libraries:

```sh
pip install -r requirements.txt
```

### 2️⃣ Install FFmpeg (Required for `pydub` and `librosa`)

#### **Windows:**
1. Download FFmpeg from: [https://ffmpeg.org/download.html](https://ffmpeg.org/download.html)
2. Extract it to a directory (e.g., `C:\ffmpeg`).
3. Add the `bin` folder to your system `PATH`:
   - Search for **"Edit the system environment variables"** in Windows.
   - Under **System Properties > Advanced > Environment Variables**, find `Path` and edit it.
   - Click **New** and add:  
     ```
     C:\ffmpeg\bin
     ```
   - Click **OK** and restart your system.

#### **Mac/Linux (Using Homebrew):**
```sh
brew install ffmpeg
```

#### **Ubuntu/Debian (Using APT):**
```sh
sudo apt update && sudo apt install ffmpeg -y
```

---

## ⚠️ Streamlit Limitations

- **Streamlit does not support FFmpeg and sounddevice** in the cloud environment.
- To enable **audio recording**, run the app **locally**.
- Use `app_local_record.py` instead of `app.py` for full recording features.

---

## 🔧 Running the App Locally

To start the app, run:

```sh
streamlit run app.py
```

If you want **local audio recording support**, run:

```sh
streamlit run app_local_record.py
```

---

## 📜 License

This project is **open-source** and available under the [Apache](https://github.com/LokeshBhaskarNR/Generic-Audio-Classifier/blob/main/LICENSE) License.

## Acknowledgements

Special thanks to the content creators who made their recordings available. The Vehicle sounds dataset was sourced from Kaggle user Jan Boubia Abderrahim.

### Source Videos

<details>
<summary>Animals</summary>

#### Cats
- "Cats meowing and purring compilation" https://youtu.be/-QWfApp3j9k?si=huzWy4ziB_Z_OhRh
- "Cute Kittens Meowing Compilation" https://youtu.be/ue_jitOw8Yw?si=BUsxpEbRMTXwE8sd
- "Kitten Meowing Sounds" https://youtu.be/_moIdma4a-Q?si=KQOIiq3w20fKR-M1
- "Cat Purring Sound Effect" https://youtu.be/pjQc9RdQo9g?si=3-B0aChhvxCTx9OC
- "Cat Meow Sound Effect" https://youtu.be/nWSNd3yXVZM?si=JM7f7wUoZuxV-sAC
- "Cute Cat Meowing" https://youtu.be/UNV0C5_A1jQ?si=yeUWlq35jRzDJiJp

#### Dogs
- "Dog Barking Sound Effects" https://youtu.be/9KScQy6sQUQ?si=W6EHNrr2AyYynKeF
- "Dogs Barking Sound Effect" https://youtu.be/dTtMcTd2SJ8?si=2TE0ltFx5TBzDN_G
- "Dog Bark Sound Effect" https://youtu.be/2fwJKCXApJU?si=TYNwwOy7Sxd4c3lZ
- "Dog Barking Sound" https://youtu.be/GuP3bN__gSw?si=1B4Ml2PWmAfTwi5C
- "Dog Bark Sound" https://youtu.be/qM98alxa6q8?si=0VhAt64SNEa4hHrD
- "Dog Barking Sounds" https://youtu.be/QksvKXITvCI?si=c-4qYDyjPKs_DxoG
- "Dogs Barking" https://youtu.be/17Hv93k22Eo?si=DDROzVxEei-oHuit
- "Dog Barking Sound Effect" https://youtu.be/I1rwbIEp-hk?si=BaYRf1bgODoNQZa0
- "Puppy Barking Sound" https://youtu.be/01le4Ln8da0?si=Tu6SxtnTHORrA67F
- "Cute Puppy Barking" https://youtu.be/zym-CrCCkVY?si=Y7zPiz02zDsn0L4o

#### Elephants
- "Elephant Sounds" https://youtu.be/cnrnkMlw3UM?si=1AmGjM6YUE1QGrfm
- "Elephant Trumpeting" https://youtu.be/Hjx-S-6U9k0?si=NUmDtWfo9G4AD2jm
- "Elephant Sound Effects" https://youtu.be/cUSFs01vkvI?si=E5McUDD-tTCoQZuk
- "Elephant Trumpeting Sound" https://youtu.be/oZWMVlabjjQ?si=Hx-6NOkS3RJK9n3s
- "Baby Elephant Sounds" https://youtu.be/LpZgVYJmJtA?si=9WPcIQeu8mBbIQz3
- "Elephant Calling" https://youtu.be/fbnKTlajlcI?si=GDTU3wVzkT_8Ma6X
- "Elephant Vocalization" https://youtu.be/Z-JT1Bv0vqg?si=raLdTFbN8N5XDzKL
- "Elephant Trumpet" https://youtu.be/al-j_CXq_Gs?si=blXrs62QSx8Z5KiF

#### Horses
- "Horse Neigh Sound Effect" https://youtu.be/qHuLE1A8QXQ?si=9pgwsbfk5PHn6u2Y
- "Horse Galloping Sound" https://youtu.be/C7y518v3EpE?si=NpVfyeBxGKK41jA3
- "Horse Whinnying" https://youtu.be/avcRRcIRnbY?si=rh6QJo4nK5XTf5yu
- "Horse Sounds" https://youtu.be/tO1r5Rher28?si=UxmmQaQDn2RlJMea
- "Horse Neighing" https://youtu.be/3tFYegxsL50?si=pcbLiik9gFPNEtiC
- "Horse Sound Effects" https://youtu.be/dnIToKkZagw?si=VCj2PxasfhmLG7Ep
- "Horse Whinny" https://youtu.be/rZibO2dNods?si=GpoctW6y7s-kwqla

#### Lions
- "Lion Roar Sound Effect" https://youtu.be/rvJTyz3HB7E?si=gU8oHuEHsFqHhWaf
- "Lion Roaring Sounds" https://youtu.be/uFcZhH_wFbs?si=kQjUYOQ98REuuJFK
- "Lion Sound Effects" https://youtu.be/I0uqJiuZGcI?si=-Y7EBieugsEK9ZoY
- "Lion Roars" https://youtu.be/uNA92B4alzs?si=TmzSbrSG_SJ05NaV
- "Big Cat Roaring" https://youtu.be/h_m8EqsraXs?si=gHian_S1i9LiMG6p
- "Lion Pride Sounds" https://youtu.be/PLgTgSiygFc?si=G2GUZZdIykgget4v
- "Lion Growling" https://youtu.be/TzTyb07JvxM?si=o8iF8nEsoYD2hlIO
- "Lion Vocalizations" https://youtu.be/DBYpJHqAR3E?si=-ao1WLUX-6Ib1Kdh
</details>

<details>
<summary>Birds</summary>

#### Crows
- "Crow Cawing Sound Effect" https://youtu.be/T8xQ-y2pfVo?si=aUvr_v8SgEhXtCEQ
- "Crow Sounds" https://youtu.be/s1gxWM_E_D8?si=eKCUhO894EQzYqZx
- "Crows Cawing" https://youtu.be/ujqJiFjbsOU?si=7VcTpDz3MTwnPLpM
- "Crow Calls" https://youtu.be/WoRbb5zaThM?si=n-mMzIr3tNf5_tfn

#### Parrots
- "Parrot Talking and Squawking" https://youtu.be/dBPu0MKa_vg?si=aairO4USAf-I2jQB
- "Parrot Sounds" https://youtu.be/o74WN6HCocY?si=KkGM2xxU5eNgXwWR
- "Parrot Vocalizations" https://youtu.be/6yoEvmlmQM0?si=zsZ1cIY7w1xSDosL
- "Talking Parrot" https://youtu.be/dBPu0MKa_vg?si=2le9yhelwK3rjHM-
- "Parrot Squawking" https://youtu.be/aj3ny_GTuhM?si=NyEofcmo-_vHTNYz
- "Parrot Sound Effects" https://youtu.be/B9dUpGFc5Uc?si=lablnWLyizKbxvdS
- "Parrot Calls" https://youtu.be/BHOUyvC-guc?si=SYsLhUR1IL3kylZm
- "Parrot Noises" https://youtu.be/oDPwVz55zGg?si=wL1MVAC45tVzvE2n

#### Peacocks
- "Peacock Calling Sound" https://youtu.be/MiF7v-gYXLE?si=h2CZiZzPqkjd2_Gs
- "Peacock Sound Effect" https://youtu.be/walgy_1QQmY?si=Uz2GiEbNNronMgWo
- "Peacock Mating Call" https://youtu.be/UgDw2iIcmQ0?si=Z2tT2cA9t_z604-p
- "Peacock Sounds" https://youtu.be/AnImnX0DRNQ?si=fBet-NSx5a_RtCQP
- "Peacock Screaming" https://youtu.be/LDoN7_Z5O-M?si=60J7k9DxIO0GIiYE
- "Peacock Calls" https://youtu.be/xP8xK0ke7SE?si=7ucdGXaMOIrelb_5

#### Sparrows
- "Sparrow Chirping Sound" https://youtu.be/h9AoB2JSoCg?si=8f2zJSu-z7lIynsC
- "Sparrow Song" https://youtu.be/8MM6uX71ovU?si=7ftGmxPObHnRVfA0
- "Sparrow Sounds" https://youtu.be/X3C_hpTxRd0?si=uhUv0exPSfN2fTIt
- "Sparrow Calling" https://youtu.be/hLbVDJI80b0?si=XGepbsltxrafGJGk
- "Sparrow Chirps" https://youtu.be/fKAhbrkiAPo?si=c2KBHUPtnsEIdqf4
</details>

<details>
<summary>Environment</summary>

#### Crowd
- "Crowd Noise Sound Effect" https://youtu.be/3jYUp9LhiQ8?si=V2H-nFJzK7YANtna
- "Crowd Ambient Sound" https://youtu.be/FnhJ2wARY4Q?si=9wbo9A55D4Wz0A1q
- "Crowd Sounds" https://youtu.be/1Jh6SuKALt4?si=HtgDCfnjmtvKDuAb
- "Stadium Crowd" https://youtu.be/88UwejHolJ8?si=Q2uYVAG1MLnnCmZW
- "Crowd Chattering" https://youtu.be/a0Ud85Xdxn4?si=WBtOqweocqcECCqx
- "Crowd Ambience" https://youtu.be/4h7tXm5b5KM?si=i8aQWenyivQon6Ji
- "People Crowd" https://youtu.be/IKB3Qiglyro?si=ohSXY5BjyHXrgx0T

#### Military
- "Military Sound Effects" https://youtu.be/qFxR1yvsvqQ?si=oUJV9LIx7xjBGBgJ
- "Military Vehicles Sounds" https://youtu.be/RGtN2GIM-ig?si=GtS5SjSbeYFpun7H
- "Military Operations Audio" https://youtu.be/0QmA_-uxaDE?si=-BrrVzVT49EVRhi-

#### Office
- "Office Ambience Sounds" https://youtu.be/D7ZZp8XuUTE?si=QSlXeMzoZG3jziZQ

#### Rainfall
- "Rain Sound Effect" https://youtu.be/y615vOsiG5w?si=nR1IS-o9KWddVN6x
- "Heavy Rain Sounds" https://youtu.be/GA1D88HF0xE?si=ztSu-_-KUfUWSaLS
- "Rainfall Audio" https://youtu.be/gP9sGBywjks?si=LpyUGSV6cvEsBq2S

#### Traffic
- "Traffic Sound Effect" https://youtu.be/yrIRAd7E6qE?si=ykIq3lnR00Kdd7oQ
- "City Traffic Sounds" https://youtu.be/GlCazmVBUMg?si=CKsUPjqG_m3rjudp
- "Urban Traffic Noise" https://youtu.be/ET8SLcviq7s?si=CdGTfbfyJZ4WMa6M
- "Highway Traffic" https://youtu.be/FdOrPosxbFU?si=zdcgBiY8vJOsuMHK
- "Busy Street Sounds" https://youtu.be/L9Jl_AxzohQ?si=ChRu86EewVuwVWj6
- "Traffic Ambience" https://youtu.be/zxxfvP8-lrU?si=xwPBKlaEJRKyZzDA
- "Road Traffic" https://youtu.be/LqL_C29sGCY?si=Jo-wWII4J52bnh2R
- "Traffic Jam Sounds" https://youtu.be/iJZcjZD0fw0?si=4-sXfx4b-Cd4s1IH
- "Intersection Traffic" https://youtu.be/9wbA9FVtWF4?si=31mA4JNLenbotBjJ

#### Wind
- "Wind Sound Effect" https://youtu.be/v2Zh0oGmmvo?si=yrsD8rInl9mxbanc
- "Strong Wind Sounds" https://youtu.be/v2Zh0oGmmvo?si=ssj46IpptaqN6zP5
</details>

<details>
<summary>Vehicles</summary>

All vehicle sounds were sourced from the "Vehicle Sounds Dataset" https://www.kaggle.com/datasets/janboubiabderrahim/vehicle-sounds-dataset by Jan Boubia Abderrahim on Kaggle.
</details>
