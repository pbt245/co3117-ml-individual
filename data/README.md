# Data: UCI Human Activity Recognition Using Smartphones

- **Source:** UCI Machine Learning Repository, dataset #240
  <https://archive.ics.uci.edu/dataset/240/human+activity+recognition+using+smartphones>
- **Citation:** D. Anguita, A. Ghio, L. Oneto, X. Parra, J. L. Reyes-Ortiz. *A Public Domain Dataset for Human
  Activity Recognition Using Smartphones.* ESANN 2013.
- **License:** CC BY 4.0 (as listed on the UCI page).

## Content
- 30 volunteers (19–48 years) wearing a Samsung Galaxy S II on the waist; accelerometer + gyroscope at 50 Hz.
- Signals are cut into 2.56 s windows with 50% overlap. Each window has **561 features** (time and frequency domain, already normalized to $[-1, 1]$).
- **6 classes:** WALKING, WALKING_UPSTAIRS, WALKING_DOWNSTAIRS, SITTING, STANDING, LAYING.
- **Official split by subject:** train has 7,352 windows from 21 subjects, test has 2,947 windows from 9 subjects.

## How it is used in this repo
- Raw files go in `data/raw/UCI HAR Dataset/` and are **not committed** (see `.gitignore`).
- `src/data.py` maps labels 1–6 to 0–5. It then holds out 20% of the *training subjects* as validation (a subject-wise split, which avoids leakage between windows of the same person) and z-scores all splits using training statistics only.

## Download
```bash
mkdir -p data/raw && cd data/raw
curl -L -o har.zip "https://archive.ics.uci.edu/static/public/240/human+activity+recognition+using+smartphones.zip"
unzip har.zip && unzip "UCI HAR Dataset.zip"
```
The UCI server can be very slow. The six files used (`{X,y,subject}_{train,test}.txt`) were checked
byte-for-byte against the official zip.
