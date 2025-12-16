# NTU-CZ4042
Repo for NTU CZ4042 Neural Network and Deep Learning course

## Overview
This project predicts the genre of a music track by combining two modalities: engineered audio features and spectrogram images. It trains an MLP on tabular features, a CNN on spectrograms, a feautre level, and a logit‑level fusion model that integrates both to improve classification performance. The repo includes data preprocessing, exploratory data analysis, training scripts, evaluation scripts, and result visualisations.

## Tech Stack
- Language: Python
- PyTorch (torch, torch.nn, torch.optim, DataLoader)
- Torchvision (image transforms)
- Scikit‑learn (KFold, StandardScaler, LabelEncoder, accuracy metrics)
- Pillow (PIL) for spectrogram images
- Pandas, NumPy
- Jupyter Notebooks (EDA and plotting)

## Dataset
Free Music Archive (FMA) dataset is used for this project and focused on 5 root genres: Rock, Electronic, Hip‑Hop, Folk, and Pop. Tracks with ambiguous multi‑label annotations are removed, and remaining child genres are mapped to their root genres to obtain 18k+ clean, single‑label examples. Each track is represented by both aggregated Librosa‑style audio features and spectrogram images, enabling the multimodal fusion experiments in this repo.

## Contributors
Sanghyun Lee, 
Yumin Park, 
Junseo Park

## Detailed Information
For a more detailed description of the dataset, model architecture, experiments, and results, please refer to the accompanying project report (PDF) included in this repository.
