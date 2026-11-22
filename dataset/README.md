# Dataset

WM-811K (LSWMD) wafer maps, Wu et al. (2015). Not in git (2 GB).

Download `LSWMD.pkl` from Kaggle (`qingyi/wm811k-wafer-map`, login needed) or http://mirlab.org/dataset/public/
and place it here as `dataset/LSWMD.pkl`. `src/phase2_data.py` reads it and writes the preprocessed arrays to `data/`.
