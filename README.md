# ML-Birdsong

Bengalese finches are songbirds used as a model organism for human language learning. They sing distinct syllables that we can use to measure the acquisition of vocal motor skills. However, a significant challenge lies in manually labeling the syllables in the birds' songs. This project aims to test various machine learning classification models to optimize syllable identification and reduce human bias inherent to manual labeling.

# Instructions

- Use feat_extraction_tutorial.ipynb to walk through the feature extraction process
  - To see how acoustic features (Average Weiner Entropy, Spectral Entropy, Average Gravity Center, Average Spectral Width, Frequency of highest magnitude, Average MFCCs, Duration) were extracted, look at function extract_from_waveform() in final_ml.py 
- use Orange Data Mining for testing different ML models
