# Conceptual Solution Proposal: AI-Based Exoplanet Detection Pipeline

This document outlines the proposed technical architecture for detecting exoplanet transit signals from noisy astronomical light curves, specifically designed to address the challenges outlined in the ISRO hackathon problem statement.

## 1. Problem Deconstruction & Objectives

The primary challenge is identifying minuscule dips in a star's brightness (transits) while actively rejecting false positives caused by:
*   **Intrinsic Noise:** Detector artifacts, cosmic ray hits.
*   **Stellar Variability:** Starspots causing out-of-transit brightness changes.
*   **Astrophysical False Positives:** Eclipsing binaries (stellar companions) which mimic transit dips.
*   **Stellar Blending:** Contamination from background/foreground sources in crowded fields.

**Objective:** Build an end-to-end AI pipeline that preprocesses raw light curve data, extracts relevant features, and classifies the signal using a Deep Learning model to isolate true planetary transits.

---

## 2. Proposed Architecture Pipeline

Our pipeline consists of three main stages: Data Preprocessing, the AI Inference Engine, and Post-Processing Validation.

### Phase 1: Data Preprocessing & Detrending
Raw light curves cannot be fed directly into a neural network due to low-frequency noise. We will implement a robust preprocessing pipeline:

1.  **Outlier Removal:** Apply Sigma-clipping to remove sudden spikes caused by cosmic rays or instrumental glitches.
2.  **Detrending (Removing Starspots):** Apply a median filter or a basis-spline (B-spline) fit to the out-of-transit data. This smooths out long-term stellar variability (like rotating starspots) while preserving the sharp, short-term transit dips.
3.  **Period Searching:** Run a Box-fitting Least Squares (BLS) algorithm to identify the most likely orbital period of the transit signal.
4.  **Phase-Folding:** Fold the time-series data over the identified period. This aligns all transit events on top of each other, amplifying the signal-to-noise ratio of the dip.

### Phase 2: AI Model Architecture (Dual-Branch 1D CNN)

To effectively classify the folded light curves and reject false positives, we propose a **Dual-Branch 1D Convolutional Neural Network (CNN)**, inspired by the highly successful AstroNet architecture.

Instead of passing the whole curve into one network, the data is split into two views:

1.  **The Global View Branch:** 
    *   **Input:** The entire phase-folded light curve binned into a fixed number of points (e.g., 2001 bins).
    *   **Purpose:** This branch allows the AI to see the "big picture". It is crucial for rejecting **Eclipsing Binaries**. It learns to detect secondary eclipses (a smaller dip when the secondary star goes behind the primary) or alternating transit depths, which are dead giveaways of a binary star system, not a planet.
2.  **The Local View Branch:**
    *   **Input:** A "zoomed-in" window focusing strictly on the primary transit event (e.g., 201 bins centered on the dip).
    *   **Purpose:** This branch allows the AI to analyze the exact shape of the dip. Planetary transits are typically "U-shaped" with flat bottoms, whereas grazing eclipsing binaries are often "V-shaped".

*The outputs of both 1D-CNN branches are flattened, concatenated together, and passed through fully connected dense layers to output a final probability score (0 to 1) of being an exoplanet.*

### Phase 3: Handling Stellar Blending & Crowded Fields
To address the specific challenge of crowded fields mentioned in the PS:
*   We will incorporate **Centroid Offset Analysis** into the post-processing phase. If the pixel-level data shows that the center of light shifts significantly during the transit dip, it strongly indicates that the dip is originating from a background blended eclipsing binary, not the target star.

---

## 3. Training Strategy

*   **Training Data:** We will utilize publicly available, labeled Kepler or TESS mission datasets containing confirmed planets, eclipsing binaries, and noise.
*   **Data Augmentation:** To make the model robust against extreme noise, we will synthetically inject planetary transit signals into highly noisy, empty light curves during training.
*   **Loss Function:** We will use **Focal Loss** instead of standard cross-entropy. In astronomical datasets, "noise" vastly outnumbers "planets." Focal loss forces the model to pay more attention to the hard-to-classify, rare planetary transit examples.

---

## 4. Verification Plan

We will evaluate the model using the following metrics:
*   **Recall (Sensitivity):** This is our most critical metric. We want to minimize False Negatives (missing a real planet).
*   **Precision:** To measure how many of our flagged "planets" were actually false positives.
*   **Confusion Matrix:** Specifically analyzing the misclassification rate between *Planets* and *Eclipsing Binaries*.
