# Known Challenges & Iterative Fixes

*This document outlines the real-world data science challenges we encountered during the pipeline development and how we solved them. This is an excellent narrative to include in the final hackathon presentation.*

## Challenge 1: The "Sim2Real" Domain Gap (Kepler-10b Failure)
After successfully training our Dual-Branch 1D-CNN to 90% validation accuracy on synthetic data, we tested it against real NASA data for the exoplanet **Kepler-10b**. 

**The Result:** The AI failed to detect the planet (Outputted 17.56% confidence).

**The Diagnosis (Why it failed):**
This is a classic "Simulation-to-Reality" domain gap. We discovered two major discrepancies between our synthetic training data and the real universe:

1.  **Planet Size Bias (Jupiters vs. Earths):** 
    *   In our `generate_dataset.py`, we generated synthetic planets with a radius ratio (`params.rp`) between `0.05` and `0.20`. This corresponds to massive "Gas Giant" planets that block out 1% to 4% of a star's light.
    *   **Kepler-10b** is famously the first *rocky, terrestrial* planet discovered by the mission. It is tiny and only blocks out **0.015%** of the star's light.
    *   *Conclusion:* Our AI had literally never seen a planetary dip that small. It correctly classified a 0.015% dip as random background noise because it was only trained to look for massive Jupiter-sized dips.
2.  **Batch Normalization Mismatch:**
    *   We added a `BatchNormalization` layer in the neural network to center the data. While this works beautifully during training on large batches, it relies on historical moving averages during inference (testing a single star). Because the real NASA data had a slightly different baseline flux distribution than our synthetic data, the scaling was mathematically shifted.

## The Solution (Next Steps)
To fix these issues and make our AI capable of detecting tiny rocky planets, we need to implement the following code updates:

*   **Fixing the Size Bias:** We will update `generate_dataset.py` to lower the minimum planet radius to `0.01`. This forces the AI to study extremely subtle, shallow transit dips during training.
*   **Fixing the Normalization:** We will remove the `BatchNormalization` layer from the CNN. Instead, we will implement manual Z-Score Standardization (Mean=0, Std=1) on the 201-bin arrays *before* they are fed into the neural network. This ensures that synthetic training data and real testing data are scaled identically.
