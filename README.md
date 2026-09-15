# Astrobit — AI-Based Detection of Earth-Like Exoplanets in Kepler Data

# Astrobit — AI-Based Detection of Earth-Like Exoplanets in Kepler Data

## Overview

Astrobit is an end-to-end exoplanet detection pipeline developed to identify transit signals in Kepler space telescope photometric data, with a particular focus on detecting weak, shallow, and long-period signals that may correspond to Earth-like exoplanets. The system takes raw Kepler stellar light-curve data as input and processes it through data cleaning, normalization, detrending, transit detection, candidate refinement, machine-learning classification, and final candidate generation.

The fundamental principle behind the project is the transit method of exoplanet detection. When a planet passes in front of its host star along our line of sight, it blocks a small amount of the star's light, producing a small decrease in the observed stellar flux. If this decrease occurs periodically, it can indicate the presence of an orbiting planet. Astrobit is designed to automatically detect these periodic brightness variations from large volumes of Kepler photometric observations.

---

## Problem Statement

Detecting exoplanets from stellar light curves is challenging because planetary transit signals can be extremely weak compared with stellar variability, instrumental effects, and observational noise. This becomes particularly difficult for Earth-like planets because their transit depths can be very small and their orbital periods can be hundreds of days. Long-period planets also produce only a small number of observable transits during the available observation baseline.

The objective of Astrobit is therefore to build an automated pipeline that can:

1. Process raw Kepler photometric data.
2. Remove unwanted trends and instrumental variations.
3. Search for periodic transit-like signals.
4. Estimate the physical properties of detected candidates.
5. Use machine learning to classify and rank the detected candidates.
6. Assign a detection confidence to each candidate.
7. Validate detections against known ground-truth transit signals.
8. Produce a final candidate catalogue suitable for further analysis.

---

## Dataset

The project uses Kepler photometric observations containing 445 stars divided into training, development, and private evaluation datasets.

- Training set: 269 stars
- Development set: 89 stars
- Private evaluation set: 87 stars

Each star contains approximately 65,000 photometric observations covering several years of Kepler observations with a cadence of approximately 29.4 minutes.

The raw data contains quantities such as:

- Observation time
- Stellar flux
- Kepler quality flags
- Observation quarter

The training and development datasets also provide ground-truth information for injected transit signals, including orbital period, transit epoch, transit depth, transit duration, planet-to-star radius ratio, and number of transits.

---

# Pipeline

The complete Astrobit pipeline follows:

```text
Raw Kepler Photometry
        ↓
Quality Filtering
        ↓
Quarter-wise Normalization
        ↓
1-Day Median Detrending
        ↓
Flattened Light Curve
        ↓
Coarse BLS Period Search
        ↓
Promising Period Selection
        ↓
Fine BLS Search
        ↓
Harmonic / Alias Analysis
        ↓
Transit Candidate Extraction
        ↓
Random Forest ML Classification
        ↓
Candidate Confidence
        ↓
Candidate Ranking
        ↓
Physical Parameter Calculation
        ↓
Final Candidate Catalogue

Astrobit is an end-to-end exoplanet detection pipeline designed to identify transit-like signals in Kepler space telescope photometry, with a particular focus on detecting shallow and long-period signals that may correspond to Earth-like exoplanets. The project processes raw Kepler stellar light curves and automatically searches for periodic decreases in stellar brightness caused by planets passing in front of their host stars. The dataset contains observations of 445 stars, divided into training, development, and private evaluation sets, with each star containing several years of high-cadence photometric observations.

The pipeline begins with raw Kepler SAP flux data, which contains instrumental effects, stellar variability, noise, quarter-to-quarter flux differences, and other long-term trends that can hide weak transit signals. The preprocessing stage removes invalid observations using Kepler quality flags, normalizes the flux independently for each observing quarter, and applies a one-day running-median detrending procedure to remove large-scale variations while preserving short transit-like features. The resulting flattened light curve is normalized around a common baseline, making small periodic decreases in brightness easier to detect.

After preprocessing, Astrobit uses the Box Least Squares (BLS) algorithm to search for periodic box-shaped decreases in stellar brightness. Because searching the complete long-period range at extremely high resolution is computationally expensive, the detector uses a coarse-to-fine search strategy. A broad coarse period grid is first evaluated to identify promising regions of the BLS periodogram, after which the strongest candidate regions are searched at higher resolution. The search covers periods extending into the hundreds of days and evaluates multiple transit durations, allowing the system to detect both relatively short-period candidates and longer-period planetary signals.

For each detected candidate, Astrobit extracts the estimated orbital period, transit depth, transit duration, transit epoch, and Signal Detection Efficiency (SDE). Since BLS can produce harmonics or aliases of the true orbital period, candidate periods are also examined in relation to common harmonic relationships such as half, double, and other allowed multiples of the detected period. This is particularly important for long-period planets because only a small number of transits may be visible within the available observation baseline, making period recovery more difficult and increasing the possibility of incorrect but statistically strong BLS solutions.

The generated BLS candidates are then passed to a machine-learning stage for additional candidate ranking. Astrobit currently uses a Random Forest classifier trained using four BLS-derived features: orbital period, transit depth, transit duration, and SDE. The model produces a confidence score that can be used to rank detected candidates rather than relying only on a binary planet/no-planet decision. The training target combines known catalog positives and injected transit signals, allowing the model to learn from both existing positive examples and synthetic ground-truth injections.

A major part of Astrobit is its ground-truth validation framework. The training and development datasets provide information about injected transit signals, including their true orbital period, transit epoch, depth, duration, planet-to-star radius ratio, and number of transits. The detected candidates can therefore be directly compared with the known truth to evaluate period recovery, detection recall, depth accuracy, duration accuracy, false positives, and false negatives. Period recovery is evaluated using the challenge's required tolerance and allowed harmonic relationships. This validation process is used to identify weaknesses in both the signal-processing and machine-learning stages instead of relying only on visual inspection of light curves.

The detected transit depth can also be converted into an estimated planet-to-star radius ratio using the transit-depth relationship, approximately \(R_p/R_s = \sqrt{\mathrm{depth}}\), with the depth converted from parts per million to fractional flux before calculation. The number of observable transits can be estimated from the observation baseline and detected orbital period while accounting for the actual observation coverage where possible. These derived quantities provide additional physical information about each candidate and are included in the final candidate representation.

The final Astrobit output is designed as a clean candidate catalogue containing the candidate identifier, detection confidence, orbital period, transit depth, transit duration, planet-to-star radius ratio, and estimated number of observed transits. This provides a compact representation of the detected planetary candidates that can be used for evaluation, further astrophysical vetting, or downstream analysis.

Astrobit is currently an end-to-end working pipeline covering raw-data processing, light-curve cleaning, BLS-based transit detection, candidate refinement, harmonic analysis, machine-learning-based candidate ranking, ground-truth validation, and final candidate generation. The main remaining challenge is improving the recovery of weak, shallow, and long-period transit signals. In particular, long-period Earth-like candidates are difficult because only a few transit events may be present in several years of observations, while stellar variability and noise can produce competing BLS peaks. Future improvements can therefore focus on stronger candidate generation, improved transit vetting, multi-transit consistency checks, quarter-to-quarter consistency, odd-even transit analysis, secondary-eclipse checks, and better probability calibration of the machine-learning stage.
