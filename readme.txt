# Astrobit

Astrobit is an AI-based pipeline for detecting Earth-like exoplanet transit signals in 
Kepler photometric data. To use the project, first download or clone the repository from GitHub and make sure the provided `.parquet`
files are placed inside the `test` folder. The Parquet files are read directly by the program, so no conversion is required.
After downloading, install the required dependencies using `pip install -r requirements.txt`. Run the pipeline in order by first executing 
`python BLS.py`, which reads and processes the Kepler Parquet files and performs Box Least Squares transit detection to generate `astrobit_intial.csv`. 
Next, run `python predictor.py`, which loads the trained Random Forest model from `astrobit_rf.pkl`, evaluates the detected candidates, and generates confidence scores
in `astrobit_secondary.csv`. Finally, run `python merger.py`, which calculates the planet-to-star radius ratio and estimated number of transits and produces the final `
astrobit_submission.csv` file. The complete pipeline can therefore be executed with `python BLS.py`, followed by `python predictor.py`, and then `python merger.py`. 
