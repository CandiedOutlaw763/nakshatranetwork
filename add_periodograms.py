import pickle
import numpy as np
from pathlib import Path
from astropy.timeseries import BoxLeastSquares
import astropy.units as u
import config

print("Loading classifications.pkl...")
results_path = config.RESULTS_DIR / "classifications.pkl"
with open(results_path, "rb") as f:
    results = pickle.load(f)

print(f"Adding periodograms to {len(results)} candidates...")
for i, result in enumerate(results):
    filepath = result.get("filepath")
    if not filepath or not Path(filepath).exists():
        continue
    
    with open(filepath, "rb") as f:
        data = pickle.load(f)
    
    t = data.get("time", np.array([]))
    f = data.get("flux", np.array([]))
    
    if len(t) < 50:
        continue
        
    model = BoxLeastSquares(t * u.day, f)
    periods = np.linspace(config.PERIOD_MIN, config.PERIOD_MAX, config.PERIOD_GRID_SIZE) * u.day
    durations = np.linspace(config.DURATION_MIN, config.DURATION_MAX, config.N_DURATIONS) * u.day
    
    bls_results = model.power(periods, durations)
    result["_periods"] = np.asarray(bls_results.period.value)
    result["_power"] = np.asarray(bls_results.power)
    
    if i % 10 == 0:
        print(f"Processed {i}/{len(results)}...")

print("Saving updated classifications.pkl...")
with open(results_path, "wb") as f:
    pickle.dump(results, f)

print("Done! Periodograms added. The file is now larger but contains all plotting data.")
