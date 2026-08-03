import numpy as np
import pandas as pd
from scipy.stats import qmc

# 1. Input your exact base weights
upper_base = np.array([0.30174562, 0.21011278, 0.17247788, 0.31289289, 0.1346193,  0.12938747])
lower_base = np.array( [-0.15029182, -0.16098318, -0.20377234, -0.13863304, -0.13868888,  0.04034309])

# Combine them into a single 12-number "baseline" vector
w_base = np.concatenate([upper_base, lower_base])
num_weights = len(w_base) # Should be 12

# 2. Setup the Latin Hypercube Sampler
num_samples = 65 # How many airfoils do you want to generate?
sampler = qmc.LatinHypercube(d=num_weights, seed=42) # seed ensures reproducibility
sample_raw = sampler.random(n=num_samples)

# 3. Scale the LHS raw output (which is 0 to 1) to your +/- 15% range
# -0.15 + (value * 0.30) shifts the range perfectly to [-0.15, +0.15]
delta_w = -0.15 + (sample_raw * 0.30)

# 4. Apply the relative perturbation formula: w_new = w_base * (1 + delta_w)
# (numpy broadcasts this multiplication cleanly across all 500 rows)
w_dataset = w_base * (1 + delta_w)

# 5. Package it into a clean Pandas DataFrame
# Create helpful column names
col_names = [f'w_u{i+1}' for i in range(6)] + [f'w_l{i+1}' for i in range(6)]

df_airfoils = pd.DataFrame(w_dataset, columns=col_names)

print(f"Successfully generated {len(df_airfoils)} airfoils.")
print("\nFirst 3 generated airfoils (rows are samples, columns are weights):")
print(df_airfoils.head(3))

# Optional: Save it immediately to a CSV so you don't lose it!
df_airfoils.to_csv("C:/Users/kenne/Downloads/my_lhs_airfoils.csv", index=False)