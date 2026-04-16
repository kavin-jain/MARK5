
import numpy as np
import pandas as pd
import time
from core.models.features import AdvancedFeatureEngine

def get_w(d, thres=1e-4):
    w = [1.0]
    k = 1
    while True:
        w_k = -w[-1] / k * (d - k + 1)
        if abs(w_k) < thres:
            break
        w.append(w_k)
        k += 1
    return w

def test_frac_diff():
    fe = AdvancedFeatureEngine()
    d = 0.4
    w = get_w(d)
    print(f"w length: {len(w)}")
    print(f"w[:5]: {w[:5]}")
    print(f"w[-5:]: {w[-5:]}")
    
    # Generate dummy data
    n = len(w) + 2
    data = pd.Series(np.arange(n, dtype=float))
    
    print(f"Testing frac_diff_ffd with {n} points, d={d}")
    
    vec_result = fe._frac_diff_ffd(data, d)
    
    # Manual calculation for the first valid point (index len(w)-1)
    s = data.values
    L = len(w)
    manual_val = 0
    for k in range(L):
        manual_val += w[k] * s[L - 1 - k]
    
    print(f"Manual calculation (index {L-1}): {manual_val}")
    print(f"Vectorized result (index {L-1}): {vec_result.iloc[L-1]}")
    
    if np.isclose(manual_val, vec_result.iloc[L-1]):
        print("✅ Vectorized matches Manual.")
    else:
        print("❌ Vectorized DOES NOT match Manual.")

if __name__ == "__main__":
    test_frac_diff()
