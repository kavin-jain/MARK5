
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

def slow_frac_diff_ffd(series, d, thres=1e-4):
    """Non-vectorized version for comparison."""
    w = get_w(d, thres)
    w_rev = w[::-1]
    s = series.dropna().values
    out = np.full(len(s), np.nan)
    
    for i in range(len(w), len(s) + 1):
        out[i-1] = np.dot(w_rev, s[i-len(w):i])
        
    return pd.Series(out, index=series.dropna().index)

def test_frac_diff():
    fe = AdvancedFeatureEngine()
    d = 0.4
    w = get_w(d)
    print(f"w length: {len(w)}")
    
    # Generate dummy data
    n = len(w) + 10
    data = pd.Series(np.arange(n, dtype=float))
    
    print(f"Testing frac_diff_ffd with {n} points, d={d}")
    
    vec_result = fe._frac_diff_ffd(data, d)
    slow_result = slow_frac_diff_ffd(data, d)
    
    # Verify correctness
    vec_vals = vec_result.dropna().values
    slow_vals = slow_result.dropna().values
    
    print(f"Vec non-NaN count: {len(vec_vals)}")
    print(f"Slow non-NaN count: {len(slow_vals)}")
    
    if len(vec_vals) > 0:
        print(f"First vec val: {vec_vals[0]}")
        print(f"First slow val: {slow_vals[0]}")
    
    try:
        np.testing.assert_allclose(vec_vals, slow_vals, rtol=1e-5, atol=1e-8)
        print("✅ Correctness verified.")
    except Exception as e:
        print(f"❌ Correctness FAILED: {e}")

if __name__ == "__main__":
    test_frac_diff()
