import time
import numpy as np
from typing import Any, Union
import orjson

# HFT Requirement: Standard json.dumps is too slow. 
# orjson is 6x-10x faster and handles numpy arrays natively.
def fast_dumps(data: Any) -> str:
    """
    Serializes data to bytes using Rust-based orjson.
    Automatically handles numpy arrays which standard json chokes on.
    """
    return orjson.dumps(
        data, 
        option=orjson.OPT_SERIALIZE_NUMPY | orjson.OPT_NON_STR_KEYS
    ).decode('utf-8')

def current_time_ns() -> int:
    """
    Returns monotonic time in nanoseconds. 
    Standard time.time() is susceptible to system clock adjustments (NTP drift),
    which can destroy tick-to-trade logic.
    """
    return time.time_ns()

def pack_features(features: np.ndarray) -> bytes:
    """
    Zero-Copy serialization for gRPC.
    Instead of sending a list of floats (repeated field), we send raw bytes.
    """
    return features.tobytes()

def unpack_features(data: bytes, dtype=np.float32) -> np.ndarray:
    """
    Zero-Copy deserialization.
    """
    return np.frombuffer(data, dtype=dtype)
