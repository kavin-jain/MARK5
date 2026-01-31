import sys
import os
import traceback

# Add project root
sys.path.insert(0, os.getcwd())

print("--- Starting Debug Setup ---")

try:
    print("1. Loading Config...")
    from core.utils.config_manager import get_config
    config = get_config()
    print("   Config Loaded.")
except Exception:
    traceback.print_exc()
    sys.exit(1)

try:
    print("2. Initializing DataBridge...")
    import numpy as np # Ensure numpy is available for IPC
    from core.ui.dashboard import DataBridge
    
    # We call connect() implicitly in __init__
    bridge = DataBridge()
    print(f"   DataBridge Initialized. Connected: {bridge.connected}")
    if not bridge.connected:
        print("   (Note: NOT connected is expected if launcher is not running, but it should not CRASH)")
except Exception:
    traceback.print_exc()
    sys.exit(1)

try:
    print("3. Initializing DataProvider...")
    from core.data.provider import DataProvider
    provider = DataProvider(config)
    print("   DataProvider Initialized.")
except Exception:
    traceback.print_exc()
    sys.exit(1)

try:
    print("4. Checking ProcessManager Syntax...")
    from core.system.process_manager import get_process_manager, DataIngestionWorker, InferenceWorker, ExecutionWorker
    print("   ProcessManager Syntax Check Passed.")
except Exception:
    traceback.print_exc()
    sys.exit(1)

print("--- Setup Successful ---")
