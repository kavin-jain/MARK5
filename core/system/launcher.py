"""
MARK5 SYSTEM LAUNCHER v8.0 - PRODUCTION GRADE
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

CHANGELOG:
- [2026-02-06] v8.0: Standardized header, production certification

TRADING ROLE: Main System Entry Point & Orchestrator
SAFETY LEVEL: CRITICAL - Bootstraps entire HFT topology

FEATURES:
✅ CPU Affinity Assignment (Core Isolation)
✅ Microservice Bootstrap (Data, Features, Inference, Exec)
✅ Graceful Shutdown Handling
"""

import sys
import time
import logging
import signal
import os
from pathlib import Path

# ── Load .env FIRST — before any module reads os.getenv() ──────────────────
try:
    from dotenv import load_dotenv
    _env_path = Path(__file__).resolve().parents[2] / ".env"
    load_dotenv(dotenv_path=_env_path, override=True)
    logging.getLogger("MARK5.Boot").info(f"✅ .env loaded from {_env_path}")
except ImportError:
    pass  # python-dotenv not installed; rely on shell exports

# Add project root to sys.path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from core.system.bootstrap import bootstrap_system
from core.system.container import container
from core.system.process_manager import get_process_manager, DataIngestionWorker, FeatureEngineeringWorker, InferenceWorker, ExecutionWorker

# Configure HFT Topology (Map processes to CPU Cores)
# Core 0: OS / UI / Misc
# Core 1: Data Ingestion (Network I/O Heavy)
# Core 2: Feature Engineering (CPU Heavy)
# Core 3: Inference (CPU/GPU Heavy)
TOPOLOGY = {
    "DataIngestion": {"cls": DataIngestionWorker, "core": 1},
    "FeatureEngine": {"cls": FeatureEngineeringWorker, "core": 2},
    "Inference":     {"cls": InferenceWorker, "core": 3},
    "Execution":     {"cls": ExecutionWorker, "core": 4},
}

class MARK5Launcher:
    def __init__(self):
        self.logger = None
        self.pm = None

    def launch(self):
        # 1. Bootstrap Core Services (Config, Redis, DB)
        # This runs in the Main Process
        bootstrap_system()
        self.logger = logging.getLogger("MARK5.Launcher")
        self.logger.info("✅ Core Services Bootstrapped")

        # 2. Initialize Process Manager
        self.pm = get_process_manager()
        
        # 3. Register HFT Topology
        self.logger.info(f"⚙️ Configuring HFT Topology on {len(TOPOLOGY)} Cores...")
        for name, spec in TOPOLOGY.items():
            self.pm.register_worker(name, spec['cls'], core_id=spec['core'])

        # 4. Ignite Processes
        self.pm.start_all()
        
        # 5. Launch UI (Main Thread) or Idle
        # If running in Docker/Headless, we just keep main thread alive
        try:
            self._main_loop()
        except KeyboardInterrupt:
            self.shutdown()

    def _main_loop(self):
        # Check if we should launch Terminal UI
        if container.config.ui.enabled:
            # Lazy import to avoid circular deps or UI libs in headless
            try:
                from core.ui.terminal import MARK5Terminal
                ui = MARK5Terminal(container)
                ui.run() # Blocking UI Loop
            except ImportError:
                self.logger.warning("UI Module not found, falling back to headless.")
                self._headless_loop()
        else:
            self._headless_loop()

    def _headless_loop(self):
        # Headless Keep-Alive
        self.logger.info("🖥️ Headless Mode Active. System Running...")
        while True:
            time.sleep(1)

    def shutdown(self):
        self.logger.info("\n🛑 System Shutdown Request Received.")
        if self.pm:
            self.pm.stop_all()
        sys.exit(0)

if __name__ == "__main__":
    launcher = MARK5Launcher()
    launcher.launch()
