import multiprocessing
import time
import logging
import os
import psutil
import threading
import signal
from typing import Dict, Type
from abc import ABC, abstractmethod

"""
MARK5 PROCESS KERNEL v8.0 - PRODUCTION GRADE
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

CHANGELOG:
- [2026-02-06] v8.0: Standardized header, production certification

TRADING ROLE: OS-Level Process Management & Optimization
SAFETY LEVEL: CRITICAL - Manages HFT Workers

FEATURES:
✅ Real-Time Scheduler Promotion (SCHED_FIFO)
✅ CPU Core Isolation & Pinning (Affinity)
✅ Lazarus Loop (Auto-Respawn of dead workers)
"""

logger = logging.getLogger("MARK5.Kernel")

def set_realtime_priority(pid):
    """
    Attempts to set SCHED_FIFO (Real-Time) priority.
    Only works on Linux with root/limits privileges.
    """
    try:
        # sched_setscheduler is only available on Unix
        if hasattr(os, 'sched_setscheduler'):
            param = os.sched_param(10) # Priority 1-99
            os.sched_setscheduler(pid, os.SCHED_FIFO, param)
            logger.info(f"⚡ PID {pid} promoted to SCHED_FIFO (Real-Time).")
        else:
            logger.warning(f"⚠️ SCHED_FIFO not supported on this OS.")
    except Exception as e:
        logger.warning(f"⚠️ Could not set SCHED_FIFO for PID {pid}: {e}. Ensure 'ulimit -r' is set or run as root.")

class BaseWorker(multiprocessing.Process, ABC):
    def __init__(self, name: str, core_id: int = None):
        super().__init__(name=name, daemon=True)
        self.stop_event = multiprocessing.Event()
        self.core_id = core_id 

    def run(self):
        self._optimize_os_priority()
        self._setup_logging()
        logger.info(f"🚀 {self.name} Online [PID: {os.getpid()}]")
        
        try:
            self._initialize_components()
            self._process_loop() # This loop must manage its own exit
        except Exception as e:
            logger.critical(f"💥 {self.name} CRASHED: {e}", exc_info=True)
        finally:
            logger.info(f"🛑 {self.name} Halted")

    def _setup_logging(self):
        logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        self.logger = logging.getLogger(f"MARK5.{self.name}")

    def _optimize_os_priority(self):
        try:
            p = psutil.Process(os.getpid())
            
            # 1. CPU Affinity (Pin to Core to avoid cache misses)
            if self.core_id is not None:
                try:
                    p.cpu_affinity([self.core_id])
                    logger.info(f"📌 {self.name} pinned to Core {self.core_id}")
                except Exception as e:
                    logger.warning(f"Could not pin to core {self.core_id}: {e}")

            # 2. Real-Time Scheduling
            set_realtime_priority(os.getpid())
                
        except Exception as e:
            logger.warning(f"OS Optimization Failed: {e}")

    def stop(self):
        self.stop_event.set()

    @abstractmethod
    def _initialize_components(self): pass

    @abstractmethod
    def _process_loop(self): pass

# --- OPTIMIZED WORKERS ---

class DataIngestionWorker(BaseWorker):
    def _initialize_components(self):
        from core.infrastructure.ipc import SharedMemoryRingBuffer
        from core.data.provider import DataProvider
        from core.utils.config_manager import get_config
        import numpy as np
        
        # 1. Setup IPC
        # Shape: [Timestamp(f64), TickerID(f64), Price(f64), Volume(f64)]
        self.shm = SharedMemoryRingBuffer(
            name="mark5_tick_buffer", 
            shape=(4,), 
            dtype=np.float64, 
            slots=100000, 
            create=True
        )
        logger.info("✅ Shared Memory Ring Buffer Created: mark5_tick_buffer")
        
        # 2. Setup Data Provider
        try:
            self.config = get_config()
            self.provider = DataProvider(self.config)
            
            # Try to Connect
            if self.provider.connect():
                self.use_real_data = True
                self.provider.register_tick_observer(self._on_tick)
                logger.info("✅ DataProvider Connected (Real Data Mode)")
                
                # Subscribe to Watchlist from Config (or defaults)
                # Assuming config has trading.watchlist, else fallback
                watchlist = getattr(self.config.trading, 'watchlist', ["COFORGE", "PERSISTENT", "KPITTECH", "HAL", "BEL"])
                if isinstance(watchlist, list):
                    self.provider.feed.subscribe(watchlist)
                    logger.info(f"Subscribed to: {watchlist}")
            else:
                self.use_real_data = False
                logger.warning("⚠️ DataProvider Connect Failed. Falling back to MOCK DATA.")
        except Exception as e:
            logger.error(f"DataProvider Init Failed: {e}. Falling back to MOCK.")
            self.use_real_data = False

        # Mock Data Setup
        if not self.use_real_data:
            self.tickers = [738561, 2953217, 408065, 341249, 1270529] # COFORGE, PERSISTENT, KPITTECH, HAL, BEL
            self.prices = {t: 2500.0 for t in self.tickers}

    def _on_tick(self, ticks):
        """Callback for Real Data"""
        import numpy as np
        import time
        # Ticks is list of TickData objects (or dicts depending on adapter)
        # We need to standardize. BaseFeed usually sends TickData objects.
        
        for tick in ticks:
            # TickData (from kite_adapter): has .token, .ltp, .volume, .timestamp
            # Dict (legacy/mock): has 'instrument_token', 'last_price', 'volume'
            if hasattr(tick, 'token'):          # TickData dataclass path
                tid   = tick.token
                price = tick.ltp
                vol   = tick.volume
                ts    = int(tick.timestamp.timestamp() * 1e9) if tick.timestamp else time.time_ns()
            elif isinstance(tick, dict):        # Dict fallback path
                tid   = tick.get('instrument_token', 0)
                price = tick.get('last_price', 0.0)
                vol   = tick.get('volume', 0)
                ts    = time.time_ns()
            else:
                continue  # Unknown tick format — skip

            # Write to SHM ring buffer: [ts, token, price, volume]
            data = np.array([ts, tid, price, vol], dtype=np.float64)
            self.shm.put(data)

    def _process_loop(self):
        import time
        import random
        import numpy as np
        
        if self.use_real_data:
            # Just keep alive, callbacks handle data
            while not self.stop_event.is_set():
                time.sleep(1)
            return

        # --- MOCK LOOP ---
        logger.info("Starting Mock Data Stream...")
        while not self.stop_event.is_set():
            # 1. Generate Tick
            tid = random.choice(self.tickers)
            price = self.prices[tid] * random.uniform(0.9995, 1.0005)
            self.prices[tid] = price
            
            # 2. Write to SHM
            ts = time.time_ns()
            data = np.array([ts, tid, price, 100], dtype=np.float64)
            self.shm.put(data)
            
            time.sleep(0.1) # 10 ticks/sec
 

class FeatureEngineeringWorker(BaseWorker):
    def _initialize_components(self):
        # Initialize Shared Memory readers here
        logger.info("FeatureEngineeringWorker Ready.")

    def _process_loop(self):
        """
        CPU Bound: BUSY WAIT STRATEGY.
        We do NOT sleep. We spin.
        """
        while not self.stop_event.is_set():
            # 1. Check for new data in Shared Memory (RingBuffer)
            # has_data = ring_buffer.check() 
            has_data = False # Mock
            
            if has_data:
                # Process immediately
                pass
            else:
                # BUSY WAIT: Do nothing, but hold the CPU.
                # Only yield slightly to prevent complete OS freeze if not using isolcpus
                # If using isolcpus, remove the sleep entirely.
                time.sleep(0.000001) 

class InferenceWorker(BaseWorker):
    def _initialize_components(self):
        from core.infrastructure.ipc import SharedMemoryRingBuffer
        from core.models.predictor import MARK5Predictor
        from core.data.provider import DataProvider
        from core.utils.config_manager import get_config
        import numpy as np
        import pandas as pd
        
        self.logger.info("Initializing Inference Engine...")
        self.config = get_config()
        
        # 1. Connect to IPC (Reader) with Retry
        import time
        retries = 10
        self.shm = None
        for i in range(retries):
            try:
                self.shm = SharedMemoryRingBuffer(
                    name="mark5_tick_buffer", 
                    shape=(4,),
                    dtype=np.float64, 
                    slots=100000, 
                    create=False
                )
                self.logger.info("✅ Connected to Tick Buffer")
                break
            except FileNotFoundError:
                if i < retries - 1:
                    self.logger.warning(f"Waiting for Tick Buffer... ({i+1}/{retries})")
                    time.sleep(1)
                else:
                    raise FileNotFoundError("Could not connect to Tick Buffer after retries")
        
        # 2. Initialize Predictors & Buffers
        self.predictors = {}
        self.buffers = {} # {ticker_id: DataFrame}
        self.id_map = {} # {ticker_id: symbol_name} -> Need a way to map ID to Symbol
        
        # We need the watchlist to know what models to load
        watchlist = getattr(self.config.trading, 'watchlist', ["COFORGE", "PERSISTENT", "KPITTECH", "HAL", "BEL"])
        
        # Initialize DataProvider for history hydration
        self.provider = DataProvider(self.config)
        
        # Hydrate Buffers
        for symbol in watchlist:
            try:
                # Load History (Warmup)
                df = self.provider.initialize_symbol(symbol, period="5d", interval="minute") # 5 days minute data
                if df is not None and not df.empty:
                    self.buffers[symbol] = df
                    self.predictors[symbol] = MARK5Predictor(symbol)
                    self.logger.info(f"Loaded Predictor for {symbol}")
                    
                    # Store ID mapping (This is tricky without live feed metadata, 
                    # but DataProvider usually knows. For now, we might rely on Symbol strings if IPC sent strings, 
                    # but IPC sends IDs (floats).
                    # We need a map. Provider -> Feed -> Token Map.
                    if self.provider.feed and hasattr(self.provider.feed, 'get_instrument_token'):
                        token = self.provider.feed.get_instrument_token(symbol)
                        if token:
                            self.id_map[token] = symbol
                else:
                     self.logger.warning(f"Could not hydrate history for {symbol}")
            except Exception as e:
                self.logger.error(f"Failed to load model for {symbol}: {e}")

        # If ID map is empty (Mock mode or failure), fallback to mocked IDs
        if not self.id_map:
             # Basic mapping for our mock tickers if running in mock mode
             mock_map = {738561: "COFORGE", 2953217: "PERSISTENT", 408065: "KPITTECH", 341249: "HAL", 1270529: "BEL"}
             self.id_map.update(mock_map)

        # 3. Setup Signal Output (Producer)
        # Shape: [Timestamp(f64), TickerID(f64), Signal(f64), Confidence(f64)]
        self.signal_shm = SharedMemoryRingBuffer(
            name="mark5_signal_buffer",
            shape=(4,),
            dtype=np.float64,
            slots=1000, # Lower volume for signals
            create=True # Owner
        )
        self.logger.info("✅ Signal Buffer Created: mark5_signal_buffer")

        self.logger.info("InferenceWorker Ready.")

    def _process_loop(self):
        import pandas as pd
        import numpy as np
        import time
        
        while not self.stop_event.is_set():
            # 1. Read Tick from IPC
            # data: [ts, id, price, vol]
            row = self.shm.get(timeout=0.001)
            
            if row is not None:
                ts, tid, price, vol = row
                symbol = self.id_map.get(str(int(tid))) # Try strict
                if not symbol: symbol = self.id_map.get(int(tid)) # Try int
                
                if symbol and symbol in self.buffers:
                    
                    df = self.buffers[symbol]
                    current_ts = pd.to_datetime(ts, unit='ns').tz_localize('UTC').tz_convert('Asia/Kolkata')
                    
                    # Optimized Append
                    new_row = pd.DataFrame([{
                        'open': price, 'high': price, 'low': price, 'close': price, 'volume': vol
                    }], index=[current_ts])
                    
                    # Concatenate and keep tailored size (last 500)
                    df = pd.concat([df, new_row])
                    if len(df) > 500:
                        df = df.iloc[-500:]
                    self.buffers[symbol] = df
                    
                    # Run Prediction
                    result = self.predictors[symbol].predict(df)
                    
                    if result['status'] == 'success':
                         sig_val = 0.0
                         if result['signal'] == 'BUY': sig_val = 1.0
                         elif result['signal'] == 'SELL': sig_val = -1.0
                         
                         if sig_val != 0.0:
                             self.logger.info(f"🔮 SIGNAL {symbol}: {result['signal']} ({result['confidence']:.2f})")
                             
                             # Write to Signal Buffer
                             # [ts, id, signal, confidence]
                             sig_data = np.array([ts, tid, sig_val, result['confidence']], dtype=np.float64)
                             self.signal_shm.put(sig_data)
            else:
                 # Yield
                 time.sleep(0.0001)

class ExecutionWorker(BaseWorker):
    def _initialize_components(self):
        from core.infrastructure.ipc import SharedMemoryRingBuffer
        from core.execution.execution_engine import ExecutionEngine
        from core.data.provider import DataProvider
        from core.utils.config_manager import get_config
        import numpy as np
        
        self.logger.info("Initializing Execution Engine...")
        self.config = get_config()
        
        # 1. Connect to IPC (Signal Reader)
        # 1. Connect to IPC (Signal Reader) with Retry
        # Shape: [Timestamp(f64), TickerID(f64), Signal(f64), Confidence(f64)]
        import time
        retries = 10
        self.signal_shm = None
        for i in range(retries):
            try:
                self.signal_shm = SharedMemoryRingBuffer(
                    name="mark5_signal_buffer",
                    shape=(4,),
                    dtype=np.float64, 
                    slots=1000, 
                    create=False
                )
                self.logger.info("✅ Connected to Signal Buffer")
                break
            except FileNotFoundError:
                if i < retries - 1:
                    self.logger.warning(f"Waiting for Signal Buffer... ({i+1}/{retries})")
                    time.sleep(1)
                else:
                    raise FileNotFoundError("Could not connect to Signal Buffer after retries")
        
        # 2. Execution Engine
        # Determine mode from config or args. Default to paper for safety.
        mode = getattr(self.config.execution, 'mode', 'paper')
        self.engine = ExecutionEngine(mode=mode)
        self.logger.info(f"Execution Engine Online (Mode: {mode})")
        
        # 3. ID Map (Need to resolve ID to Symbol for Orders)
        self.id_map = {} 
        try:
             self.provider = DataProvider(self.config)
             # Mock Map Fallback
             mock_map = {738561: "COFORGE", 2953217: "PERSISTENT", 408065: "KPITTECH", 341249: "HAL", 1270529: "BEL"}
             self.id_map.update(mock_map)
             
             if self.provider.connect() and hasattr(self.provider.feed, 'get_instrument_token'):
                 # Ideally we preload common tokens
                 watchlist = getattr(self.config.trading, 'watchlist', [])
                 for sym in watchlist:
                     token = self.provider.feed.get_instrument_token(sym)
                     if token: self.id_map[token] = sym
        except Exception as e:
            self.logger.warning(f"ID Map Init warning: {e}")

    def _process_loop(self):
        import time
        
        while not self.stop_event.is_set():
            # Read Signal
            row = self.signal_shm.get(timeout=0.01)
            
            if row is not None:
                ts, tid, sig_val, conf = row
                
                symbol = self.id_map.get(str(int(tid))) or self.id_map.get(int(tid))
                if not symbol:
                    self.logger.warning(f"Unknown Ticker ID: {tid}")
                    continue
                
                # Logic: Execute on High Confidence
                if conf > 0.6: # Configurable threshold
                    side = "BUY" if sig_val > 0 else "SELL"
                    
                    # Execute
                    # Qty management is complex (Risk Manager). 
                    # For now, fixed qty 1.
                    success = self.engine.execute_order(
                        symbol=symbol,
                        side=side,
                        qty=1,
                        order_type="MARKET"
                    )
                    
                    if success:
                        self.logger.info(f"⚡ ORDER EXECUTED: {side} {symbol} (Conf: {conf:.2f})")
                    else:
                        self.logger.error(f"Order Failed: {side} {symbol}")
            else:
                time.sleep(0.001)

class ProcessManager:
    _instance = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(ProcessManager, cls).__new__(cls)
        return cls._instance

    def __init__(self):
        if hasattr(self, 'initialized'): return
        self.logger = logging.getLogger("MARK5.ProcMan")
        self.processes = {}
        self.worker_defs = {} 
        self.lock = threading.Lock()
        self.running = False
        self.initialized = True

    def register_worker(self, name: str, worker_cls: Type[BaseWorker], core_id: int = None):
        self.worker_defs[name] = {'cls': worker_cls, 'core_id': core_id}

    def start_all(self):
        self.running = True
        for name, spec in self.worker_defs.items():
            self._spawn_worker(name, spec)
        
        # Monitor thread
        threading.Thread(target=self._lazarus_loop, daemon=True).start()

    def _spawn_worker(self, name, spec):
        worker = spec['cls'](name=name, core_id=spec['core_id'])
        worker.start()
        with self.lock:
            self.processes[name] = worker

    def _lazarus_loop(self):
        while self.running:
            time.sleep(1.0)
            with self.lock:
                for name, worker in list(self.processes.items()):
                    if not worker.is_alive():
                        self.logger.critical(f"💀 WORKER {name} DIED. RESPAWNING.")
                        del self.processes[name]
                        self._spawn_worker(name, self.worker_defs[name])

    def stop_all(self):
        self.running = False
        for p in self.processes.values():
            p.stop()
            p.join(timeout=1)
            if p.is_alive(): p.terminate()

def get_process_manager() -> ProcessManager:
    return ProcessManager()
