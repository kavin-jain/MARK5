"""
MARK5 ROBUST REGISTRY v8.0 - HARDENED EDITION
Revisions:
    v7.0: Atomic writes, thread-safe IO, staleness guard.
    v8.0: Critical fixes:
        1. CONCURRENCY: Lock protects all registry mutations (not just file I/O)
        2. UNIQUE IDS: UUID-based model IDs prevent collisions
        3. PATH SECURITY: Validates paths against base directory
        4. STRONG CHECKSUM: SHA-256 replaces MD5
        5. SAFE LOAD: Corrupted JSON raises exception instead of silent wipe
        6. INPUT VALIDATION: Ticker normalization, metadata check
"""

import json
import logging
import hashlib
import os
import shutil
import uuid
import tempfile
from datetime import datetime
from pathlib import Path
from typing import Dict, Optional
from threading import Lock

logger = logging.getLogger(__name__)

# Constants
CHUNK_SIZE = 8192
STATUS_ACTIVE = 'active'
STATUS_ARCHIVED = 'archived'
STATUS_MISSING = 'missing'


class RegistryError(Exception):
    """Base exception for registry operations."""
    pass


class RegistryCorruptionError(RegistryError):
    """Raised when registry file is corrupted."""
    pass


class PathValidationError(RegistryError):
    """Raised when model path fails security validation."""
    pass


class RobustModelRegistry:
    """
    Thread-safe model registry with atomic persistence.
    
    Manages registration, retrieval, and lifecycle of ML models
    for the MARK5 trading system.
    """
    
    def __init__(
        self,
        registry_path: str = 'models/registry.json',
        base_model_dir: Optional[str] = None
    ):
        """
        Initialize the registry.
        
        Args:
            registry_path: Path to the JSON registry file
            base_model_dir: Base directory for model files (for path validation).
                           If None, uses parent of registry_path.
        """
        self.registry_path = Path(registry_path).resolve()
        self.base_model_dir = (
            Path(base_model_dir).resolve() if base_model_dir
            else self.registry_path.parent
        )
        self._lock = Lock()
        self.registry: Dict[str, Dict] = self._load_registry()
    
    def _validate_path(self, path: str) -> Path:
        """
        Validate that path is within allowed base directory.
        
        Args:
            path: Path to validate
            
        Returns:
            Resolved Path object
            
        Raises:
            PathValidationError: If path escapes base directory
        """
        resolved = Path(path).resolve()
        
        try:
            resolved.relative_to(self.base_model_dir)
        except ValueError:
            raise PathValidationError(
                f"Path '{path}' is outside allowed directory '{self.base_model_dir}'"
            )
        
        if not resolved.is_file():
            raise FileNotFoundError(f"Model file '{path}' does not exist or is not a file")
        
        return resolved
    
    def _validate_ticker(self, ticker: str) -> str:
        """Normalize and validate ticker symbol."""
        if not isinstance(ticker, str):
            raise ValueError(f"Ticker must be string, got {type(ticker).__name__}")
        normalized = ticker.strip().upper()
        if not normalized:
            raise ValueError("Ticker cannot be empty")
        return normalized
    
    def _validate_metadata(self, metadata: Dict) -> None:
        """Validate that metadata is JSON-serializable."""
        if not isinstance(metadata, dict):
            raise TypeError(f"Metadata must be dict, got {type(metadata).__name__}")
        try:
            json.dumps(metadata)
        except (TypeError, ValueError) as e:
            raise TypeError(f"Metadata is not JSON-serializable: {e}")
    
    def _load_registry(self) -> Dict:
        """
        Load registry from file.
        
        Returns:
            Registry dict
            
        Raises:
            RegistryCorruptionError: If file is corrupted
        """
        with self._lock:
            if not self.registry_path.exists():
                return {}
            try:
                with open(self.registry_path, 'r') as f:
                    data = json.load(f)
                    if not isinstance(data, dict):
                        raise RegistryCorruptionError("Registry must be a JSON object")
                    return data
            except json.JSONDecodeError as e:
                logger.error(f"Registry corrupted at {self.registry_path}: {e}")
                # Create backup of corrupted file
                backup_path = self.registry_path.with_suffix('.corrupted.bak')
                try:
                    shutil.copy2(self.registry_path, backup_path)
                    logger.info(f"Backed up corrupted registry to {backup_path}")
                except OSError:
                    pass
                raise RegistryCorruptionError(f"Invalid JSON in registry: {e}")
            except OSError as e:
                logger.error(f"Failed to read registry: {e}")
                raise RegistryError(f"I/O error reading registry: {e}")
    
    def _atomic_save(self) -> None:
        """
        Atomically save registry to disk.
        
        Uses temp file + rename pattern for crash safety.
        Caller must hold self._lock.
        """
        dir_name = self.registry_path.parent
        dir_name.mkdir(parents=True, exist_ok=True)
        
        fd, tmp_path = tempfile.mkstemp(
            suffix='.json',
            prefix='registry_',
            dir=dir_name
        )
        try:
            with os.fdopen(fd, 'w') as f:
                json.dump(self.registry, f, indent=2, sort_keys=True)
                f.flush()
                os.fsync(f.fileno())
            shutil.move(tmp_path, self.registry_path)
        except Exception as e:
            if os.path.exists(tmp_path):
                os.unlink(tmp_path)
            raise RegistryError(f"Failed to save registry: {e}")
    
    def _compute_checksum(self, file_path: Path) -> str:
        """
        Compute SHA-256 checksum of file.
        
        Args:
            file_path: Path to file
            
        Returns:
            Hex digest of SHA-256 hash
        """
        sha256 = hashlib.sha256()
        with open(file_path, 'rb') as f:
            for chunk in iter(lambda: f.read(CHUNK_SIZE), b""):
                sha256.update(chunk)
        return sha256.hexdigest()
    
    def _generate_model_id(self, ticker: str, model_type: str) -> str:
        """Generate unique model ID using UUID."""
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        unique_suffix = uuid.uuid4().hex[:8]
        return f"{ticker}_{model_type}_{timestamp}_{unique_suffix}"
    
    def register_model(
        self,
        ticker: str,
        model_type: str,
        path: str,
        metadata: Optional[Dict] = None
    ) -> str:
        """
        Register a new model, archiving any previous active version.
        
        Args:
            ticker: Ticker symbol
            model_type: Model type identifier
            path: Path to model file
            metadata: Optional metadata dict
            
        Returns:
            Unique model ID
            
        Raises:
            PathValidationError: If path fails security check
            FileNotFoundError: If model file doesn't exist
            TypeError: If metadata not JSON-serializable
        """
        normalized_ticker = self._validate_ticker(ticker)
        validated_path = self._validate_path(path)
        
        if metadata is None:
            metadata = {}
        self._validate_metadata(metadata)
        
        with self._lock:
            model_id = self._generate_model_id(normalized_ticker, model_type)
            
            # Ensure unique ID (highly unlikely collision, but safe)
            while model_id in self.registry:
                model_id = self._generate_model_id(normalized_ticker, model_type)
            
            # Archive previous active models for this ticker/type
            for mid, data in self.registry.items():
                if (data.get('ticker') == normalized_ticker and 
                    data.get('model_type') == model_type and
                    data.get('status') == STATUS_ACTIVE):
                    data['status'] = STATUS_ARCHIVED
                    data['archived_at'] = datetime.now().isoformat()
            
            # Create new entry
            entry = {
                'ticker': normalized_ticker,
                'model_type': model_type,
                'path': str(validated_path),
                'metadata': metadata,
                'status': STATUS_ACTIVE,
                'created_at': datetime.now().isoformat(),
                'checksum': self._compute_checksum(validated_path)
            }
            
            self.registry[model_id] = entry
            self._atomic_save()
            
            logger.info(f"Registered model {model_id} for {normalized_ticker}/{model_type}")
            return model_id
    
    def get_production_model(self, ticker: str, model_type: str) -> Optional[Dict]:
        """
        Get the active model for a ticker/type.
        
        Args:
            ticker: Ticker symbol
            model_type: Model type identifier
            
        Returns:
            Model entry dict or None if not found
        """
        normalized_ticker = self._validate_ticker(ticker)
        
        with self._lock:
            # Find active model
            for mid, data in self.registry.items():
                if (data.get('ticker') == normalized_ticker and
                    data.get('model_type') == model_type and
                    data.get('status') == STATUS_ACTIVE):
                    
                    # Integrity check: verify file exists
                    if not os.path.exists(data['path']):
                        logger.error(f"Model {mid} file missing, marking as missing")
                        data['status'] = STATUS_MISSING
                        data['missing_since'] = datetime.now().isoformat()
                        self._atomic_save()
                        return None
                    
                    # Return copy to prevent external mutation
                    return dict(data)
            
            return None
    
    def get_model_by_id(self, model_id: str) -> Optional[Dict]:
        """Get a specific model by ID."""
        with self._lock:
            entry = self.registry.get(model_id)
            return dict(entry) if entry else None
    
    def list_models(
        self,
        ticker: Optional[str] = None,
        model_type: Optional[str] = None,
        status: Optional[str] = None
    ) -> Dict[str, Dict]:
        """
        List models with optional filtering.
        
        Args:
            ticker: Filter by ticker (optional)
            model_type: Filter by model type (optional)
            status: Filter by status (optional)
            
        Returns:
            Dict of matching model_id -> entry
        """
        normalized_ticker = self._validate_ticker(ticker) if ticker else None
        
        with self._lock:
            result = {}
            for mid, data in self.registry.items():
                if normalized_ticker and data.get('ticker') != normalized_ticker:
                    continue
                if model_type and data.get('model_type') != model_type:
                    continue
                if status and data.get('status') != status:
                    continue
                result[mid] = dict(data)
            return result
    
    def verify_checksum(self, model_id: str) -> bool:
        """
        Verify that a model file's checksum matches the registry.
        
        Returns:
            True if checksum matches, False otherwise
        """
        with self._lock:
            entry = self.registry.get(model_id)
            if not entry:
                return False
            
            path = Path(entry['path'])
            if not path.exists():
                return False
            
            current_checksum = self._compute_checksum(path)
            return current_checksum == entry.get('checksum')
    
    def health_check(self) -> Dict[str, any]:
        """
        Perform health check on registry.
        
        Returns:
            Dict with health status and statistics
        """
        with self._lock:
            total = len(self.registry)
            active = sum(1 for d in self.registry.values() if d.get('status') == STATUS_ACTIVE)
            archived = sum(1 for d in self.registry.values() if d.get('status') == STATUS_ARCHIVED)
            missing = sum(1 for d in self.registry.values() if d.get('status') == STATUS_MISSING)
            
            return {
                'healthy': True,
                'registry_path': str(self.registry_path),
                'base_model_dir': str(self.base_model_dir),
                'total_entries': total,
                'active': active,
                'archived': archived,
                'missing': missing
            }
