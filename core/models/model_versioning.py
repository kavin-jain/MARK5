import json
import os
import tempfile
from pathlib import Path
from typing import Dict, Optional

class ModelVersionManager:
    def __init__(self, config: Dict):
        self.config = config
        # Use config path or default to models/versions.json
        version_path_str = self.config.get("model_versions_path", "models/versions.json")
        self.version_file = Path(version_path_str)
        self._versions: Dict[str, int] = self._load_versions()

    def _load_versions(self) -> Dict[str, int]:
        if not self.version_file.exists():
            return {}
        try:
            with open(self.version_file, 'r') as f:
                return json.load(f)
        except Exception:
            return {}

    def _save_versions(self):
        # Ensure parent directory exists
        self.version_file.parent.mkdir(parents=True, exist_ok=True)
        # Write to a temp file in the same directory, then atomically replace
        # the target file. This prevents concurrent writers from producing a
        # corrupt (partially-written) JSON file.
        dir_path = str(self.version_file.parent)
        with tempfile.NamedTemporaryFile(
            mode='w', dir=dir_path, suffix='.tmp', delete=False
        ) as tmp:
            json.dump(self._versions, tmp, indent=2)
            tmp_path = tmp.name
        os.replace(tmp_path, str(self.version_file))

    def get_latest_version(self, ticker: str) -> int:
        return self._versions.get(ticker, 0)

    def increment_version(self, ticker: str) -> int:
        current = self.get_latest_version(ticker)
        new_version = current + 1
        self._versions[ticker] = new_version
        self._save_versions()
        return new_version
