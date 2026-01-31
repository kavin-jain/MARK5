import json
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
        with open(self.version_file, 'w') as f:
            json.dump(self._versions, f, indent=2)

    def get_latest_version(self, ticker: str) -> int:
        return self._versions.get(ticker, 0)

    def increment_version(self, ticker: str) -> int:
        current = self.get_latest_version(ticker)
        new_version = current + 1
        self._versions[ticker] = new_version
        self._save_versions()
        return new_version
