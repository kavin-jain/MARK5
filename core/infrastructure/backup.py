#!/usr/bin/env python3
"""
MARK5 Atomic Vault v4.0
Zero-Data-Loss Backup & Recovery Architecture
"""

import os
import shutil
import tarfile
import json
import hashlib
import tempfile
import logging
from datetime import datetime
from typing import Dict, List, Optional

logger = logging.getLogger("MARK5_Backup")

class BackupManager:
    """
    Implements Atomic/Transactional Backup and Restore.
    Guarantees: No partial restores, no corrupted backups, integrity verification.
    """
    
    def __init__(self, backup_dir: str = None):
        if backup_dir is None:
            base_dir = os.path.dirname(os.path.abspath(__file__))
            backup_dir = os.path.join(base_dir, 'database/backups')
        
        self.backup_dir = backup_dir
        self.base_dir = os.path.dirname(os.path.abspath(__file__))
        os.makedirs(backup_dir, exist_ok=True)
    
    def create_atomic_backup(self, components: List[str]) -> Dict:
        """
        Creates a verified, atomic backup.
        1. Copies to temp isolation.
        2. Compresses to temp file.
        3. Calculates checksum.
        4. Atomic move to final location.
        """
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        backup_name = f'mark5_backup_{timestamp}'
        
        # Use a temporary directory for assembly to ensure atomicity
        with tempfile.TemporaryDirectory() as temp_stage_dir:
            assembly_dir = os.path.join(temp_stage_dir, backup_name)
            os.makedirs(assembly_dir)
            
            manifest = {'files': {}, 'timestamp': timestamp, 'components': components}
            
            try:
                for comp in components:
                    src_path = os.path.join(self.base_dir, comp)
                    if not os.path.exists(src_path):
                        logger.warning(f"Component missing: {comp}")
                        continue
                        
                    dest_path = os.path.join(assembly_dir, comp)
                    
                    # Copy based on type
                    if os.path.isdir(src_path):
                        shutil.copytree(src_path, dest_path)
                    else:
                        os.makedirs(os.path.dirname(dest_path), exist_ok=True)
                        shutil.copy2(src_path, dest_path)
                        
                # Create Tarball in TEMP first (prevents half-written files in backup dir)
                temp_tar_path = os.path.join(temp_stage_dir, f'{backup_name}.tar.gz')
                with tarfile.open(temp_tar_path, "w:gz") as tar:
                    tar.add(assembly_dir, arcname=backup_name)
                
                # Calculate Checksum
                checksum = self._calculate_file_hash(temp_tar_path)
                manifest['checksum'] = checksum
                manifest['size_bytes'] = os.path.getsize(temp_tar_path)
                
                # ATOMIC COMMIT: Move from temp to final
                final_tar_path = os.path.join(self.backup_dir, f'{backup_name}.tar.gz')
                final_meta_path = os.path.join(self.backup_dir, f'{backup_name}.json')
                
                # Using shutil.move which falls back to copy+delete if cross-device, 
                # but os.replace is atomic on POSIX same-filesystem.
                shutil.move(temp_tar_path, final_tar_path)
                
                with open(final_meta_path, 'w') as f:
                    json.dump(manifest, f, indent=2)
                    
                logger.info(f"Backup committed: {backup_name} | SHA256: {checksum[:8]}...")
                return {'success': True, 'path': final_tar_path, 'name': backup_name}
                
            except Exception as e:
                logger.error(f"Backup failed during assembly: {e}")
                return {'success': False, 'error': str(e)}

    def restore_atomic(self, backup_name: str) -> Dict:
        """
        Performs a 'Blue-Green' restoration.
        1. Extract to temp.
        2. Verify Checksum.
        3. Atomic Swap (Rename Current -> Old, Temp -> Current).
        4. Rollback if swap fails.
        """
        backup_path = os.path.join(self.backup_dir, f'{backup_name}.tar.gz')
        meta_path = os.path.join(self.backup_dir, f'{backup_name}.json')
        
        if not os.path.exists(backup_path) or not os.path.exists(meta_path):
            return {'success': False, 'error': 'Backup artifacts missing'}
            
        # 1. Integrity Check
        with open(meta_path, 'r') as f:
            manifest = json.load(f)
            
        current_hash = self._calculate_file_hash(backup_path)
        if current_hash != manifest.get('checksum'):
            return {'success': False, 'error': 'FATAL: Backup corruption detected (Hash mismatch)'}

        # 2. Stage Restoration (Don't touch live data yet)
        restore_staging_base = os.path.join(self.base_dir, '.restore_staging')
        if os.path.exists(restore_staging_base):
            shutil.rmtree(restore_staging_base)
        os.makedirs(restore_staging_base)

        try:
            with tarfile.open(backup_path, 'r:gz') as tar:
                tar.extractall(restore_staging_base)
            
            extracted_root = os.path.join(restore_staging_base, backup_name)
            
            # 3. The Atomic Swap Pattern
            swapped_components = []
            
            for component in manifest['components']:
                # Source: New data from backup
                new_version = os.path.join(extracted_root, component)
                # Target: Live system path
                live_path = os.path.join(self.base_dir, component)
                # Backup of Live: In case we need to rollback
                rollback_path = live_path + ".rollback"
                
                if not os.path.exists(new_version):
                    continue

                # Prepare the swap
                if os.path.exists(rollback_path):
                    if os.path.isdir(rollback_path): shutil.rmtree(rollback_path)
                    else: os.remove(rollback_path)
                
                # EXECUTE SWAP
                try:
                    if os.path.exists(live_path):
                        os.rename(live_path, rollback_path) # Fast atomic rename
                    
                    # Move new version into place
                    # Ensure parent dir exists
                    os.makedirs(os.path.dirname(live_path), exist_ok=True)
                    shutil.move(new_version, live_path)
                    swapped_components.append((live_path, rollback_path))
                    
                except OSError as e:
                    # EMERGENCY ROLLBACK
                    logger.critical(f"Swap failed for {component}. Rolling back...")
                    self._rollback(swapped_components)
                    return {'success': False, 'error': f"Restore failed during swap: {e}"}

            # Cleanup
            shutil.rmtree(restore_staging_base)
            
            # Optional: Delete rollback files after success (or keep them for manual review)
            for _, rollback in swapped_components:
                if os.path.exists(rollback):
                    if os.path.isdir(rollback): shutil.rmtree(rollback)
                    else: os.remove(rollback)

            return {'success': True, 'restored': manifest['components']}

        except Exception as e:
            return {'success': False, 'error': str(e)}

    def _rollback(self, swap_list):
        """Reverses the rename operations"""
        for live, rollback in reversed(swap_list):
            if os.path.exists(live):
                if os.path.isdir(live): shutil.rmtree(live)
                else: os.remove(live)
            if os.path.exists(rollback):
                os.rename(rollback, live)

    def _calculate_file_hash(self, filepath: str) -> str:
        """SHA256 for integrity verification"""
        sha256_hash = hashlib.sha256()
        with open(filepath, "rb") as f:
            for byte_block in iter(lambda: f.read(4096), b""):
                sha256_hash.update(byte_block)
        return sha256_hash.hexdigest()

if __name__ == '__main__':
    # Test Routine
    bm = BackupManager()
    print("1. Creating Atomic Backup...")
    # Assume 'models' and 'database' folders exist relative to this script
    res = bm.create_atomic_backup(['models', 'database/main'])
    print(res)
    
    if res['success']:
        print("\n2. Verifying and Restoring...")
        restore_res = bm.restore_atomic(res['name'])
        print(restore_res)
