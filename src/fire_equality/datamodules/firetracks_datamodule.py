"""
FireTracks DataModule for PyTorch Lightning
forload processed_firetracks_pixel_binary.pth 
"""

import os
import torch
from pathlib import Path
from torch.utils.data import DataLoader, random_split
from pytorch_lightning import LightningDataModule

from .firetracks_loader import FireTracksDataset


class FireTracksBinaryDataModule(LightningDataModule):
    """FireTracksdatamodule"""
    
    def __init__(
        self,
        data_path: str = 'processed_firetracks_pixel_binary.pth',
        batch_size: int = 32,
        train_ratio: float = 0.7,
        val_ratio: float = 0.15,
        test_ratio: float = 0.15,
        num_workers: int = 0,
        seed: int = 42,
        load_multiple_files: bool = False  # load
    ):
        super().__init__()
        # 
        if isinstance(data_path, str):
            self.data_path = data_path
            self.data_paths = None
        elif isinstance(data_path, (list, tuple)):
            self.data_paths = list(data_path)
            self.data_path = self.data_paths[0] if self.data_paths else None
        else:
            self.data_path = data_path
            self.data_paths = None
        
        self.batch_size = batch_size
        self.train_ratio = train_ratio
        self.val_ratio = val_ratio
        self.test_ratio = test_ratio
        self.num_workers = num_workers
        self.seed = seed
        self.load_multiple_files = load_multiple_files or (self.data_paths is not None)
        
        # 
        assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6, \
            f"1.0,: {train_ratio + val_ratio + test_ratio}"
        
        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None
    
    def _find_data_file(self, data_path):
        """data file path"""
        #,
        if os.path.isabs(data_path):
            if os.path.exists(data_path):
                return data_path
            else:
                raise FileNotFoundError(f": {data_path}")
        
        # (FireEqual)
        current_file = os.path.abspath(__file__)
        project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(current_file))))
        
        # ()
        possible_paths = []
        
        #,()
        if '..' in data_path or '/' in data_path or '\\' in data_path:
            resolved = os.path.normpath(os.path.join(project_root, data_path))
            possible_paths.append(resolved)
        
        #  data/processed  dataset 
        if 'data/processed' not in data_path and 'dataset' not in data_path:
            for sub in ('data/processed', 'dataset'):
                possible_paths.append(os.path.join(project_root, sub, os.path.basename(data_path)))
        elif 'dataset' not in data_path and not data_path.startswith('data/'):
            possible_paths.append(os.path.join(project_root, 'data/processed', os.path.basename(data_path)))
        
        # 
        possible_paths.append(os.path.join(project_root, os.path.basename(data_path)))
        
        # (Hydra)
        possible_paths.extend([
            os.path.join(os.getcwd(), data_path),  # 
            os.path.join(os.getcwd(), os.path.basename(data_path)),  # 
        ])
        
        # 
        for path in possible_paths:
            abs_path = os.path.abspath(path)
            if os.path.exists(abs_path) and os.path.isfile(abs_path):
                return abs_path
        
        #,
        error_msg = f" '{data_path}'.:\n"
        for p in possible_paths:
            abs_p = os.path.abspath(p)
            exists = "✓" if os.path.exists(abs_p) else "✗"
            error_msg += f"  {exists} {abs_p}\n"
        error_msg += f"\n: {project_root}\n"
        error_msg += f": {os.getcwd()}\n"
        raise FileNotFoundError(error_msg)
    
    def _load_single_file(self, data_path):
        """load"""
        # #region agent log
        import json
        import os
        log_path = Path(__file__).parent.parent.parent.parent / '.cursor' / 'debug.log'
        try:
            file_size = os.path.getsize(data_path) if os.path.exists(data_path) else 0
            with open(log_path, 'a', encoding='utf-8') as f:
                f.write(json.dumps({
                    "sessionId": "debug-session",
                    "runId": "run2",
                    "hypothesisId": "F",
                    "location": "firetracks_datamodule.py:_load_single_file",
                    "message": "Before loading file",
                    "data": {"data_path": str(data_path), "file_exists": os.path.exists(data_path), "file_size_gb": file_size / (1024**3)},
                    "timestamp": int(__import__('time').time() * 1000)
                }) + '\n')
        except: pass
        # #endregion
        
        # PyTorch 2.6+ omegaconf
        try:
            import omegaconf
            torch.serialization.add_safe_globals([
                omegaconf.listconfig.ListConfig,
                omegaconf.dictconfig.DictConfig,
                omegaconf.omegaconf.OmegaConf,
            ])
            # #region agent log
            try:
                with open(log_path, 'a', encoding='utf-8') as f:
                    f.write(json.dumps({
                        "sessionId": "debug-session",
                        "runId": "run2",
                        "hypothesisId": "G",
                        "location": "firetracks_datamodule.py:_load_single_file",
                        "message": "Trying weights_only=True",
                        "data": {},
                        "timestamp": int(__import__('time').time() * 1000)
                    }) + '\n')
            except: pass
            # #endregion
            data = torch.load(data_path, weights_only=True)
        except Exception as e:
            # #region agent log
            try:
                with open(log_path, 'a', encoding='utf-8') as f:
                    f.write(json.dumps({
                        "sessionId": "debug-session",
                        "runId": "run2",
                        "hypothesisId": "H",
                        "location": "firetracks_datamodule.py:_load_single_file",
                        "message": "weights_only=True failed, trying weights_only=False",
                        "data": {"error": str(e)[:200]},
                        "timestamp": int(__import__('time').time() * 1000)
                    }) + '\n')
            except: pass
            # #endregion
            try:
                data = torch.load(data_path, weights_only=False)
            except Exception as e2:
                # #region agent log
                try:
                    with open(log_path, 'a', encoding='utf-8') as f:
                        f.write(json.dumps({
                            "sessionId": "debug-session",
                            "runId": "run2",
                            "hypothesisId": "I",
                            "location": "firetracks_datamodule.py:_load_single_file",
                            "message": "weights_only=False also failed",
                            "data": {"error": str(e2)[:200]},
                            "timestamp": int(__import__('time').time() * 1000)
                        }) + '\n')
                except: pass
                # #endregion
                raise
        
        # #region agent log
        try:
            with open(log_path, 'a', encoding='utf-8') as f:
                f.write(json.dumps({
                    "sessionId": "debug-session",
                    "runId": "run2",
                    "hypothesisId": "J",
                    "location": "firetracks_datamodule.py:_load_single_file",
                    "message": "File loaded successfully",
                    "data": {"data_keys": list(data.keys()) if isinstance(data, dict) else "not_dict", "has_spatiotemporal_samples": 'spatiotemporal_samples' in data if isinstance(data, dict) else False},
                    "timestamp": int(__import__('time').time() * 1000)
                }) + '\n')
        except: pass
        # #endregion
        
        if 'spatiotemporal_samples' not in data:
            raise ValueError(f" '{data_path}'  'spatiotemporal_samples'")
        
        return data['spatiotemporal_samples']
    
    def setup(self, stage=None):
        """load"""
        if self.train_dataset is not None:
            # 
            return
        
        #  data_path 
        from omegaconf import ListConfig
        is_list = isinstance(self.data_path, (list, tuple, ListConfig))
        
        # load
        if self.load_multiple_files or is_list:
            # :load
            # ( self.data_paths  self.data_path)
            if self.data_paths:
                file_list = self.data_paths
            elif is_list:
                file_list = self.data_path
            else:
                raise ValueError("")
            
            # 
            data_files = []
            for path in file_list:
                path_str = str(path)  # convert to string
                found_path = self._find_data_file(path_str)
                data_files.append(found_path)
            
            print(f"📊 :load {len(data_files)} ")
            
            # load()
            all_samples = []
            import gc
            for i, data_file in enumerate(data_files):
                print(f"   [{i+1}/{len(data_files)}] load: {os.path.basename(data_file)}")
                samples = self._load_single_file(data_file)
                all_samples.extend(samples)
                print(f"      load {len(samples):,}, {len(all_samples):,} ")
                # 
                del samples
                gc.collect()
            
            spatiotemporal_samples = all_samples
            print(f"   ✅ load {len(spatiotemporal_samples):,} ")
            
        else:
            # :load
            data_path_str = str(self.data_path)  # ensure this is a string
            data_path = self._find_data_file(data_path_str)
            print(f"📊 load: {data_path}")
            spatiotemporal_samples = self._load_single_file(data_path)
            print(f"   ✅ : {len(spatiotemporal_samples):,}")
        
        # 
        full_dataset = FireTracksDataset(
            spatiotemporal_samples,
            target_type='binary_classification'
        )
        
        # 
        total_size = len(full_dataset)
        train_size = int(total_size * self.train_ratio)
        val_size = int(total_size * self.val_ratio)
        test_size = total_size - train_size - val_size
        
        self.train_dataset, self.val_dataset, self.test_dataset = random_split(
            full_dataset,
            [train_size, val_size, test_size],
            generator=torch.Generator().manual_seed(self.seed)
        )
        
        print(f"   📊 :")
        print(f"      - : {len(self.train_dataset)}  ({self.train_ratio*100:.1f}%)")
        print(f"      - : {len(self.val_dataset)}  ({self.val_ratio*100:.1f}%)")
        print(f"      - : {len(self.test_dataset)}  ({self.test_ratio*100:.1f}%)")
    
    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=True if self.num_workers > 0 else False
        )
    
    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True if self.num_workers > 0 else False
        )
    
    def test_dataloader(self):
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True if self.num_workers > 0 else False
        )

