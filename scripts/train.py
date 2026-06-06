"""
(2002-2020)
 SimpleConvLSTM model:
- : loss(NLLLoss),EarlyStopping  val/loss.
- Grid search / multirun  model: val/auc( ModelCheckpoint monitor=val/auc).
- fairness analysis:val/auc  F1-score ().
"""
import os
import sys
from pathlib import Path

#  src ( release )
_release_root = Path(__file__).resolve().parent.parent
src_dir = _release_root / 'src'
if str(src_dir) not in sys.path:
    sys.path.insert(0, str(src_dir))

# (Hydra)
original_cwd = os.getcwd()

# 
from fire_equality.train import train
import hydra
from omegaconf import DictConfig

@hydra.main(config_path="../configs", config_name="config", version_base=None)
def main(config: DictConfig):
    """,Hydra"""
    # #region agent log
    import json
    log_path = Path(__file__).resolve().parent.parent / '.cursor' / 'debug.log'
    try:
        with open(log_path, 'a', encoding='utf-8') as f:
            f.write(json.dumps({
                "sessionId": "debug-session",
                "runId": "run1",
                "hypothesisId": "A",
                "location": "train_merged_dataset.py:25",
                "message": "Function entry",
                "data": {"has_datamodule": hasattr(config, 'datamodule')},
                "timestamp": int(__import__('time').time() * 1000)
            }) + '\n')
    except: pass
    # #endregion
    
    # ()
    if hasattr(config, 'datamodule'):
        # #region agent log
        try:
            with open(log_path, 'a', encoding='utf-8') as f:
                f.write(json.dumps({
                    "sessionId": "debug-session",
                    "runId": "run1",
                    "hypothesisId": "B",
                    "location": "train_merged_dataset.py:30",
                    "message": "Before struct check",
                    "data": {
                        "datamodule_keys": list(config.datamodule.keys()) if hasattr(config.datamodule, 'keys') else "N/A",
                        "datamodule_type": str(type(config.datamodule)),
                        "is_struct": getattr(config.datamodule, '_is_struct', 'unknown')
                    },
                    "timestamp": int(__import__('time').time() * 1000)
                }) + '\n')
        except: pass
        # #endregion
        
        # Release ( Hydra )
        project_root = Path(__file__).resolve().parent.parent
        
        # 
        merged_file = project_root / 'data/processed/processed_firetracks_pixel_binary_2002-2020.pth'
        
        #,()
        if merged_file.exists():
            # #region agent log
            try:
                with open(log_path, 'a', encoding='utf-8') as f:
                    f.write(json.dumps({
                        "sessionId": "debug-session",
                        "runId": "run1",
                        "hypothesisId": "C",
                        "location": "train_merged_dataset.py:50",
                        "message": "Before setting data_path",
                        "data": {"merged_file_exists": True},
                        "timestamp": int(__import__('time').time() * 1000)
                    }) + '\n')
            except: pass
            # #endregion
            
            config.datamodule.data_path = str(merged_file)
            
            # #region agent log
            try:
                from omegaconf import OmegaConf
                struct_before = OmegaConf.is_struct(config.datamodule)
                with open(log_path, 'a', encoding='utf-8') as f:
                    f.write(json.dumps({
                        "sessionId": "debug-session",
                        "runId": "run1",
                        "hypothesisId": "D",
                        "location": "train_merged_dataset.py:60",
                        "message": "Before setting load_multiple_files",
                        "data": {"is_struct": struct_before, "has_load_multiple_files": 'load_multiple_files' in config.datamodule},
                        "timestamp": int(__import__('time').time() * 1000)
                    }) + '\n')
            except Exception as e:
                try:
                    with open(log_path, 'a', encoding='utf-8') as f:
                        f.write(json.dumps({
                            "sessionId": "debug-session",
                            "runId": "run1",
                            "hypothesisId": "D",
                            "location": "train_merged_dataset.py:60",
                            "message": "Error checking struct",
                            "data": {"error": str(e)},
                            "timestamp": int(__import__('time').time() * 1000)
                        }) + '\n')
                except: pass
            # #endregion
            
            #  struct 
            from omegaconf import OmegaConf
            OmegaConf.set_struct(config.datamodule, False)
            config.datamodule.load_multiple_files = False
            
            # #region agent log
            try:
                with open(log_path, 'a', encoding='utf-8') as f:
                    f.write(json.dumps({
                        "sessionId": "debug-session",
                        "runId": "run1",
                        "hypothesisId": "E",
                        "location": "train_merged_dataset.py:75",
                        "message": "After setting load_multiple_files",
                        "data": {"load_multiple_files": config.datamodule.load_multiple_files},
                        "timestamp": int(__import__('time').time() * 1000)
                    }) + '\n')
            except: pass
            # #endregion
            
            print(f"📊 : {merged_file}")
        else:
            # (loading,)
            data_files = [
                'data/processed/processed_firetracks_pixel_binary_2002-2004.pth',
                'data/processed/processed_firetracks_pixel_binary_2005-2007.pth',
                'data/processed/processed_firetracks_pixel_binary_2008-2009.pth',
                'data/processed/processed_firetracks_pixel_binary_2010-2013.pth',
                'data/processed/processed_firetracks_pixel_binary_2014-2015.pth',
                'data/processed/processed_firetracks_pixel_binary_2016-2018.pth',
                'data/processed/processed_firetracks_pixel_binary_2019-2020.pth',
            ]
            # 
            data_paths = [str(project_root / f) for f in data_files]
            # 
            existing_paths = [p for p in data_paths if Path(p).exists()]
            
            if existing_paths:
                #  struct 
                from omegaconf import OmegaConf
                OmegaConf.set_struct(config.datamodule, False)
                
                config.datamodule.data_path = existing_paths  # 
                config.datamodule.load_multiple_files = True
                print(f"📊 ({len(existing_paths)} ):")
                for p in existing_paths:
                    print(f"   - {Path(p).name}")
            else:
                raise FileNotFoundError(
                    f"!\n"
                    f":\n"
                    f"  - {merged_file}\n"
                    f"  - :\n" + 
                    "\n".join(f"    - {f}" for f in data_files)
                )
    
    return train(config)

if __name__ == "__main__":
    main()

