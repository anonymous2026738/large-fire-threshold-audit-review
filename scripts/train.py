"""
训练数据集（2002-2020）
使用 SimpleConvLSTM 模型进行训练：
- 训练任务：按 loss（NLLLoss）优化，EarlyStopping 按 val/loss。
- Grid search / multirun 选最佳 model：按 val/auc（主 ModelCheckpoint monitor=val/auc）。
- 公平性分析：val/auc 与 F1-score 同时作为对比指标（不同社会经济群体）。
"""
import os
import sys
from pathlib import Path

# 添加 src 目录到路径（公开 release 布局）
_release_root = Path(__file__).resolve().parent.parent
src_dir = _release_root / 'src'
if str(src_dir) not in sys.path:
    sys.path.insert(0, str(src_dir))

# 保存原始工作目录（Hydra会改变工作目录）
original_cwd = os.getcwd()

# 导入训练函数
from fire_equality.train import train
import hydra
from omegaconf import DictConfig

@hydra.main(config_path="../configs", config_name="config", version_base=None)
def main(config: DictConfig):
    """主函数，使用Hydra装饰器"""
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
    
    # 更新数据路径（支持单文件或多文件模式）
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
        
        # Release 根目录（与 Hydra 工作目录无关）
        project_root = Path(__file__).resolve().parent.parent
        
        # 检查是否存在合并后的文件
        merged_file = project_root / 'data/processed/processed_firetracks_pixel_binary_2002-2020.pth'
        
        # 如果合并文件存在，使用合并文件（更快）
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
            
            # 禁用 struct 模式以允许添加新键
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
            
            print(f"📊 使用合并后的数据集: {merged_file}")
        else:
            # 否则使用多文件模式（流式加载，避免内存爆炸）
            data_files = [
                'data/processed/processed_firetracks_pixel_binary_2002-2004.pth',
                'data/processed/processed_firetracks_pixel_binary_2005-2007.pth',
                'data/processed/processed_firetracks_pixel_binary_2008-2009.pth',
                'data/processed/processed_firetracks_pixel_binary_2010-2013.pth',
                'data/processed/processed_firetracks_pixel_binary_2014-2015.pth',
                'data/processed/processed_firetracks_pixel_binary_2016-2018.pth',
                'data/processed/processed_firetracks_pixel_binary_2019-2020.pth',
            ]
            # 转换为绝对路径
            data_paths = [str(project_root / f) for f in data_files]
            # 检查文件是否存在
            existing_paths = [p for p in data_paths if Path(p).exists()]
            
            if existing_paths:
                # 禁用 struct 模式以允许添加新键
                from omegaconf import OmegaConf
                OmegaConf.set_struct(config.datamodule, False)
                
                config.datamodule.data_path = existing_paths  # 传递列表
                config.datamodule.load_multiple_files = True
                print(f"📊 使用多文件模式（{len(existing_paths)} 个文件）:")
                for p in existing_paths:
                    print(f"   - {Path(p).name}")
            else:
                raise FileNotFoundError(
                    f"找不到数据文件！\n"
                    f"请确保以下文件之一存在：\n"
                    f"  - {merged_file}\n"
                    f"  - 或者以下文件中的至少一个：\n" + 
                    "\n".join(f"    - {f}" for f in data_files)
                )
    
    return train(config)

if __name__ == "__main__":
    main()

