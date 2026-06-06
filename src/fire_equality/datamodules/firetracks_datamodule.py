"""
FireTracks DataModule for PyTorch Lightning
用于加载 processed_firetracks_pixel_binary.pth 数据
"""

import os
import torch
from pathlib import Path
from torch.utils.data import DataLoader, random_split
from pytorch_lightning import LightningDataModule

from .firetracks_loader import FireTracksDataset


class FireTracksBinaryDataModule(LightningDataModule):
    """FireTracks二分类数据模块"""
    
    def __init__(
        self,
        data_path: str = 'processed_firetracks_pixel_binary.pth',
        batch_size: int = 32,
        train_ratio: float = 0.7,
        val_ratio: float = 0.15,
        test_ratio: float = 0.15,
        num_workers: int = 0,
        seed: int = 42,
        load_multiple_files: bool = False  # 是否支持多文件加载
    ):
        super().__init__()
        # 支持单个文件路径或文件列表
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
        
        # 验证比例
        assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6, \
            f"比例之和必须为1.0，当前: {train_ratio + val_ratio + test_ratio}"
        
        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None
    
    def _find_data_file(self, data_path):
        """智能查找数据文件路径"""
        # 如果是绝对路径，直接返回
        if os.path.isabs(data_path):
            if os.path.exists(data_path):
                return data_path
            else:
                raise FileNotFoundError(f"数据文件不存在: {data_path}")
        
        # 获取项目根目录（FireEqual）
        current_file = os.path.abspath(__file__)
        project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(current_file))))
        
        # 尝试的路径列表（按优先级排序）
        possible_paths = []
        
        # 如果路径包含相对路径符号，尝试解析（优先）
        if '..' in data_path or '/' in data_path or '\\' in data_path:
            resolved = os.path.normpath(os.path.join(project_root, data_path))
            possible_paths.append(resolved)
        
        # 尝试 data/processed 或 dataset 文件夹
        if 'data/processed' not in data_path and 'dataset' not in data_path:
            for sub in ('data/processed', 'dataset'):
                possible_paths.append(os.path.join(project_root, sub, os.path.basename(data_path)))
        elif 'dataset' not in data_path and not data_path.startswith('data/'):
            possible_paths.append(os.path.join(project_root, 'data/processed', os.path.basename(data_path)))
        
        # 项目根目录
        possible_paths.append(os.path.join(project_root, os.path.basename(data_path)))
        
        # 也尝试当前工作目录（Hydra可能会改变工作目录）
        possible_paths.extend([
            os.path.join(os.getcwd(), data_path),  # 当前工作目录
            os.path.join(os.getcwd(), os.path.basename(data_path)),  # 当前目录的文件名
        ])
        
        # 查找存在的文件
        for path in possible_paths:
            abs_path = os.path.abspath(path)
            if os.path.exists(abs_path) and os.path.isfile(abs_path):
                return abs_path
        
        # 如果都找不到，抛出详细的错误信息
        error_msg = f"找不到数据文件 '{data_path}'。尝试的路径:\n"
        for p in possible_paths:
            abs_p = os.path.abspath(p)
            exists = "✓" if os.path.exists(abs_p) else "✗"
            error_msg += f"  {exists} {abs_p}\n"
        error_msg += f"\n项目根目录: {project_root}\n"
        error_msg += f"当前工作目录: {os.getcwd()}\n"
        raise FileNotFoundError(error_msg)
    
    def _load_single_file(self, data_path):
        """加载单个数据文件"""
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
        
        # PyTorch 2.6+ 需要允许omegaconf对象
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
            raise ValueError(f"数据文件 '{data_path}' 中没有找到 'spatiotemporal_samples'")
        
        return data['spatiotemporal_samples']
    
    def setup(self, stage=None):
        """加载数据并划分数据集"""
        if self.train_dataset is not None:
            # 已经设置过了
            return
        
        # 检查 data_path 是否为列表类型
        from omegaconf import ListConfig
        is_list = isinstance(self.data_path, (list, tuple, ListConfig))
        
        # 确定要加载的文件列表
        if self.load_multiple_files or is_list:
            # 多文件模式：加载多个文件
            # 获取文件列表（可能是 self.data_paths 或 self.data_path）
            if self.data_paths:
                file_list = self.data_paths
            elif is_list:
                file_list = self.data_path
            else:
                raise ValueError("多文件模式需要提供文件列表")
            
            # 查找所有文件路径
            data_files = []
            for path in file_list:
                path_str = str(path)  # 转换为字符串
                found_path = self._find_data_file(path_str)
                data_files.append(found_path)
            
            print(f"📊 多文件模式：将加载 {len(data_files)} 个数据文件")
            
            # 流式加载所有文件（避免内存爆炸）
            all_samples = []
            import gc
            for i, data_file in enumerate(data_files):
                print(f"   [{i+1}/{len(data_files)}] 加载: {os.path.basename(data_file)}")
                samples = self._load_single_file(data_file)
                all_samples.extend(samples)
                print(f"      已加载 {len(samples):,} 个样本，累计 {len(all_samples):,} 个样本")
                # 释放内存
                del samples
                gc.collect()
            
            spatiotemporal_samples = all_samples
            print(f"   ✅ 总共加载 {len(spatiotemporal_samples):,} 个样本")
            
        else:
            # 单文件模式：加载单个文件
            data_path_str = str(self.data_path)  # 确保是字符串
            data_path = self._find_data_file(data_path_str)
            print(f"📊 加载数据文件: {data_path}")
            spatiotemporal_samples = self._load_single_file(data_path)
            print(f"   ✅ 总样本数: {len(spatiotemporal_samples):,}")
        
        # 创建完整数据集
        full_dataset = FireTracksDataset(
            spatiotemporal_samples,
            target_type='binary_classification'
        )
        
        # 划分数据集
        total_size = len(full_dataset)
        train_size = int(total_size * self.train_ratio)
        val_size = int(total_size * self.val_ratio)
        test_size = total_size - train_size - val_size
        
        self.train_dataset, self.val_dataset, self.test_dataset = random_split(
            full_dataset,
            [train_size, val_size, test_size],
            generator=torch.Generator().manual_seed(self.seed)
        )
        
        print(f"   📊 数据集划分:")
        print(f"      - 训练集: {len(self.train_dataset)} 样本 ({self.train_ratio*100:.1f}%)")
        print(f"      - 验证集: {len(self.val_dataset)} 样本 ({self.val_ratio*100:.1f}%)")
        print(f"      - 测试集: {len(self.test_dataset)} 样本 ({self.test_ratio*100:.1f}%)")
    
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

