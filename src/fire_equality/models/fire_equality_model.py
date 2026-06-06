"""
Fire Equality ConvLSTM Model
PyTorch LightningConvLSTMmodel,, 
configuration
"""

from typing import Any, List

import torch
from pytorch_lightning import LightningModule
from torchmetrics import AUROC, AveragePrecision, F1Score
from torchmetrics.classification.accuracy import Accuracy

try:
    from .modules.fire_modules import SimpleConvLSTM
except ImportError:
    # 
    from models.modules.fire_modules import SimpleConvLSTM


def combine_dynamic_static_inputs(dynamic, static, clc, access_mode):
    """
   , CLC
    
    Args:
        dynamic: 
        static: 
        clc: 
        access_mode:  ('spatial', 'temporal', 'spatiotemporal')
    
    Returns:
        input tensor
    """
    assert access_mode in ['spatial', 'temporal', 'spatiotemporal']
    if access_mode == 'spatial':
        dynamic = dynamic.float()
        static = static.float()
        input_list = [dynamic, static]
        inputs = torch.cat(input_list, dim=1)
    elif access_mode == 'temporal':
        bsize, timesteps, _ = dynamic.shape
        static = static.unsqueeze(dim=1)
        repeat_list = [1 for _ in range(static.dim())]
        repeat_list[1] = timesteps
        static = static.repeat(repeat_list)
        input_list = [dynamic, static]
        if clc is not None:
            clc = clc.unsqueeze(dim=1).repeat(repeat_list)
            input_list.append(clc)
        inputs = torch.cat(input_list, dim=2).float()
    else:
        bsize, timesteps, _, _, _ = dynamic.shape
        static = static.unsqueeze(dim=1)
        repeat_list = [1 for _ in range(static.dim())]
        repeat_list[1] = timesteps
        static = static.repeat(repeat_list)
        input_list = [dynamic, static]
        if clc is not None:
            clc = clc.unsqueeze(dim=1).repeat(repeat_list)
            input_list.append(clc)
        inputs = torch.cat(input_list, dim=2).float()
    return inputs


class ConvLSTM_fire_equality_model(LightningModule):
    """
    Fire Equality ConvLSTMmodel
    
    PyTorch Lightning,PyTorch5:
        - Computations (init)
        - Train loop (training_step)
        - Validation loop (validation_step)
        - Test loop (test_step)
        - Optimizers (configure_optimizers)
    """

    def __init__(
            self,
            dynamic_features=None,
            static_features=None,
            hidden_size: int = 32,
            lstm_layers: int = 1,
            lr: float = 0.001,
            positive_weight: float = 0.5,
            lr_scheduler_step: int = 10,
            lr_scheduler_gamma: float = 0.1,
            weight_decay: float = 0.0005,
            dropout: float = 0.5,
            access_mode='spatiotemporal',
            clc='vec'
    ):
        super().__init__()
        #,self.hparams
        self.save_hyperparameters()

        # initializemodel
        self.model = SimpleConvLSTM(hparams=self.hparams)

        # :,
        self.criterion = torch.nn.NLLLoss(weight=torch.tensor([1. - positive_weight, positive_weight]))
        
        #, 
        # epoch

        # Accuracy, AUROC, AUPRC, F1
        self.train_accuracy = Accuracy(task='binary')
        self.train_auc = AUROC(task='binary')
        self.train_auprc = AveragePrecision(task='binary')
        self.train_f1 = F1Score(task='binary')

        self.val_accuracy = Accuracy(task='binary')
        self.val_auc = AUROC(task='binary')
        self.val_auprc = AveragePrecision(task='binary')
        self.val_f1 = F1Score(task='binary')

        self.test_accuracy = Accuracy(task='binary')
        self.test_auc = AUROC(task='binary')
        self.test_auprc = AveragePrecision(task='binary')
        self.test_f1 = F1Score(task='binary')

    def forward(self, x: torch.Tensor):
        """"""
        return self.model(x)

    def step(self, batch: Any):
        """
        //
        
        Args:
            batch: 
                - 4: (dynamic, static, clc, y) - 
                - 2: (features, y) - (FireTracksDataset)
        
        Returns:
            loss, preds, preds_proba, y
        """
        # 
        if len(batch) == 2:
            # : (features, y)
            # features shape: [B, C, T, H, W]
            features, y = batch
            y = y.long()
            
            # featuresmodel [B, T, C, H, W]
            # features [B, C, T, H, W], [B, T, C, H, W]
            features = features.permute(0, 2, 1, 3, 4)  # [B, C, T, H, W] -> [B, T, C, H, W]
            
            #,features
            # model [B, T, C, H, W]
            logits = self.forward(features)
        else:
            # : (dynamic, static, clc, y)
            dynamic, static, clc, y = batch
            y = y.long()
            if not self.hparams.get('clc', False):
                clc = None
            inputs = combine_dynamic_static_inputs(dynamic, static, clc, 'spatiotemporal')
            logits = self.forward(inputs)
        
        loss = self.criterion(logits, y)
        preds = torch.argmax(logits, dim=1)
        preds_proba = torch.exp(logits)[:, 1]
        return loss, preds, preds_proba, y

    def training_step(self, batch: Any, batch_idx: int):
        """"""
        loss, preds, preds_proba, targets = self.step(batch)

        # 
        self.train_accuracy.update(preds, targets)
        self.train_auc.update(preds_proba, targets)
        self.train_auprc.update(preds_proba, targets)
        self.train_f1.update(preds, targets)

        # 
        self.log("train/loss", loss, on_step=False, on_epoch=True, prog_bar=False)
        self.log("train/acc", self.train_accuracy, on_step=False, on_epoch=True, prog_bar=False)
        self.log("train/auc", self.train_auc, on_step=False, on_epoch=True, prog_bar=False)
        self.log("train/auprc", self.train_auprc, on_step=False, on_epoch=True, prog_bar=False)
        self.log("train/f1", self.train_f1, on_step=False, on_epoch=True, prog_bar=False)
        return {"loss": loss, "preds": preds, "targets": targets}

    def on_train_epoch_end(self):
        """ epoch (Lightning 2.x  training_epoch_end)"""
        pass

    def validation_step(self, batch: Any, batch_idx: int):
        """"""
        loss, preds, preds_proba, targets = self.step(batch)

        # 
        self.val_accuracy.update(preds, targets)
        self.val_auc.update(preds_proba, targets)
        self.val_auprc.update(preds_proba, targets)
        self.val_f1.update(preds, targets)

        # 
        self.log("val/loss", loss, on_step=False, on_epoch=True, prog_bar=False)
        self.log("val/acc", self.val_accuracy, on_step=False, on_epoch=True, prog_bar=False)
        self.log("val/auc", self.val_auc, on_step=False, on_epoch=True, prog_bar=False)
        self.log("val/auprc", self.val_auprc, on_step=False, on_epoch=True, prog_bar=False)
        self.log("val/f1", self.val_f1, on_step=False, on_epoch=True, prog_bar=False)
        return {"loss": loss, "preds": preds, "targets": targets, "preds_proba": preds_proba}

    def on_validation_epoch_end(self):
        """ epoch (Lightning 2.x  validation_epoch_end)"""
        pass

    def test_step(self, batch: Any, batch_idx: int):
        """"""
        loss, preds, preds_proba, targets = self.step(batch)

        # 
        self.test_accuracy.update(preds, targets)
        self.test_auc.update(preds_proba, targets)
        self.test_auprc.update(preds_proba, targets)
        self.test_f1.update(preds, targets)

        # 
        self.log("test/loss", loss, on_step=False, on_epoch=True, prog_bar=False)
        self.log("test/acc", self.test_accuracy, on_step=False, on_epoch=True, prog_bar=False)
        self.log("test/auc", self.test_auc, on_step=False, on_epoch=True, prog_bar=False)
        self.log("test/auprc", self.test_auprc, on_step=False, on_epoch=True, prog_bar=False)
        self.log("test/f1", self.test_f1, on_step=False, on_epoch=True, prog_bar=False)
        return {"loss": loss, "preds": preds, "targets": targets}

    def on_test_epoch_end(self):
        """ epoch (Lightning 2.x  test_epoch_end)"""
        pass

    def configure_optimizers(self):
        """
        configuration
        
        :NLLLoss ()
        :Adam
        :StepLR ()
        
        Returns:
            dict: optimizerlr_scheduler
        """
        # Adam
        optimizer = torch.optim.Adam(
            params=self.parameters(), 
            lr=self.hparams.lr, 
            weight_decay=self.hparams.weight_decay
        )

        # StepLR
        lr_scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer, 
            step_size=self.hparams.lr_scheduler_step,
            gamma=self.hparams.lr_scheduler_gamma
        )
        return {'optimizer': optimizer, 'lr_scheduler': lr_scheduler}

