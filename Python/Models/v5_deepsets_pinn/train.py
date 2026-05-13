"""Wrapper de pl.Trainer.fit para v5_deepsets_pinn.

Importante: este modelo usa autograd dentro del step, por lo que NO se puede
correr con `precision='16-mixed'` directamente sin cuidado extra (los second-
order grad pueden fallar en fp16). Default queda en fp32.

Si `train_div / train_curl` se quedan en ~1e-18 desde epoch 0 (el bug que
mostró v2), significa que la red está ignorando `query_xyz` o que los
gradientes están saturando. Diagnóstico rápido: hacer un forward manual con
dos query_xyz distintos y verificar que `B_pred` cambia.
"""
import pytorch_lightning as pl
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from pytorch_lightning.loggers import CometLogger


def train(lit_model, loader_tr, loader_va, *,
          n_epochs, patience, ckpt_dir, run_tag,
          accelerator="auto", devices="auto",
          deterministic=True,
          gradient_clip_val=None,
          comet_project="pmdkernel",
          log_every_n_steps=50):
    early_stop = EarlyStopping(monitor="val_loss", patience=patience, mode="min")
    ckpt = ModelCheckpoint(
        dirpath=str(ckpt_dir),
        filename=run_tag,
        monitor="val_loss",
        mode="min",
        save_top_k=1,
        save_last=False,
    )
    logger = CometLogger(project=comet_project, name=run_tag)
    logger.experiment.add_tags(["v5_deepsets_pinn", "physics", "div_curl"])

    trainer = pl.Trainer(
        max_epochs=n_epochs,
        callbacks=[early_stop, ckpt],
        logger=logger,
        accelerator=accelerator,
        devices=devices,
        log_every_n_steps=log_every_n_steps,
        deterministic=deterministic,
        gradient_clip_val=gradient_clip_val,
    )
    trainer.fit(lit_model, loader_tr, loader_va)
    return trainer
