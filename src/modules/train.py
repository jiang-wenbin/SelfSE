import os
import time
from pathlib import Path
from datetime import datetime
import lightning as L
from lightning.pytorch.loggers import TensorBoardLogger
from lightning.pytorch.callbacks import ModelCheckpoint, LearningRateMonitor, TQDMProgressBar, EarlyStopping
from lightning.pytorch.utilities.model_summary import ModelSummary
from omegaconf import OmegaConf
from rich import print as rprint
import alpha.enh.system as enh_system
from .utils.init_conf import get_exp_name
from modules.utils.logging import logger
from modules.utils.common import format_time, average_model, merge_dicts


class CustomEarlyStopping(EarlyStopping):
    def __init__(self, monitor, patience=3, verbose=False, mode='min', **kwargs):
        super().__init__(monitor=monitor, patience=patience, verbose=verbose, 
                         mode=mode, **kwargs)

    def load_state_dict(self, callback_state):
        self.wait_count = callback_state['wait_count']
        self.stopped_epoch = callback_state['stopped_epoch']
        self.best_score = callback_state['best_score']
        # self.patience = callback_state['patience'] # don't load patience


class LoggingCallback(L.Callback):
    def __init__(self, log_every_n_steps=100, log_every_n_epochs=1):
        super().__init__()
        self.log_every_n_steps = log_every_n_steps
        self.log_every_n_epochs = log_every_n_epochs
        self.epoch_start_time = None

    def on_train_epoch_start(self, trainer, pl_module):
        self.epoch_start_time = time.perf_counter()

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):        
        if (batch_idx + 1) % self.log_every_n_steps == 0:
            metrics = trainer.callback_metrics
            if metrics:
                info = ', '.join([f'{k}: {v:.2f}' for k, v in metrics.items() if v is not None and 'train' in k])
                logger.info(f'Epoch {trainer.current_epoch}, {batch_idx + 1}/{trainer.num_training_batches}: {info}')

    def on_train_epoch_end(self, trainer, pl_module):
        if (trainer.current_epoch % self.log_every_n_epochs != 0) or self.epoch_start_time is None:
            return
        epoch_time = time.perf_counter() - self.epoch_start_time
        time_str = format_time(epoch_time)
        metrics = trainer.callback_metrics
        if metrics:
            info = ', '.join([
                f'{k}: {v:.2e}' if k.startswith('lr') else f'{k}: {v:.2f}' 
                for k, v in metrics.items() if v is not None])
            logger.info(f'Epoch {trainer.current_epoch}/{trainer.max_epochs}: {info} | time: {time_str}')

    def on_validation_epoch_end(self, trainer, pl_module):
        if trainer.sanity_checking:  # Sanity checking phase, don't logging
            return
        if trainer.val_check_interval == 1.0: # float [0.0, 1.0] or int (default: 1.0, don't logging)
            return
        metrics = trainer.callback_metrics
        if metrics:
            info = ', '.join(
                [f'{k}: {v:.2f}' for k, v in metrics.items() if v is not None and 'valid' in k])
            logger.info(f'Epoch {trainer.current_epoch}: {info}')


def get_model(conf):
    name = conf['system']['name']
    # Load model based on system name
    for system in [enh_system]:
        if hasattr(system, name):
            model = getattr(system, name)(conf=conf)
            break
    else:
        raise ValueError(f'system name error: {name}')
    # Try to initialize model from ckpt
    model.init_model(OmegaConf.to_container(conf['model']))
    return model


def get_callbacks(conf):
    callbacks = [LearningRateMonitor(logging_interval='step')]
    callbacks_conf = conf.get('trainer_callbacks')
    if callbacks_conf:
        model_checkpoint = callbacks_conf.get('model_checkpoint')
        if model_checkpoint:
            callbacks.append(ModelCheckpoint(Path(conf['root_dir'])/'checkpoints',
                                             **model_checkpoint))
        early_stop = callbacks_conf.get('early_stop')
        if early_stop:
            callbacks.append(CustomEarlyStopping(**early_stop))
        progress_bar = callbacks_conf.get('progress_bar')
        if progress_bar:
            callbacks.append(TQDMProgressBar(**progress_bar))
        progress_logger = callbacks_conf.get('progress_logger')
        if progress_logger:
            default_kwargs = {"log_every_n_steps": 200, "log_every_n_epochs": 1}
            kwargs = default_kwargs if progress_logger is True else \
                merge_dicts(default_kwargs, progress_logger)
            callbacks.append(LoggingCallback(**kwargs))
    return callbacks


# train model
def train(conf):
    exp_name = get_exp_name(conf)
    tb_logger = TensorBoardLogger(save_dir=conf['system']['log_dir'], name=exp_name, 
                                  version=conf['version'], default_hp_metric=False)
    root_dir = conf['root_dir']
    logger.info('root_dir: ' + root_dir)
    callbacks = get_callbacks(conf)
    model = get_model(conf)
    trainer = L.Trainer(default_root_dir=root_dir, logger=tb_logger, 
                         callbacks=callbacks, **conf['trainer'])
    trainer.fit(model, ckpt_path=conf['ckpt'])
    if os.path.exists(root_dir):
        for item in callbacks:
            if isinstance(item, ModelCheckpoint):
                item.to_yaml(os.path.join(root_dir, 'best_k_models.yaml'))
    if conf['system'].get('train_and_test', False):
        trainer.test(model)
    logger.info('[End] {} {} {}'.format('-'*10, datetime.now().strftime('%Y-%m-%d %H:%M:%S.%f')[:-3], '-'*10))


def test(conf):
    if conf.get('avg_model') is not None:
        avg_ckpt = average_model(conf)
        if avg_ckpt:
            conf['ckpt'] = str(avg_ckpt)
    trainer = L.Trainer(logger=False, **conf['trainer'])
    model = get_model(conf)
    trainer.test(model, ckpt_path=conf['ckpt'])


def valid(conf):
    assert conf['ckpt']
    trainer = L.Trainer(logger=False, **conf['trainer'])
    model = get_model(conf)
    trainer.validate(model, ckpt_path=conf['ckpt'])  


def model_summary(conf):
    model = get_model(conf)
    rprint(ModelSummary(model, max_depth=conf.get('max_depth', 1)))
    