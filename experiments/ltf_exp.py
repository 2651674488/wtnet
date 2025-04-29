import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from data_provider.data_factory import data_provider
from task_modules.ltf_module import LTFModule
from configs.ltf_config import *
from utils.tools import get_csv_logger


def ltf_experiment(config,gpus):
    pl.seed_everything(2025)
    train_dataset, train_dl = data_provider(config, "train")
    val_dataset, val_dl = data_provider(config, "val")
    test_dataset, test_dl = data_provider(config, "test")

    class DataModule(pl.LightningDataModule):
        def __init__(self, train_dl, val_dl, test_dl):
            super().__init__()
            self.train_dl = train_dl
            self.val_dl = val_dl
            self.test_dl = test_dl
            self.train_dataset = train_dataset
            self.val_dataset = val_dataset
            self.test_dataset = test_dataset

        def train_dataloader(self):
            return self.train_dl

        def val_dataloader(self):
            return self.val_dl

        def test_dataloader(self):
            return self.test_dl

    data_module = DataModule(train_dl, val_dl, test_dl)
    model = LTFModule(config)
    monitor_metric = "val_mse"
    callbacks = []
    ckpt_callback = ModelCheckpoint(monitor=monitor_metric,
                                    save_top_k=1,
                                    mode="min")
    callbacks.append(ckpt_callback)
    es_callback = EarlyStopping(monitor=monitor_metric,
                                mode="min",
                                patience=10)
    callbacks.append(es_callback)
    logger = get_csv_logger("logs/ltf",
                            name=f"{config.name}_{config.pred_len}")
    trainer = pl.Trainer(devices=[gpus],
                         accelerator="gpu",
                         precision="32-true",
                         callbacks=callbacks,
                         logger=logger,
                         max_epochs=40,
                         gradient_clip_val=config.grad_clip_val)
    trainer.fit(model, datamodule=data_module)
    model = LTFModule.load_from_checkpoint(ckpt_callback.best_model_path)
    trainer.test(model, datamodule=data_module)


def run_ltf(args):
    dataset_dict = {}
    dataset_dict['etth1'] = ETTh1_LTFConfig
    dataset_dict['etth2'] = ETTh2_LTFConfig
    dataset_dict['ettm1'] = ETTm1_LTFConfig
    dataset_dict['ettm2'] = ETTm2_LTFConfig
    dataset_dict['ecl'] = ECL_LTFConfig
    dataset_dict['traffic'] = Traffic_LTFConfig
    dataset_dict['weather'] = Weather_LTFConfig
    dataset_dict['exchange'] = Exchange_LTFConfig
    datasets = []
    dataset_args = set([str.lower(d) for d in args.dataset])
    if "all" in dataset_args:
        datasets += [
            ETTh1_LTFConfig, ETTh2_LTFConfig, ETTm1_LTFConfig,
            ETTm2_LTFConfig, Traffic_LTFConfig, Weather_LTFConfig,
            Exchange_LTFConfig
        ]
    else:
        datasets += [dataset_dict[d] for d in dataset_args]
    pred_lens = []
    pred_len_args = set([str.lower(p) for p in args.pred_len])
    if "all" in pred_len_args:
        pred_lens += [96, 192, 336, 720]
    else:
        pred_lens += [int(d) for d in pred_len_args]
    for d in datasets:
        for p in pred_lens:
            config = d(p)
            print(f"dataset:{config.name},pred_len:{config.pred_len}")
            ltf_experiment(config, args.gpus)
