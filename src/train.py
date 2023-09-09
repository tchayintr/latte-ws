import argparse
import datetime
import numpy as np
import os
import pytorch_lightning as pl
from pytorch_lightning.callbacks import (ModelCheckpoint, LearningRateMonitor)
from pytorch_lightning.strategies import DDPStrategy
from pytorch_lightning.loggers import TensorBoardLogger
from lightning_lite.utilities.seed import seed_everything
import random
import torch
from transformers import set_seed

from taggers.bert_tagger import BertTagger
from taggers.lstm_tagger import LSTMTagger
from taggers.tagger import Tagger

os.environ['TOKENIZERS_PARALLELISM'] = 'false'


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--num-gpus', type=int, default=1)
    parser.add_argument('--ddp-timeout', type=int, default=7200)
    parser.add_argument('--overfit-batches', type=float, default=0.0)
    parser.add_argument('--ckpt-path',
                        help='Specify a checkpoint path to resume training')
    parser.add_argument('--normalize-unicode', action='store_true')
    parser.add_argument('--criterion-type',
                        choices=['crf'],
                        default='crf')
    parser.add_argument(
        '--metric-type',
        choices=['word-bin', 'word-bin-th'],
        default='word-bin')
    parser.add_argument('--lang',
                        choices=['ja', 'th', 'zh'])
    parser.add_argument('--use-binary', action='store_true')
    parser.add_argument('--seed', type=int, default=None)
    parser.add_argument('--run', choices=['bert', 'lstm'], default='bert')
    parser = pl.Trainer.add_argparse_args(parser)
    parser = Tagger.add_model_specific_args(parser)
    parser = BertTagger.add_model_specific_args(parser)
    parser = LSTMTagger.add_model_specific_args(parser)
    args = parser.parse_args()
    return args

def set_seeds(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    g = torch.Generator()
    g.manual_seed(seed)
    set_seed(seed)
    seed_everything(seed)


def run(args):
    if args.seed is not None:
        set_seeds(args.seed)

    if args.run == 'lstm':
        print('Model: LSTM')
        model = LSTMTagger(args)
    else:
        print('Model: BERT')
        model = BertTagger(args)

    if args.ckpt_path:
        print('Resume: {}'.format(args.ckpt_path))

    checkpoint_callback = ModelCheckpoint(monitor='valid/micro-F1',
                                          mode='max',
                                          save_top_k=1,
                                          verbose=True)
    lr_callback = LearningRateMonitor(logging_interval='epoch')
    logger = TensorBoardLogger(save_dir=args.save_dir,
                               name=args.model_name,
                               version=args.model_version,
                               default_hp_metric=False)

    trainer = pl.Trainer.from_argparse_args(
        args,
        accelerator='gpu',
        devices=args.num_gpus,
        default_root_dir=args.save_dir,
        callbacks=[checkpoint_callback, lr_callback],
        strategy=DDPStrategy(find_unused_parameters=True,
                             timeout=datetime.timedelta(
                                 seconds=args.ddp_timeout))
        if args.num_gpus > 1 else None,
        replace_sampler_ddp=(False if args.num_gpus > 1 else True),
        overfit_batches=args.overfit_batches,
        logger=logger)

    trainer.fit(model, ckpt_path=args.ckpt_path)
    print('best model: {}'.format(checkpoint_callback.best_model_path))

    if args.test_file:
        print('test: {}'.format(checkpoint_callback.best_model_path))
        trainer.test(ckpt_path=checkpoint_callback.best_model_path)


if __name__ == '__main__':
    args = get_args()
    run(args)
