import argparse
import os
from pathlib import Path
import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger
from lightning_lite.utilities.seed import seed_everything
import sys

from taggers.bert_tagger import BertTagger
from taggers.lstm_tagger import LSTMTagger
from taggers.tagger import Tagger

os.environ['TOKENIZERS_PARALLELISM'] = 'false'


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--num-gpus', type=int, default=1)
    parser.add_argument('--ddp-timeout', type=int, default=3600)
    parser.add_argument('--ckpt-path',
                        required=True,
                        type=Path,
                        help='Specify a checkpoint path to resume training')
    parser.add_argument('--normalize-unicode', action='store_true')
    parser.add_argument('--lang', choices=['ja', 'th', 'zh'])
    parser.add_argument('--use-binary', action='store_true')
    parser.add_argument('--seed', type=int, default=None)
    parser.add_argument('--run', choices=['bert', 'lstm'], default='bert')
    parser = pl.Trainer.add_argparse_args(parser)
    parser = Tagger.add_model_specific_args(parser)
    parser = BertTagger.add_model_specific_args(parser)
    parser = LSTMTagger.add_model_specific_args(parser)
    args = parser.parse_args()
    return args


def check_args(args):
    error_msgs = []
    if args.test_file is None:
        error_msgs.append('Error: {} is required'.format('--test_file'))
    if args.ckpt_path is None:
        error_msgs.append('Error: {} is required'.format('--ckpt_path'))

    if error_msgs:
        for error_msg in error_msgs:
            print(error_msg)
        sys.exit()


def write_data(data, path, prefix=None):
    if prefix is not None:
        if not os.path.exists(prefix):
            os.makedirs(prefix)
        filepath = prefix / path
    else:
        filepath = path

    with open(filepath, mode='w', encoding='utf8') as f:
        for d in data:
            print(d, file=f)


def make_decode_filepath(decode_root_dir, model_name, model_version, suffix):
    prefix = decode_root_dir / '{}/version_{}/'.format(model_name,
                                                       model_version)
    filename = '{}.decode'.format(suffix)
    return prefix, filename


def run(args):
    check_args(args)

    if args.seed is not None:
        seed_everything(args.seed)

    print('Load: {}'.format(args.ckpt_path))
    if args.run == 'lstm':
        print('Model: LSTM')
        model = LSTMTagger.load_from_checkpoint(args.ckpt_path)
    else:
        print('Model: BERT')
        model = BertTagger.load_from_checkpoint(args.ckpt_path)

    model.eval()
    logger = TensorBoardLogger(save_dir=args.save_dir,
                               name=args.model_name,
                               version=args.model_version,
                               default_hp_metric=False)

    trainer = pl.Trainer.from_argparse_args(args,
                                            accelerator='gpu',
                                            devices=args.num_gpus,
                                            default_root_dir=args.save_dir,
                                            logger=logger)

    print('decode: {}'.format(args.test_file))
    preds = trainer.predict(model, return_predictions=True)
    res = [pred_str for pred in preds for pred_str in pred]
    prefix, filename = make_decode_filepath(args.decode_save_dir,
                                            args.model_name,
                                            args.model_version,
                                            args.ckpt_path.stem)
    decode_filepath = prefix / filename

    print('written: {}'.format(decode_filepath))
    write_data(res, filename, prefix)


if __name__ == '__main__':
    args = get_args()
    run(args)
