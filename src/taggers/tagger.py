import abc
import argparse
from pathlib import Path
import pytorch_lightning as pl
import torch
import torch.nn as nn
from typing import Any

from core.modules.crf import CRF
from utils.word_metrics import WORD_F1
from utils.word_oov_cws_metrics import CWS_WORD_F1, CWS_OOV_RECALL
from utils.word_tws_metrics import TWS_WORD_F1
from utils.binary_metrics import BINARY_F1
from utils.data.datamodule import DataModule
from utils.data.dataset import pad_sequence

ORG_INPUT_KEY = 'org_ids'
LATTICE_INPUT_KEY = 'lattice'
MAPPING_TABLE_KEY = 'mapping'


class Tagger(pl.LightningModule):

    def __init__(self, hparams, *args, **kwargs):
        super(Tagger, self).__init__()
        self.save_hyperparameters(hparams)

        if ((self.hparams.unuse_attn and not self.hparams.unuse_gnn)
                and not self.hparams.generate_unigram_node):
            raise AssertionError(
                '`--unuse-attn` but not `--unuse-gnn` requires ' +
                '`--generate-unigram-node`')

        if (self.hparams.metric_type == 'word-bin-th'
                and not self.hparams.lang == 'th'):
            raise AssertionError(
                '`--metric-type=word-bin-th` supports `--lang=th`')

        if (self.hparams.metric_type == 'word-bin'
                and self.hparams.lang == 'th'):
            raise AssertionError(
                '`--metric-type=word-bin` supports `--lang=[zh, ja]`')

        self.data_module = None
        self._setup_data_module()

        self.train_set = None
        self.valid_set = None
        self.test_set = None

        self.criterion = None
        self._setup_criterion()

        self.metric = None
        self.oov_metric = None
        self.bin_metric = None
        self._setup_metric()

    @abc.abstractmethod
    def forward(self) -> Any:
        pass

    @abc.abstractmethod
    def training_step(self, batch, batch_idx=None):
        pass

    @abc.abstractmethod
    def validation_step(self, batch, batch_idx=None):
        pass

    @abc.abstractmethod
    def test_step(self, batch, batch_idx=None):
        pass

    @abc.abstractmethod
    def predict_step(self, batch, batch_idx=None):
        pass

    @abc.abstractmethod
    def validation_epoch_end(self, outputs):
        pass

    @abc.abstractmethod
    def test_epoch_end(self, outputs):
        pass

    @abc.abstractmethod
    def configure_optimizers(self):
        pass

    @abc.abstractmethod
    def _get_token_embeddings(self, token_ids: torch.Tensor):
        pass

    def on_train_start(self):
        self.logger.log_hyperparams(self.hparams, {
            'hp/': 0,
        })

    def _setup_data_module(self):
        params = self._get_data_module_params()
        data_module = DataModule(**params)
        data_module.setup()
        self.data_module = data_module

    def _setup_criterion(self):
        params = self._get_criterion_params()
        if params['criterion_type'] == 'crf':
            self.criterion = CRF(constraint_type='BMES',
                                 ignore_index_for_mask=0)
        else:
            self.criterion = nn.CrossEntropyLoss(ignore_index=0)

    def _setup_metric(self):
        params = self._get_metric_params()
        dist_sync_on_step = True if self.hparams.num_gpus > 1 else False
        if params['metric_type'] == 'word-bin':
            self.metric = CWS_WORD_F1(ignore_index=-1,
                                      constraint_type='BMES',
                                      dist_sync_on_step=dist_sync_on_step)
            self.bin_metric = BINARY_F1(ignore_index=-1,
                                        constraint_type='BX',
                                        dist_sync_on_step=dist_sync_on_step)
        elif params['metric_type'] == 'word-bin-th':
            self.metric = TWS_WORD_F1(ignore_index=-1,
                                      constraint_type='BMES',
                                      dist_sync_on_step=dist_sync_on_step)
            self.bin_metric = BINARY_F1(ignore_index=-1,
                                        constraint_type='BX',
                                        dist_sync_on_step=dist_sync_on_step)
        else:
            raise ValueError('invalid metric type: {}'.format(
                params['metric_type']))
        self.oov_metric = CWS_OOV_RECALL(vocab=self.data_module._train_vocab,
                                         dist_sync_on_step=dist_sync_on_step)

    def _get_data_module_params(self):
        return dict({
            'train_file':
            self.hparams.train_file,
            'valid_file':
            self.hparams.valid_file,
            'test_file':
            self.hparams.test_file if self.hparams.test_file else None,
            'data_root_dir':
            self.hparams.data_root_dir,
            'ext_dic_file':
            self.hparams.ext_dic_file if self.hparams.ext_dic_file else None,
            'batch_size':
            self.hparams.batch_size,
            'pretrained_model':
            self.hparams.pretrained_model,
            'model_max_seq_length':
            self.hparams.model_max_seq_length,
            'wv_model_path':
            self.hparams.wv_model_path,
            'lang':
            self.hparams.lang,
            'normalize_unicode':
            self.hparams.normalize_unicode,
            'max_token_length':
            self.hparams.max_token_length,
            'min_token_freq_for_trie':
            self.hparams.min_token_freq_for_trie,
            'node_comp_type':
            self.hparams.node_comp_type,
            'graph_dropout':
            self.hparams.graph_dropout,
            'train_split_ratio':
            self.hparams.train_split_ratio,
            'generate_unigram_node':
            self.hparams.generate_unigram_node,
            'include_dataset_token':
            self.hparams.include_dataset_token,
            'unc_token_ratio':
            self.hparams.unc_token_ratio,
            'include_unc_token':
            self.hparams.include_unc_token,
            'include_lattice': (not self.hparams.unuse_gnn
                                or not self.hparams.unuse_attn),
            'build_dynamic_graph':
            not self.hparams.unuse_dynamic_graph,
            'include_valid_vocab':
            self.hparams.include_valid_vocab,
            'exclude_special_token_map':
            self.hparams.exclude_special_token_map,
            'seed':
            self.hparams.seed,
            'use_binary':
            self.hparams.use_binary
        })

    def _get_criterion_params(self):
        return dict({'criterion_type': self.hparams.criterion_type})

    def _get_metric_params(self):
        return dict({'metric_type': self.hparams.metric_type})

    def _get_lattice_inputs(self, inputs):
        return inputs[LATTICE_INPUT_KEY]

    def _get_mapping_inputs(self, inputs):
        return inputs[MAPPING_TABLE_KEY]

    def _get_org_ids(self, inputs):
        return inputs[ORG_INPUT_KEY]

    def _get_train_and_valid_data(self):
        return self.data_module.train_set, self.data_module.valid_set

    def _get_test_data(self):
        return self.data_module.test_set

    def _get_vocab(self):
        return self.data_module.vocab

    def _get_train_vocab(self):
        return self.data_module._train_vocab

    def _get_char_vocab(self):
        return self.data_module.char_vocab

    def _get_node_vocab(self):
        return self.data_module.node_vocab

    def _get_wv_model(self):
        return self.data_module.wv_model

    def _get_dataset_token(self):
        return self.data_module.dataset_token

    def _get_node_attrs_from_lattice(self, batch) -> torch.Tensor:
        return pad_sequence(
            [data.x.detach().clone() for data in batch.to_data_list()])

    def _get_char_node_attrs_from_lattice(self, batch_lattice) -> torch.Tensor:
        char_node_ids = self._get_char_node_ids_from_lattice(batch_lattice)
        char_node_attrs = []
        for node_ids, data in zip(char_node_ids, batch_lattice.to_data_list()):
            node_attrs = data.x.detach().clone()
            char_node_attrs.append(node_attrs[node_ids])
        return pad_sequence([attrs for attrs in char_node_attrs])

    def _get_tokens_from_lattice(self, batch):
        return [data.token for data in batch.to_data_list()]

    def _embed_lattice_tokens(self, inputs):
        lattice = inputs.detach().clone()
        token_ids = lattice.token_id
        lattice.x = self._get_token_embeddings(token_ids)
        return lattice

    def _get_mapping_table_and_mask(self, node_attrs, mapping, padding_idx=0):
        '''
        node_attrs: node embeddings from graph
        mapping: indexes for character-to-candidate tokens
        return:
            mapping_table: table to keep for each batch which has n characters,
                and for each character has candidate tokens with t
                he embed_size dimension.
                e.g. mapping_table.shape = (2, 5, 10, 300) means
                    there are 2 batches, each batch has at most 5 characters,
                    each character has at most 10 candidate tokens with 300
                    feature dimension
            mask: a tensor to ignore padding token in mapping
        '''
        batch_size, n_tokens, embed_size = node_attrs.shape
        batch_size, n_unigrams, n_cand_tokens = mapping.shape
        mask = mapping != padding_idx

        flatten_mapping = mapping.view(batch_size, -1)
        maps = [
            node_attrs[batch_idx][indices]
            for batch_idx, indices in enumerate(flatten_mapping)
        ]

        mapping_table = torch.stack(maps)
        mapping_table = mapping_table.view(batch_size, n_unigrams,
                                           n_cand_tokens, embed_size)
        return mapping_table, mask

    def _get_seq_lengths_from_lattice(self, batch) -> torch.Tensor:
        return batch.len

    def _pad_ignore_labels(self, ys, ts, ignore_label_index=-1):
        pad_labels = ys == ignore_label_index
        ts[pad_labels] = ignore_label_index
        return ts

    def _construct_binary_labels(self, ts, ys) -> torch.Tensor:
        '''
        bmes_dict = {'B': 0, 'M': 1, 'E': 2, 'S': 3}
        '''
        preds = ts.detach().clone()
        golds = ys.detach().clone()

        pred_b_mask = preds == 0
        pred_m_mask = preds == 1
        pred_e_mask = preds == 2
        pred_s_mask = preds == 3
        preds[pred_b_mask] = 1
        preds[pred_m_mask] = 0
        preds[pred_e_mask] = 0
        preds[pred_s_mask] = 1

        gold_b_mask = golds == 0
        gold_m_mask = golds == 1
        gold_e_mask = golds == 2
        gold_s_mask = golds == 3
        golds[gold_b_mask] = 1
        golds[gold_m_mask] = 0
        golds[gold_e_mask] = 0
        golds[gold_s_mask] = 1

        return preds, golds

    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = argparse.ArgumentParser(parents=[parent_parser],
                                         add_help=False)
        parser.add_argument('--train-file', type=Path)
        parser.add_argument('--valid-file', type=Path)
        parser.add_argument('--test-file', type=Path)
        parser.add_argument('--data-root-dir', type=Path, default='.')
        parser.add_argument('--ext-dic-file', type=Path)
        parser.add_argument('--save-dir', type=Path, default='.')
        parser.add_argument('--decode-save-dir', type=Path, default='.')
        parser.add_argument('--model-name')
        parser.add_argument('--model-version', type=int, default=0)
        parser.add_argument('--batch-size', type=int, default=32)
        parser.add_argument('--max-epochs', type=int, default=20)
        parser.add_argument('--accumulate-grad-batches', type=int, default=4)
        parser.add_argument('--gradient-clip-val', type=float, default=5.0)
        parser.add_argument('--embed-size', type=int, default=256)
        parser.add_argument('--lr', type=float, default=1e-3)
        parser.add_argument('--gnn-lr', type=float, default=1e-3)
        parser.add_argument('--optimized-decay',
                            action='store_true',
                            help='weight decay on subset of model params')
        parser.add_argument('--scheduler',
                            action='store_true',
                            help='use scheduler to control lr value')
        parser.add_argument('--lr-decay-rate', type=float, default=0.9)
        parser.add_argument('--num-layers', type=int, default=2)
        parser.add_argument('--gnn-type',
                            choices=['gcn', 'gat'],
                            default='gat')
        parser.add_argument('--gnn-num-heads', type=int, default=2)
        parser.add_argument('--unidirectional-gnn', action='store_true')
        parser.add_argument('--shallow-gnn', action='store_true')
        parser.add_argument('--dropout', type=float, default=0.2)
        parser.add_argument('--graph-dropout', type=float, default=0.0)
        parser.add_argument('--attn-dropout', type=float, default=0.1)
        parser.add_argument('--max-token-length', type=int, default=4)
        parser.add_argument('--min-token-freq-for-trie', type=int, default=1)
        parser.add_argument('--attn-comp-type',
                            choices=['wavg', 'wcon'],
                            default='wavg')
        parser.add_argument('--unuse-dynamic-graph', action='store_true')
        parser.add_argument('--unuse-gnn', action='store_true')
        parser.add_argument('--unuse-attn', action='store_true')
        parser.add_argument('--include-dataset-token', action='store_true')
        parser.add_argument('--generate-unigram-node', action='store_true')
        parser.add_argument('--unc-token-ratio', type=float, default=0.0)
        parser.add_argument('--include-unc-token', action='store_true')
        parser.add_argument('--include-valid-vocab',
                            action='store_true',
                            help='Specify to include validation vocab')
        parser.add_argument('--use-extra-lstm-layer', action='store_true')
        parser.add_argument('--train-split-ratio', type=float, default=0.0)
        parser.add_argument('--decode', action='store_true')
        return parser
