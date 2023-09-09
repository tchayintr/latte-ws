import argparse
from functools import partial
import os
from pathlib import Path
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from typing import List, Union

from core.modules.attention import Attention
from core.modules.gnn import LatticeEncoder
from taggers.tagger import Tagger
from utils.data.dataset import (LSTMDataset, pad_sequence, ids2tokens,
                                ids2chunks, construct_embed_from_pretrained)

CHAR_INPUT_KEY = 'input_ids'


class LSTMTagger(Tagger):

    def __init__(self, hparams, *args, **kwargs):
        super(LSTMTagger, self).__init__(hparams)
        '''get all vocab for reference'''
        char_vocab = self._get_char_vocab()
        node_vocab = self._get_node_vocab()

        if (not self.hparams.unuse_gnn or not self.hparams.unuse_attn):
            '''force to use extra lstm layer'''
            self.hparams.use_extra_lstm_layer = True
        elif (self.hparams.unuse_gnn or self.hparams.unuse_attn):
            self.hparams.use_extra_lstm_layer = False

        # char/node embeddings
        char_embed_size, node_embed_size = 0, 0
        if self.hparams.wv_model_path:
            wv_model = self._get_wv_model()
            char_embed_size = 300
            self.hparams.char_embed_size = char_embed_size
            self.char_embed = construct_embed_from_pretrained(
                char_vocab,
                embed_size=char_embed_size,
                wv_model=wv_model,
                padding_idx=0,
                freeze=self.hparams.unfreeze_wv)
            if (not self.hparams.unuse_gnn or not self.hparams.unuse_attn):
                node_embed_size = 300
                self.hparams.node_embed_size = node_embed_size
                self.node_embed = construct_embed_from_pretrained(
                    node_vocab,
                    embed_size=node_embed_size,
                    wv_model=wv_model,
                    padding_idx=0,
                    freeze=not self.hparams.unfreeze_wv)
        else:
            char_embed_size = self.hparams.char_embed_size
            self.char_embed = nn.Embedding(num_embeddings=len(char_vocab),
                                           embedding_dim=char_embed_size,
                                           padding_idx=0)
            if (not self.hparams.unuse_gnn or not self.hparams.unuse_attn):
                node_embed_size = self.hparams.node_embed_size
                self.node_embed = nn.Embedding(num_embeddings=len(node_vocab),
                                               embedding_dim=node_embed_size,
                                               padding_idx=0)

        self.lstm = nn.LSTM(input_size=char_embed_size,
                            hidden_size=self.hparams.hidden_size,
                            num_layers=self.hparams.num_layers,
                            batch_first=True,
                            bidirectional=True)
        self.hparams.hidden_size *= 2

        # gnn
        if not self.hparams.unuse_gnn:
            self.gnn = LatticeEncoder(
                embed_size=node_embed_size,
                hidden_size=self.hparams.hidden_size,
                num_layers=self.hparams.num_layers,
                num_heads=self.hparams.gnn_num_heads,
                dropout=self.hparams.dropout,
                gnn_type=self.hparams.gnn_type,
                add_self_loops=True,
                bidirectional=not self.hparams.unidirectional_gnn,
                shallow=self.hparams.shallow_gnn)

        # attn
        attn_feat_size = 1
        if not self.hparams.unuse_attn:
            self.attention = Attention(
                node_embed_size,
                self.hparams.hidden_size,
                attn_comp_type=self.hparams.attn_comp_type,
                inner_dropout=self.hparams.attn_dropout)
            if self.hparams.attn_comp_type == 'wavg':
                attn_feat_size = 2
            elif self.hparams.attn_comp_type == 'wcon':
                attn_feat_size = sum(
                    [i
                     for i in range(1, self.hparams.max_token_length + 1)]) + 1

        if self.hparams.use_extra_lstm_layer:
            if (not self.hparams.unuse_gnn and self.hparams.unuse_attn):
                extra_lstm_input_size = (self.hparams.hidden_size +
                                         node_embed_size)
            elif ((not self.hparams.unuse_gnn and not self.hparams.unuse_attn)
                  or (self.hparams.unuse_gnn and not self.hparams.unuse_attn)):
                extra_lstm_input_size = (self.hparams.hidden_size *
                                         attn_feat_size)
            else:
                raise AssertionError
            self.lstm2 = nn.LSTM(input_size=extra_lstm_input_size,
                                 hidden_size=self.hparams.hidden_size,
                                 num_layers=self.hparams.num_layers,
                                 batch_first=True,
                                 bidirectional=True)

        # classifier
        classifier_in = (
            self.hparams.hidden_size *
            (attn_feat_size if not self.hparams.use_extra_lstm_layer else 2))
        self.classifier = nn.Linear(classifier_in, 4)

        # dropout
        self.dropout = nn.Dropout(self.hparams.dropout)

        # datasets
        self.train_set, self.valid_set = self._get_train_and_valid_data()
        self.test_set = self._get_test_data()

        # evaluation
        self.word_str_eval = (self.hparams.metric_type == 'word'
                              or self.hparams.metric_type == 'word-bin-th')
        self.use_bin_eval = (self.hparams.metric_type == 'word-bin'
                             or self.hparams.metric_type == 'word-bin-th')

        # ddp
        self.use_ddp = True if self.hparams.num_gpus > 1 else False

    def forward(self, inputs, *args, **kwargs):
        # unigrams
        char_inputs = self._get_char_inputs(inputs)
        es = self.char_embed(char_inputs)
        outputs, (_, __) = self.lstm(es)
        if self.training:
            outputs = self.dropout(outputs)

        if not self.hparams.unuse_gnn or not self.hparams.unuse_attn:
            # tokens (gnn)
            lattice_inputs = self._get_lattice_inputs(inputs)
            lattice_outputs = self._embed_lattice_tokens(lattice_inputs)

            if not self.hparams.unuse_gnn:
                lattice_outputs = self.gnn(lattice_outputs)
                char_node_attrs = self._get_char_node_attrs_from_lattice(
                    lattice_outputs)

                if self.hparams.unuse_attn:
                    outputs = torch.cat([outputs, char_node_attrs], dim=-1)
            node_attrs = self._get_node_attrs_from_lattice(lattice_outputs)

            if not self.hparams.unuse_attn:
                # unigram-tokens mapping
                mapping_inputs = self._get_mapping_inputs(inputs)
                mapping_table, mapping_mask = self._get_mapping_table_and_mask(
                    node_attrs, mapping_inputs, padding_idx=0)

                if not self.hparams.exclude_special_token_map:
                    input_lengths = self._get_input_lengths(char_inputs)
                    mapping_inputs, mapping_table, mapping_mask = (
                        self._trim_special_tokens(mapping_inputs,
                                                  mapping_table, mapping_mask,
                                                  input_lengths))

                # attention
                outputs, _ = self.attention(outputs, mapping_table,
                                            mapping_mask)

        if self.hparams.use_extra_lstm_layer:
            outputs, (_, __) = self.lstm2(outputs)
            if self.training:
                outputs = self.dropout(outputs)

        outputs = self.classifier(outputs)
        return outputs

    def training_step(self, batch, batch_idx=None):
        xs, ys = batch
        ps = self.forward(xs)
        loss = self._compute_active_loss(xs, ys, ps)
        self.log('train/loss',
                 loss,
                 on_epoch=True,
                 prog_bar=False,
                 logger=True,
                 sync_dist=True,
                 rank_zero_only=True)
        return {'loss': loss}

    def validation_step(self, batch, batch_idx=None):
        xs, ys = batch
        ps = self.forward(xs)
        loss = self._compute_active_loss(xs, ys, ps)
        ts = self._tagging(xs, ys, ps)

        org_ids = self._get_org_ids(xs)
        ts_str, ys_str = self._reconstruct_org_seqs(ts, ys, org_ids)

        if self.use_bin_eval:
            ts_bin, ys_bin = self._construct_binary_labels(ts, ys)
            self.bin_metric(ts_bin, ys_bin)

        if self.word_str_eval:
            ts, ys = ts_str, ys_str
        self.metric(ts, ys)
        self.oov_metric(ts_str, ys_str)

        self.log('valid/loss',
                 loss,
                 on_epoch=True,
                 prog_bar=True,
                 logger=True,
                 sync_dist=True,
                 rank_zero_only=True)
        return {'valid/loss': loss}

    def validation_end(self, outputs):
        avg_loss = torch.stack([output['valid/loss']
                                for output in outputs]).mean()
        return {'valid/loss': avg_loss}

    def test_step(self, batch, batch_idx=None):
        xs, ys = batch
        ps = self.forward(xs)
        ts = self._tagging(xs, ys, ps)
        org_ids = self._get_org_ids(xs)
        ts_str, ys_str = self._reconstruct_org_seqs(ts, ys, org_ids)
        if self.use_bin_eval:
            ts_bin, ys_bin = self._construct_binary_labels(ts, ys)
            self.bin_metric(ts_bin, ys_bin)

        if self.word_str_eval:
            ts, ys = ts_str, ys_str
        self.metric(ts, ys)
        self.oov_metric(ts_str, ys_str)

        return

    def predict_step(self, batch, batch_idx=None):
        raise NotImplementedError

    def validation_epoch_end(self, outputs):
        avg_loss = torch.stack([output['valid/loss']
                                for output in outputs]).mean()
        self.log('valid/loss',
                 avg_loss,
                 on_epoch=True,
                 prog_bar=True,
                 logger=True,
                 sync_dist=True,
                 rank_zero_only=True)
        self.log('valid/micro-F1',
                 self.metric.compute(),
                 on_epoch=True,
                 prog_bar=True,
                 logger=True,
                 sync_dist=True,
                 rank_zero_only=True)
        self.log('valid/oov-recall',
                 self.oov_metric.compute(),
                 on_epoch=True,
                 prog_bar=True,
                 logger=True,
                 sync_dist=True,
                 rank_zero_only=True)
        self.metric.reset()
        self.oov_metric.reset()
        if self.use_bin_eval:
            self.log('valid/bin-micro-F1',
                     self.bin_metric.compute(),
                     on_epoch=True,
                     prog_bar=True,
                     logger=True,
                     sync_dist=True,
                     rank_zero_only=True)
            self.bin_metric.reset()
        return

    def test_epoch_end(self, outputs):
        self.log('test/micro-F1',
                 self.metric.compute(),
                 on_epoch=True,
                 prog_bar=True,
                 logger=True,
                 sync_dist=True,
                 rank_zero_only=True)
        self.log('test/oov-recall',
                 self.oov_metric.compute(),
                 on_epoch=True,
                 prog_bar=True,
                 logger=True,
                 sync_dist=True,
                 rank_zero_only=True)
        self.metric.reset()
        self.oov_metric.reset()
        if self.use_bin_eval:
            self.log('test/bin-micro-F1',
                     self.bin_metric.compute(),
                     on_epoch=True,
                     prog_bar=True,
                     logger=True,
                     sync_dist=True,
                     rank_zero_only=True)
            self.bin_metric.reset()
        return

    def predict_epoch_end(self, outputs):
        raise NotImplementedError

    def train_dataloader(self):
        params = self._get_dataloader_params(train=True)
        return DataLoader(self.train_set, **params)

    def val_dataloader(self):
        params = self._get_dataloader_params(train=False)
        return DataLoader(self.valid_set, **params)

    def test_dataloader(self):
        params = self._get_dataloader_params(train=False)
        return DataLoader(self.test_set, **params)

    def configure_optimizers(self):
        parameters = self.parameters()
        optimizer = torch.optim.AdamW(parameters, self.hparams.lr)
        return_values = [optimizer]
        if self.hparams.scheduler:
            scheduler = torch.optim.lr_scheduler.ExponentialLR(
                optimizer, gamma=self.hparams.lr_decay_rate)
            return_values = ([optimizer], [scheduler])
        return return_values

    def _get_dataloader_params(self, train=True):
        include_lattice = (not self.hparams.unuse_attn
                           or not self.hparams.unuse_gnn)
        build_dynamic_graph = False
        dynamic_graph_dropout = 0.0
        if train:
            build_dynamic_graph = (not self.hparams.unuse_dynamic_graph
                                   and self.hparams.graph_dropout > 0.0)
            dynamic_graph_dropout = self.hparams.graph_dropout
        return dict({
            'batch_size':
            self.hparams.batch_size,
            'shuffle':
            False if self.use_ddp else train,
            'num_workers':
            int(os.cpu_count() / (4 if not self.use_ddp else 2)),
            # 1,
            'pin_memory':
            True,
            'collate_fn':
            partial(LSTMDataset.generate_batch,
                    char_vocab=self._get_char_vocab(),
                    include_lattice=include_lattice,
                    build_dynamic_graph=build_dynamic_graph,
                    dynamic_graph_dropout=dynamic_graph_dropout)
        })

    def _get_char_inputs(self, inputs):
        return inputs[CHAR_INPUT_KEY]

    def _get_input_lengths(self, inputs):
        '''get length for each input (padding_idx=0)'''
        return torch.sum(inputs != 0, dim=-1)

    # TODO: node_comp_type support
    def _get_token_embeddings(self, token_ids: torch.tensor) -> torch.Tensor:
        if (self.hparams.node_comp_type == 'init'
                or self.hparams.node_comp_type == 'wv'):
            raise NotImplementedError
        elif (self.hparams.node_comp_type == 'avg'
              or self.hparams.node_comp_type == 'concat'):
            raise NotImplementedError
        elif (self.hparams.node_comp_type == 'none'):
            outputs = self.node_embed(token_ids)
        else:
            raise AssertionError
        return torch.squeeze(outputs)

    def _get_char_node_ids_from_lattice(self, batch_lattice) -> List:
        '''unigrams need to be placed sequencially'''
        char_node_ids = []
        for i, data in enumerate(batch_lattice.to_data_list()):
            '''disregard <bos>, <eos>, and dataset_token'''
            '''unigrams'''
            _char_node_ids = [
                node_idx
                for node_idx, (start_idx, end_idx) in enumerate(data.span)
                if end_idx - start_idx == 1
            ]
            char_node_ids.append(_char_node_ids)
        return char_node_ids

    def _compute_active_loss(self, xs, ys, ps):
        if self._get_criterion_params()['criterion_type'] == 'crf':
            active_loss = ys != -1
            active_logits = ps
            ignore_label_index = torch.tensor(
                self.criterion.ignore_index_for_mask).type_as(ys)
            active_labels = torch.where(active_loss, ys, ignore_label_index)
            log_likelihood = self.criterion(active_logits, active_labels,
                                            active_loss)
            loss = -log_likelihood / ys.size(0)
        else:
            active_loss = ys != -1
            active_logits = ps.view(-1, 4)
            ignore_label_index = torch.tensor(
                self.criterion.ignore_index).type_as(ys)
            active_labels = torch.where(active_loss, ys.view(-1),
                                        ignore_label_index)
            loss = self.criterion(active_logits, active_labels)
        return loss

    def _tagging(self, xs, ys, ps):
        if self._get_criterion_params()['criterion_type'] == 'crf':
            mask = ys != -1
            outputs = self.criterion.viterbi_tags(ps, mask)
            return pad_sequence([
                torch.tensor(tag, device=ps.device) for tag, score in outputs
            ])
        else:
            return torch.argmax(ps, dim=2)

    def _trim_special_tokens(self, mapping_inputs, mapping_table, mapping_mask,
                             lengths):

        front_offset = 1
        '''disregard bos and dataset_token'''
        if self.hparams.include_dataset_token:
            front_offset += 1
        '''disregard eos'''

        trimmed_mapping_inputs = []
        trimmed_mapping_table = []
        trimmed_mapping_mask = []
        for batch_idx, input_len in enumerate(lengths):
            active_input_offset = front_offset + input_len
            back_offset = active_input_offset + 1
            '''mapping_inputs'''
            trimmed_mapping_inputs.append(
                torch.cat([
                    mapping_inputs[batch_idx]
                    [front_offset:active_input_offset],
                    mapping_inputs[batch_idx][back_offset:]
                ]))
            '''mapping_table'''
            trimmed_mapping_table.append(
                torch.cat([
                    mapping_table[batch_idx][front_offset:active_input_offset],
                    mapping_table[batch_idx][back_offset:]
                ]))
            '''mapping_mask'''
            trimmed_mapping_mask.append(
                torch.cat([
                    mapping_mask[batch_idx][front_offset:active_input_offset],
                    mapping_mask[batch_idx][back_offset:]
                ]))
        trimmed_mapping_inputs = pad_sequence(trimmed_mapping_inputs)
        trimmed_mapping_table = pad_sequence(trimmed_mapping_table)
        trimmed_mapping_mask = pad_sequence(trimmed_mapping_mask)

        return (trimmed_mapping_inputs, trimmed_mapping_table,
                trimmed_mapping_mask)

    def _reconstruct_org_seqs(self, ts, ys,
                              org_ids) -> Union[List[str], List[str]]:
        vocab = self._get_char_vocab()
        seqs = [ids2tokens(ids.tolist(), vocab) for ids in org_ids]
        preds, golds = [], []
        for seq, pred_ids, gold_ids in zip(seqs, ts.tolist(), ys.tolist()):
            pred_ids = pred_ids[:len(seq)]
            gold_ids = gold_ids[:len(seq)]
            str_pred = ' '.join(ids2chunks(pred_ids, seq))
            str_gold = ' '.join(ids2chunks(gold_ids, seq))
            preds.append(str_pred)
            golds.append(str_gold)
        return preds, golds

    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = argparse.ArgumentParser(parents=[parent_parser],
                                         add_help=False)
        parser.add_argument('--char-embed-size', type=int, default=128)
        parser.add_argument('--node-embed-size', type=int, default=300)
        parser.add_argument('--hidden-size', type=int, default=600)
        parser.add_argument('--wv-model-path', type=Path)
        parser.add_argument('--unfreeze-wv', action='store_true')
        parser.add_argument(
            '--exclude-special-token-map',
            action='store_true',
            help='Specify to exclude special tokens from mapping table')
        return parser
