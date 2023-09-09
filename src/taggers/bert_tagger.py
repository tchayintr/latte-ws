import argparse
from functools import partial
import os
import torch
import torch.nn as nn
import transformers
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from typing import List, Union

from core.modules.attention import Attention
from core.modules.gnn import LatticeEncoder
from taggers.tagger import Tagger
from utils.data.dataset import (BertDataset, pad_sequence, ids2tokens,
                                ids2chunks, lattice2cspans, construct_embed,
                                construct_embed_from_pretrained)

BERT_CHAR_INPUT_KEYS = ['input_ids', 'attention_mask', 'token_type_ids']
CHAR_ORG_IDS = 'org_ids'


class BertTagger(Tagger):

    def __init__(self, hparams, *args, **kwargs):
        super(BertTagger, self).__init__(hparams)
        if self.hparams.exclude_special_token_map:
            raise AssertionError(
                'bert tagger does not support `--exclude-special-token-map`')

        # bert: config and model
        bert_config = transformers.AutoConfig.from_pretrained(
            self.hparams.pretrained_model, output_hidden_states=True)
        self.bert = transformers.AutoModel.from_pretrained(
            self.hparams.pretrained_model, config=bert_config)

        # bert: resize if any token is added
        self.resize_bert_embeddings(
            self.bert,
            vocab_size=self._get_tokenizer_vocab_size(),
            org_vocab_size=bert_config.vocab_size)

        # bert: freezing
        if self.hparams.freeze_bert:
            print('[INFO]: freezing BERT weight')
            self.freeze_bert(self.bert)

        self.hparams.hidden_size = self.bert.config.hidden_size
        if self.hparams.bert_mode == 'concat':
            self.hparams.hidden_size *= 4

        # node embeddings
        if self.hparams.node_comp_type == 'init':
            params = self._get_embed_params()
            self.node_embed = construct_embed(**params)
            node_embed_size = self.hparams.embed_size
        elif (self.hparams.node_comp_type == 'wv'
              and self.hparams.wv_model_path):
            self.hparams.embed_size = 300
            params = self._get_pretrained_embed_params()
            self.node_embed = construct_embed_from_pretrained(**params)
            node_embed_size = self.hparams.embed_size
        else:
            node_embed_size = self.hparams.hidden_size
            if (self.hparams.node_comp_type == 'wvc'
                    and self.hparams.wv_model_path):
                self.hparams.hidden_size += 300
                node_embed_size = self.hparams.hidden_size
            elif self.hparams.node_comp_type == 'concat':
                self.node_embed_proj_size = (node_embed_size *
                                             self.hparams.max_token_length)
                self.node_embed_proj = nn.Linear(self.node_embed_proj_size,
                                                 node_embed_size)
            self.node_embed = self.bert

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
            if self.hparams.unuse_attn:
                self.hparams.hidden_size *= 2

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
            self.lstm = nn.LSTM(input_size=extra_lstm_input_size,
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
        bert_char_inputs = self.get_bert_char_inputs(inputs)
        char_outputs = self.bert(**bert_char_inputs)
        outputs = self._get_feats_from_bert_outputs(char_outputs)
        if self.training:
            outputs = self.dropout(outputs)

        if not self.hparams.unuse_gnn or not self.hparams.unuse_attn:
            # tokens (gnn)
            lattice_inputs = self._get_lattice_inputs(inputs)
            lattice_outputs = self._embed_lattice_tokens(lattice_inputs)

            # contextualised char nodes/context-free char nodes
            '''whether to use contextualised char from char_outputs'''
            if not self.hparams.unuse_context_char_node:
                '''
                replace context-free char in graph with
                contextualised char from char_outputs
                '''
                char_input_lengths = self._get_char_input_lengths(inputs)
                lattice_outputs = self._replace_char_node_attrs(
                    lattice_outputs, outputs, char_input_lengths)
            '''whether to concat nodes with wv'''
            if self.hparams.node_comp_type == 'wvc':
                outputs = self._concat_outputs_with_wv(outputs,
                                                       lattice_outputs)
                lattice_outputs = self._concat_node_attrs_with_wv(
                    lattice_outputs)

            if not self.hparams.unuse_gnn:
                # unigram-tokens mapping
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

                # attention
                outputs, _ = self.attention(outputs, mapping_table,
                                            mapping_mask)

        if self.hparams.use_extra_lstm_layer:
            outputs, (_, __) = self.lstm(outputs)
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
        ts = self._pad_ignore_labels(ys, ts)

        org_ids = self._get_org_ids(xs)
        ts, ys = self._pad_special_label_tokens(ts, ys, org_ids)
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

    def test_step(self, batch, batch_idx=None):
        xs, ys = batch
        ps = self.forward(xs)
        ts = self._tagging(xs, ys, ps)
        ts = self._pad_ignore_labels(ys, ts)

        org_ids = self._get_org_ids(xs)
        ts, ys = self._pad_special_label_tokens(ts, ys, org_ids)
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
        xs, ys = batch
        ps = self.forward(xs)
        ts = self._tagging(xs, ys, ps)
        ts = self._pad_ignore_labels(ys, ts)

        org_ids = self._get_org_ids(xs)
        ts, ys = self._pad_special_label_tokens(ts, ys, org_ids)
        ts_str, ys_str = self._reconstruct_org_seqs(ts, ys, org_ids)

        return ts_str

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
        # pl does not support yet
        raise NotImplementedError

    def train_dataloader(self):
        params = self._get_dataloader_params(train=True)
        sampler = DistributedSampler(self.train_set) if self.use_ddp else None
        return DataLoader(self.train_set, sampler=sampler, **params)

    def val_dataloader(self):
        params = self._get_dataloader_params(train=False)
        sampler = DistributedSampler(self.valid_set) if self.use_ddp else None
        return DataLoader(self.valid_set, sampler=sampler, **params)

    def test_dataloader(self):
        params = self._get_dataloader_params(train=False)
        sampler = DistributedSampler(self.test_set) if self.use_ddp else None
        return DataLoader(self.test_set, sampler=sampler, **params)

    def predict_dataloader(self):
        params = self._get_dataloader_params(train=False)
        return DataLoader(self.test_set, **params)

    def configure_optimizers(self):
        parameters = self._optimizer_grouped_parameters()
        optimizer = torch.optim.AdamW(parameters, self.hparams.lr)
        return_values = [optimizer]
        if self.hparams.scheduler:
            scheduler = torch.optim.lr_scheduler.ExponentialLR(
                optimizer, gamma=self.hparams.lr_decay_rate)
            return_values = ([optimizer], [scheduler])
        return return_values

    def _optimizer_grouped_parameters(self):
        optimizer_grouped_parameters = []

        # common
        uncommon = ['bert', 'gnn', 'classifier']
        common_params = [
            p for n, p in self.named_parameters()
            if not any(uc in n for uc in uncommon)
        ]
        optimizer_grouped_parameters.append({
            'params': common_params,
            'lr': self.hparams.lr
        })

        # gnn
        if not self.hparams.unuse_gnn:
            optimizer_grouped_parameters.append({
                'params': self.gnn.parameters(),
                'lr': self.hparams.gnn_lr
            })

        # bert
        optimizer_grouped_parameters.append({
            'params':
            self.classifier.parameters(),
            'lr':
            self.hparams.bert_lr
        })
        if self.hparams.optimized_decay:
            no_decay = ['bias', 'LayerNorm.weight']
            bert_decay_params = [
                p for n, p in self.bert.named_parameters()
                if not any(nd in n for nd in no_decay)
            ]
            bert_no_decay_params = [
                p for n, p in self.bert.named_parameters()
                if any(nd in n for nd in no_decay)
            ]
            optimizer_grouped_parameters.append({
                'params': bert_decay_params,
                'weight_decay': 0.01,
                'lr': self.hparams.bert_lr
            })
            optimizer_grouped_parameters.append({
                'params': bert_no_decay_params,
                'weight_decay': 0.0,
                'lr': self.hparams.bert_lr
            })
        else:
            optimizer_grouped_parameters.append({
                'params':
                self.bert.parameters(),
                'lr':
                self.hparams.bert_lr
            })

        return optimizer_grouped_parameters

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
            partial(BertDataset.generate_batch,
                    char_vocab=self._get_char_vocab(),
                    include_lattice=include_lattice,
                    build_dynamic_graph=build_dynamic_graph,
                    dynamic_graph_dropout=dynamic_graph_dropout)
        })

    def _get_embed_params(self):
        return dict({
            'vocab': self._get_node_vocab(),
            'embed_size': self.hparams.embed_size,
            'padding_idx': 0
        })

    def _get_pretrained_embed_params(self):
        return dict({
            'vocab': self._get_node_vocab(),
            'embed_size': self.hparams.embed_size,
            'wv_model': self._get_wv_model(),
            'pad_token': '[PAD]',
            'unk_token': '[UNK]',
            'bos_token': '[CLS]',
            'eos_token': '[SEP]',
            'padding_idx': 0,
            'freeze': True
        })

    def _get_tokenizer(self):
        return self.data_module.tokenizer

    def _get_tokenizer_vocab_size(self):
        return len(self.data_module.tokenizer)

    def _get_token_embeddings(self, token_ids: torch.tensor) -> torch.Tensor:
        if (self.hparams.node_comp_type == 'init'
                or self.hparams.node_comp_type == 'wv'):
            outputs = self.node_embed(token_ids)

        elif (self.hparams.node_comp_type == 'avg'
              or self.hparams.node_comp_type == 'concat'):
            padding_idx = (self.node_embed.get_input_embeddings().padding_idx
                           if
                           self.node_embed.get_input_embeddings().padding_idx
                           is not None else 0)
            mask = token_ids == padding_idx

            node_outputs = self._get_feats_from_bert_outputs(
                self.node_embed(token_ids))
            padding_value = torch.tensor(padding_idx).type_as(node_outputs)
            node_outputs[mask] = padding_value

            if self.hparams.node_comp_type == 'avg':
                outputs = node_outputs.mean(dim=1, keepdim=True)
            elif self.hparams.node_comp_type == 'concat':
                b, l, h = node_outputs.shape
                outputs = node_outputs.view(b, 1, l * h)
                outputs = self.node_embed_proj(outputs)

        elif (self.hparams.node_comp_type == 'none'
              or self.hparams.node_comp_type == 'extend'
              or self.hparams.node_comp_type == 'wvc'):
            outputs = self._get_feats_from_bert_outputs(
                self.node_embed(token_ids))

        else:
            raise AssertionError
        return torch.squeeze(outputs)

    def _get_feats_from_bert_outputs(self, outputs):
        if self.hparams.bert_mode == 'none':
            feats = outputs[0]
        elif self.hparams.bert_mode == 'concat':
            feats = outputs[2][-4:]
            feats = torch.cat(feats, dim=-1)
        elif self.hparams.bert_mode == 'sum':
            feats = outputs[2][-4:]
            feats = torch.stack(feats, dim=0).sum(dim=0)
        elif self.hparams.bert_mode == 'sum-all':
            feats = outputs[2][:]
            feats = torch.stack(feats, dim=0).sum(dim=0)
        else:
            raise ValueError
        return feats

    def _get_char_input_lengths(self, inputs):
        '''get org_ids and convert it into tensor of input length'''
        org_ids = self.get_char_input_org_ids(inputs)
        char_input_lengths = torch.sum(org_ids != 0, dim=-1)
        return char_input_lengths

    def _get_char_wv_mats(self, batch_lattice):
        vector_lookup_fn = self._get_wv_model().get_word_vector
        char_node_ids = self._get_char_node_ids_from_lattice(batch_lattice)
        wv_mats = []
        for node_ids, tokens in zip(char_node_ids, batch_lattice.token):
            char_tokens = [tokens[id] for id in node_ids]
            wv_mat = torch.stack([
                torch.tensor(vector_lookup_fn(char_token), device=self.device)
                for char_token in char_tokens
            ])
            wv_mats.append(wv_mat)
        return pad_sequence(wv_mats)

    def _get_node_attrs_wv(self, batch_lattice):
        '''flatten tokens'''
        tokens = [token for tokens in batch_lattice.token for token in tokens]
        '''get wv_model and get vector lookup function'''
        vector_lookup_fn = self._get_wv_model().get_word_vector
        '''gather vectors from tokens'''
        wv_mat = torch.stack([
            torch.tensor(vector_lookup_fn(token), device=self.device)
            for token_idx, token in enumerate(tokens)
        ])
        return wv_mat

    def _get_char_node_ids_from_lattice(self, batch_lattice) -> List:
        '''unigrams need to be placed sequencially'''
        char_node_ids = []
        for i, data in enumerate(batch_lattice.to_data_list()):
            '''[BOS], [EOS], and dataset_token'''
            special_node_ids = [
                node_idx
                for node_idx, (start_idx, end_idx) in enumerate(data.span)
                if (end_idx -
                    start_idx == 0 and (start_idx >= 0 and end_idx >= 0))
            ]
            '''unigrams'''
            _char_node_ids = [
                node_idx
                for node_idx, (start_idx, end_idx) in enumerate(data.span)
                if end_idx - start_idx == 1
            ]
            if self.hparams.include_dataset_token:
                _char_node_ids = [special_node_ids[2]] + _char_node_ids
            _char_node_ids = [special_node_ids[0]
                              ] + _char_node_ids + [special_node_ids[1]]
            char_node_ids.append(_char_node_ids)
        return char_node_ids

    def _get_char_node_indices(self, batch_lattice, batch, input_lengths):
        '''get all char node indices for referring to base chars'''
        batch_lattice_cspans = [
            lattice2cspans(lattice)
            for lattice in batch_lattice.to_data_list()
        ]
        char_node_indices = []
        '''offset to shift spans for special_tokens'''
        offset = (2 if self.hparams.include_dataset_token else 1)
        for cspans, input_len in zip(batch_lattice_cspans,
                                     input_lengths.tolist()):
            indices = []
            for cspan in cspans:
                char_node_idx = cspan[0]
                char_node_idx += offset
                indices.append(char_node_idx)
            '''
            add special_tokens indices
            i.e., [CLS], [dataset_token], and [SEP]
            '''
            indices = [0] + ([1] if self.hparams.include_dataset_token else
                             []) + indices + [input_len + offset]
            char_node_indices.append(indices)
        return char_node_indices

    def _replace_char_node_attrs(self, batch_lattice, batch, input_lengths):
        '''
        replace nodes by their corresponding char node ids
        - char_node_ids specifies the rows to be replaced in graph
            (batch-level)
        - char_node_indices specifies the rows to be accessed in bert-output
            (batch-level)
        '''
        '''get char_node_ids for batch_lattice'''
        char_node_ids = self._get_char_node_ids_from_lattice(batch_lattice)
        '''get char_node_ids for batch'''
        char_node_indices = self._get_char_node_indices(
            batch_lattice, batch, input_lengths)
        _batch_lattice = batch_lattice.detach().clone()
        for i, (node_ids, node_indices, data) in enumerate(
                zip(char_node_ids, char_node_indices,
                    _batch_lattice.to_data_list())):
            data.x[node_ids] = batch[i][node_indices].detach().clone()
        return _batch_lattice

    def _concat_outputs_with_wv(self, outputs: torch.Tensor,
                                batch_lattice: torch.Tensor) -> torch.Tensor:
        char_wv_mats = self._get_char_wv_mats(batch_lattice)
        return torch.cat([outputs, char_wv_mats], dim=2)

    def _concat_node_attrs_with_wv(
            self, batch_lattice: torch.Tensor) -> torch.Tensor:
        wv_mat = self._get_node_attrs_wv(batch_lattice)
        _batch_lattice = batch_lattice.detach().clone()
        node_attrs = _batch_lattice.x.detach().clone()
        _batch_lattice.x = torch.cat([node_attrs, wv_mat], dim=1)
        return _batch_lattice

    def _compute_active_loss(self, xs, ys, ps):
        if self._get_criterion_params()['criterion_type'] == 'crf':
            active_loss = xs['attention_mask'] == 1
            active_logits = ps
            # NOTE: allennlp bug unseen label is needed even use mask
            ## temporary change padding to 0 and using mask
            ignore_label_index = torch.tensor(
                self.criterion.ignore_index_for_mask).type_as(ys)
            active_labels = torch.where(active_loss, ys, ignore_label_index)
            log_likelihood = self.criterion(active_logits, active_labels,
                                            active_loss)
            loss = -log_likelihood / ys.size(0)  # mean
        else:
            active_loss = xs['attention_mask'].view(-1) == 1
            active_logits = ps.view(-1, 4)
            ignore_label_index = torch.tensor(
                self.criterion.ignore_index).type_as(ys)
            active_labels = torch.where(active_loss, ys.view(-1),
                                        ignore_label_index)
            loss = self.criterion(active_logits, active_labels)
        return loss

    def _tagging(self, xs, ys, ps):
        if self._get_criterion_params()['criterion_type'] == 'crf':
            mask = xs['attention_mask'] != 0
            outputs = self.criterion.viterbi_tags(ps, mask)
            return pad_sequence([
                torch.tensor(tag, device=ps.device) for tag, score in outputs
            ])
        else:
            return torch.argmax(ps, dim=2)

    def _pad_special_label_tokens(self, ts, ys, org_ids) -> torch.Tensor:
        offset = 1
        if self.hparams.include_dataset_token:
            offset = 2
        seq_mask = torch.sum(org_ids != 0, dim=-1)
        '''front'''
        ts[:, :offset] = -1
        ys[:, :offset] = -1
        '''back'''
        for batch_idx, seq_len in enumerate(seq_mask):
            ts[batch_idx, seq_len + offset] = -1
            ys[batch_idx, seq_len + offset] = -1
        return ts, ys

    def _reconstruct_org_seqs(self, ts, ys,
                              org_ids) -> Union[List[str], List[str]]:
        len_offset = 1
        if self.hparams.include_dataset_token:
            len_offset = 2
        vocab = self._get_char_vocab()
        seqs = [ids2tokens(ids.tolist(), vocab) for ids in org_ids]
        preds, golds = [], []
        for seq, pred_ids, gold_ids in zip(seqs, ts.tolist(), ys.tolist()):
            '''
            seq: len=L
            seq + special_tokens: len=L+offset
            loc(seq w/o padding): [offset:L+offset]
            special_tokens: [CLS], [SEP], [DATASET]
            '''
            pred_ids = pred_ids[len_offset:len(seq) + len_offset]
            gold_ids = gold_ids[len_offset:len(seq) + len_offset]
            str_pred = ' '.join(ids2chunks(pred_ids, seq))
            str_gold = ' '.join(ids2chunks(gold_ids, seq))
            preds.append(str_pred)
            golds.append(str_gold)
        return preds, golds

    @staticmethod
    def freeze_bert(pretrained_model):
        for param in pretrained_model.base_model.parameters():
            param.requires_grad = False

    @staticmethod
    def get_bert_char_inputs(inputs):
        return dict({
            data: inputs[data]
            for data in inputs if data in BERT_CHAR_INPUT_KEYS
        })

    @staticmethod
    def get_char_input_org_ids(inputs):
        return inputs[CHAR_ORG_IDS]

    @staticmethod
    def resize_bert_embeddings(pretrained_model, vocab_size: int,
                               org_vocab_size: int):
        if vocab_size != org_vocab_size:
            pretrained_model.resize_token_embeddings(vocab_size)

    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = argparse.ArgumentParser(parents=[parent_parser],
                                         add_help=False)
        parser.add_argument('--bert-mode',
                            choices=['none', 'concat', 'sum', 'sum-all'],
                            default='none')
        parser.add_argument('--freeze-bert', action='store_true')
        parser.add_argument('--bert-lr', type=float, default=2e-5)
        parser.add_argument(
            '--pretrained-model',
            choices=[
                'data/ptm/latte-mc3-bert-base-japanese-char-v2',
                'data/ptm/latte-mc7-bert-base-chinese',
                'data/ptm/latte-mc5-bert-base-multilingual-cased',
                'data/ptm/bert-base-japanese-char-v2',
                'data/ptm/bert-base-chinese',
                'data/ptm/bert-base-multilingual-cased', 'bert-base-chinese',
                'cl-tohoku/bert-base-japanese-char-v2',
                'bert-base-multilingual-cased'
            ])
        parser.add_argument('--model-max-seq-length', type=int, default=-1)
        parser.add_argument(
            '--node-comp-type',
            choices=['none', 'init', 'extend', 'avg', 'concat', 'wv', 'wvc'],
            default='none')
        parser.add_argument('--unuse-context-char-node', action='store_true')
        return parser
