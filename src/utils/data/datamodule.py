import copy
from collections import Counter
import fasttext
import logging
from pathlib import Path
import pickle
import pytorch_lightning as pl
from tqdm import tqdm
import transformers

from utils.data.dataset import (Dataset, BertDataset, LSTMDataset, split_data)

from utils import graph

# uncomment to debug
logging.basicConfig(level='DEBUG')
logger = logging.getLogger(__name__)


class DataModule(pl.LightningDataModule):
    def __init__(self, train_file, valid_file, test_file, data_root_dir,
                 ext_dic_file, batch_size, pretrained_model,
                 model_max_seq_length, wv_model_path, lang, normalize_unicode,
                 max_token_length, min_token_freq_for_trie, node_comp_type,
                 graph_dropout, train_split_ratio, generate_unigram_node,
                 include_dataset_token, unc_token_ratio, include_unc_token,
                 include_lattice, build_dynamic_graph, include_valid_vocab,
                 exclude_special_token_map, seed,
                 use_binary):
        super(DataModule, self).__init__()
        self.train_file = train_file
        self.valid_file = valid_file
        self.test_file = test_file
        self.data_root_dir = data_root_dir
        self.ext_dic_file = ext_dic_file
        self.batch_size = batch_size
        self.pretrained_model = pretrained_model
        self.model_max_seq_length = model_max_seq_length
        self.wv_model_path = wv_model_path
        self.lang = lang
        self.normalize_unicode = normalize_unicode
        self.max_token_length = max_token_length
        self.min_token_freq_for_trie = min_token_freq_for_trie
        self.node_comp_type = node_comp_type
        self.graph_dropout = graph_dropout
        self.train_split_ratio = train_split_ratio
        self.generate_unigram_node = generate_unigram_node
        self.include_dataset_token = include_dataset_token
        self.unc_token_ratio = unc_token_ratio
        self.include_unc_token = include_unc_token
        self.include_lattice = include_lattice
        self.build_dynamic_graph = build_dynamic_graph
        self.include_valid_vocab = include_valid_vocab
        self.exclude_special_token_map = exclude_special_token_map
        self.seed = seed
        self.use_binary = use_binary

        self.train_data = None
        self.valid_data = None
        self.test_data = None
        self.ext_vocab_data = None

        self.vocab = None
        self.char_vocab = None
        self.node_vocab = None
        self._train_vocab = None

        self.train_set = None
        self.valid_set = None
        self.test_set = None

        self.tokenizer = None
        self.wv_model = None
        self.trie = None
        self.dataset_token = None
        self.unc_token = '[UNC]'

    def setup(self, stage=None):
        if self.use_binary and self._binary_filepath_exists():
            self._setup_from_binary()
            if self.node_comp_type == 'wv':
                self._setup_wv_model()
        else:
            # bert
            if self.pretrained_model is not None:
                self._setup_data(model_type='bert')
                self._setup_ext_vocab_data()
                self._setup_trie()
                self._setup_vocab(model_type='bert')
                self._setup_tokenizer(extend=self.node_comp_type == 'extend')
                if self.node_comp_type == 'wv' or self.node_comp_type == 'wvc':
                    self._setup_wv_model()
                params = self._get_bert_dataset_params(
                    tokenizer=self.tokenizer)
                self.train_set = BertDataset(data=self.train_data,
                                             train=True,
                                             **params)
                self.valid_set = BertDataset(data=self.valid_data,
                                             train=False,
                                             **params)
                self.test_set = BertDataset(
                    data=self.test_data, train=False, **
                    params) if self.test_data else None
            # lstm
            elif self.pretrained_model is None:
                self._setup_data(model_type='lstm')
                self._setup_ext_vocab_data()
                self._setup_trie()
                self._setup_vocab(model_type='lstm')
                if self.wv_model_path is not None:
                    self._setup_wv_model()
                params = self._get_lstm_dataset_params()
                self.train_set = LSTMDataset(data=self.train_data,
                                             train=True,
                                             **params)
                self.valid_set = LSTMDataset(data=self.valid_data,
                                             train=False,
                                             **params)
                self.test_set = LSTMDataset(
                    data=self.test_data, train=False, **
                    params) if self.test_file else None
            else:
                raise NotImplementedError

            if self.use_binary:
                self._save_data_to_binary()

    def _setup_data(self, model_type: str = 'bert'):
        '''
        Load datasets, normalise lines, split lines into list,
        augment (add) dataset token, e.g., [BCCWJ]  to each line,
        if specified
        '''
        train_file_path = self.data_root_dir / self.train_file
        train_dataset_token = train_file_path.stem.split('.')[0]
        '''valid file'''
        valid_file_path, valid_dataset_token = None, None
        if self.valid_file is not None:
            valid_file_path = self.data_root_dir / self.valid_file
            valid_dataset_token = valid_file_path.stem.split('.')[0]
            if self.train_split_ratio > 0.0:
                logger.info('--train-split-ratio is not being used')
        '''test file'''
        test_file_path, test_dataset_token = None, None
        if self.test_file is not None:
            test_file_path = self.data_root_dir / self.test_file
            test_dataset_token = test_file_path.stem.split('.')[0]

        if self.include_dataset_token:
            dataset_tokens = list(
                dict.fromkeys([
                    train_dataset_token, valid_dataset_token,
                    test_dataset_token
                ]))
            dataset_tokens = [dtoken for dtoken in dataset_tokens if dtoken]
            if len(dataset_tokens) != 1:
                raise AssertionError(
                    'Multiple dataset tokens is not supported')
            if model_type == 'bert':
                self.dataset_token = '[' + dataset_tokens[0].upper() + ']'
            elif model_type == 'lstm':
                self.unc_token = '<' + self.unc_token.lower().strip('[]') + '>'
                self.dataset_token = '<' + dataset_tokens[0].lower() + '>'
            else:
                raise AssertionError

        # train/valid data
        self.train_data = Dataset.load_file_and_augment(
            train_file_path,
            self.normalize_unicode,
            self.lang,
            self.dataset_token,
            include_unc_token=self.include_unc_token,
            unc_token=self.unc_token,
            unc_token_ratio=(self.unc_token_ratio
                             if self.include_unc_token else 0.0))

        if self.valid_file is not None:
            self.valid_data = Dataset.load_file_and_augment(
                valid_file_path,
                self.normalize_unicode,
                self.lang,
                self.dataset_token,
                include_unc_token=self.include_unc_token,
                unc_token=self.unc_token,
                unc_token_ratio=(self.unc_token_ratio
                                 if self.include_unc_token else 0.0))
        elif self.valid_file is None and self.train_split_ratio > 0.0:
            self.train_data, self.valid_data = split_data(
                self.train_data,
                split_ratio=self.train_split_ratio,
                seed=self.seed,
                shuffle=True if self.seed is not None else False)
        else:
            raise AssertionError(
                '--valid-file is required for --train-split-ratio')

        # test data
        if self.test_file is not None:
            self.test_data = Dataset.load_file_and_augment(
                test_file_path,
                self.normalize_unicode,
                self.lang,
                self.dataset_token,
                include_unc_token=self.include_unc_token,
                unc_token=self.unc_token,
                unc_token_ratio=(self.unc_token_ratio
                                 if self.include_unc_token else 0.0))

    def _setup_ext_vocab_data(self):
        if self.ext_dic_file:
            ext_vocab_data = Dataset.read_dataset(self.ext_dic_file)
            ext_vocab_data = [(Dataset.normalize_line(line, self.lang)
                               if self.normalize_unicode else line)
                              for line in ext_vocab_data]
            self.ext_vocab_data = sorted(set(ext_vocab_data))

    def _setup_trie(self):
        data = copy.deepcopy(self.train_data)
        if self.include_valid_vocab:
            if self.valid_data is None:
                raise AssertionError(
                    '`--include-valid-vocab` requires `--valid-file` ' +
                    'or `--train-split-ratio`')
            data.extend(self.valid_data)
        if self.ext_vocab_data is not None:
            data.extend(self.ext_vocab_data)

        token_counter = Counter()
        for s in data:
            token_counter.update(s)
            if self.generate_unigram_node:
                '''
                s is ext_vocab_data if str
                    (ext_vocab_data has no dataset_token)
                s is line if list
                '''
                is_ext_vocab = isinstance(s, str)
                if (self.include_dataset_token and not is_ext_vocab):
                    s = s[1:]
                unigrams = list(''.join(s))
                token_counter.update(unigrams)
        token_counter = Counter({
            token: freq
            for token, freq in token_counter.items()
            if freq >= self.min_token_freq_for_trie or len(token) == 1
        })
        '''generate unigrams for valid/test'''
        if self.generate_unigram_node:
            data = []
            if (not self.include_valid_vocab and self.valid_data):
                data.extend(self.valid_data)
            if self.test_data:
                data.extend(self.test_data)
            if data:
                for s in data:
                    if self.include_dataset_token:
                        s = s[:1]
                    unigrams = list(''.join(s))
                    token_counter.update(unigrams)
                token_counter = Counter({
                    token: freq
                    for token, freq in token_counter.items()
                    if freq >= self.min_token_freq_for_trie or len(token) == 1
                })

        trie = graph.Trie()
        for token in tqdm(
                token_counter.keys(),
                desc=('Construct Trie(train={}, valid={}, test={}, ' +
                      'train_u={}, valid_u={}, test_u={}, ext_dic={})').format(
                          self.train_data is not None,
                          self.include_valid_vocab, False,
                          self.generate_unigram_node,
                          self.generate_unigram_node,
                          self.generate_unigram_node, self.ext_vocab_data
                          is not None)):
            trie.add_token(token)
        trie._build_trie()
        self.trie = trie

    def _setup_vocab(self, model_type: str = 'bert'):
        '''
        No any test gold (word) token
        '''
        # params for constructing vocab
        params = self._get_vocab_params(model_type)

        # collection of char_vocab, node_vocab, and _train_vocab
        '''
        construct unigram vocab from train/valid/test/ext
        include words from train/valid(optional)/ext generated by self.trie,
        and unigrams from train/valid/test/ext
        '''
        token_counter = Counter()
        data = copy.deepcopy(self.train_data)
        if self.valid_data:
            data.extend(self.valid_data)
        if self.test_data:
            data.extend(self.test_data)
        if self.ext_vocab_data:
            data.extend(self.ext_vocab_data)
        for s in tqdm(data,
                      desc=('Setup vocab(train_u={}, valid_u={}, test_u={}, ' +
                            'edic={}, trie={})').format(
                                self.train_data is not None, self.valid_data
                                is not None, self.test_data is not None,
                                self.ext_vocab_data is not None, self.trie
                                is not None)):
            '''
            s is ext_vocab_data if str
                (ext_vocab_data has no dataset_token)
            s is line if list
            '''
            is_ext_vocab = isinstance(s, str)
            '''discard dataset token'''
            if (self.include_dataset_token and not is_ext_vocab):
                s = s[1:]
            unigrams = list(''.join(s))
            token_counter.update(unigrams)
            '''
            search word from train trie
            - found tokens should include unigram, train vocab, and ext vocab
            '''
            tokens = [
                token for token, span in self.trie.search_tokens_from_trie(s)
            ]
            token_counter.update(tokens)
        self.vocab = Dataset.construct_vocab(tokens=token_counter.keys(),
                                             **params)

        # char vocab
        '''construct char vocab for char_embed and org_ids'''
        token_counter = Counter()
        for s in tqdm(
                data,
                desc=('Setup char vocab(train_u={}, valid_u={}, test_u={}, ' +
                      'ext_dic_u={}, use_trie={})').format(
                          self.train_data is not None, self.valid_data
                          is not None, self.test_data is not None,
                          self.ext_vocab_data is not None, False)):
            is_ext_vocab = isinstance(s, str)
            '''discard dataset token'''
            if (self.include_dataset_token and not is_ext_vocab):
                s = s[1:]
            unigrams = list(''.join(s))
            token_counter.update(unigrams)
        self.char_vocab = Dataset.construct_vocab(tokens=token_counter.keys(),
                                                  **params)

        # train vocab for oov evaluation
        '''construct train vocab for oov-recall evaluation'''
        token_counter = Counter()
        train_data = copy.deepcopy(self.train_data)
        for s in tqdm(train_data,
                      desc=('Setup train vocab(train={}, valid={}, test={}, ' +
                            'ext_dict={})').format(self.train_data is not None,
                                                   False, False, False)):
            if self.include_dataset_token:
                s = s[1:]
            token_counter.update(s)
        self._train_vocab = Dataset.construct_vocab(
            tokens=token_counter.keys(), **params)

        # node vocab for lattice
        if self.include_lattice:
            if (self.include_valid_vocab or self.ext_vocab_data):
                token_counter = Counter()
                if self.include_valid_vocab:
                    if self.valid_data is None:
                        raise AssertionError(
                            '`--include-valid-vocab` requires `--valid-file` '
                            + 'or `--train-split-ratio`')
                    train_data.extend(self.valid_data)
                if self.ext_vocab_data:
                    train_data.extend(self.ext_vocab_data)
                '''transfrom valid/test vocab into char'''
                if self.generate_unigram_node:
                    if self.valid_data and not self.include_valid_vocab:
                        train_data.extend([
                            ([self.dataset_token] if self.include_dataset_token
                             else []) + list(''.join(
                                 s[(1 if self.include_dataset_token else 0):]))
                            for s in self.valid_data
                        ])
                    if self.test_data:
                        train_data.extend([
                            ([self.dataset_token] if self.include_dataset_token
                             else []) + list(''.join(
                                 s[(1 if self.include_dataset_token else 0):]))
                            for s in self.test_data
                        ])

                for s in tqdm(
                        train_data,
                        desc=(
                            'Setup node vocab(train={}, valid={}, valid_u={}, '
                            + 'test={}, test_u={}, ext_dic={})').format(
                                self.train_data is not None,
                                self.include_valid_vocab,
                                self.generate_unigram_node, False,
                                self.generate_unigram_node, self.ext_vocab_data
                                is not None)):
                    token_counter.update(s)
                    if self.generate_unigram_node:
                        is_ext_vocab = isinstance(s, str)
                        if (self.include_dataset_token and not is_ext_vocab):
                            s = s[1:]
                        unigrams = list(''.join(s))
                        '''find unique unigrams'''
                        unigrams = list(set(unigrams) - set(s))
                        token_counter.update(unigrams)

                self.node_vocab = Dataset.construct_vocab(
                    tokens=token_counter.keys(), **params)
            else:
                self.node_vocab = copy.deepcopy(self._train_vocab)

    def _setup_tokenizer(self, extend: bool = False):
        tokenizer_bin_path = Path(self.pretrained_model) / 'tokenizer.pkl'
        # own pretrained model
        if tokenizer_bin_path.exists():
            '''initialise tokenizer from binary'''
            logger.info(f'Load binary tokenizer: {tokenizer_bin_path}')
            with open(tokenizer_bin_path, 'rb') as f:
                self.tokenizer = pickle.load(f)
        # huggingface pretrained model
        else:
            '''use_fast=True for enabling is_split_into_words'''
            logger.info(f'Load tokenizer: {self.pretrained_model}')
            self.tokenizer = transformers.AutoTokenizer.from_pretrained(
                self.pretrained_model, use_fast=False)

        special_tokens = {
            'bos_token': '[BOS]',
            'eos_token': '[EOS]',
        }
        self.tokenizer.add_special_tokens(special_tokens)
        '''node_comp_type == `extend`, extend bert-tokenizer vocab'''
        if extend:
            self.tokenizer.add_tokens(list(self.vocab.keys()))

    def _setup_wv_model(self):
        if self.wv_model_path is None:
            raise AssertionError
        wv_model_path_str = str(self.wv_model_path.resolve())
        self.wv_model = fasttext.load_model(wv_model_path_str)

    def _setup_data_from_binary(self):
        (self.train_set, self.valid_set, self.test_set, self.tokenizer,
         self.trie, self.vocab, self.char_vocab, self.node_vocab,
         self._train_vocab) = self._load_data_from_binary()

    def _get_binary_filepath(self) -> Path:
        if self.pretrained_model:
            base_filename = self.train_file.stem.split('.')[0]
            pretrained_model_name = self.pretrained_model.replace('/', '.')
            ext = '.bin'
            binary_filepath = Path(
                self.data_root_dir /
                (base_filename + '.' + pretrained_model_name + ext))
            return binary_filepath
        else:
            return Path(self.data_root_dir / (self.train_file.stem + '.bin'))

    def _load_data_from_binary(self) -> Dataset:
        binary_filepath = self._get_binary_filepath()
        logger.info(f'Load binary data: {binary_filepath}')
        with open(binary_filepath, 'rb') as f:
            _binary_data = pickle.load(f)
            train_set = _binary_data['train_set']
            valid_set = _binary_data['valid_set']
            test_set = _binary_data['test_set']
            tokenizer = _binary_data['tokenizer']
            trie = _binary_data['trie']
            vocab = _binary_data['vocab']
            char_vocab = _binary_data['char_vocab']
            node_vocab = _binary_data['node_vocab']
            _train_vocab = _binary_data['_train_vocab']
        return (train_set, valid_set, test_set, tokenizer, trie, vocab,
                char_vocab, node_vocab, _train_vocab)

    def _save_data_to_binary(self) -> bool:
        binary_filepath = self._get_binary_filepath()
        logger.info(f'Save binaly data: {binary_filepath}')
        if not binary_filepath.exists():
            with open(binary_filepath, 'wb') as f:
                obj = dict({
                    'train_set': self.train_set,
                    'valid_set': self.valid_set,
                    'test_set': self.test_set,
                    'tokenizer': self.tokenizer,
                    'trie': self.trie,
                    'vocab': self.vocab,
                    'char_vocab': self.node_vocab,
                    'node_vocab': self.node_vocab,
                    '_train_vocab': self._train_vocab,
                })
                pickle.dump(obj, f)
                return True
        return False

    def _binary_filepath_exists(self):
        binary_filepath = self._get_binary_filepath()
        if binary_filepath.exists():
            return True
        return False

    def _get_bert_dataset_params(self, tokenizer=None):
        return dict({
            'char_vocab': self.char_vocab,
            'node_vocab': self.node_vocab,
            'tokenizer': tokenizer,
            'model_max_seq_length': self.model_max_seq_length,
            'lang': self.lang,
            'normalize_unicode': self.normalize_unicode,
            'max_token_length': self.max_token_length,
            'trie_org': self.trie,
            'dataset_token': self.dataset_token,
            'include_lattice': self.include_lattice,
            'node_comp_type': self.node_comp_type,
            'graph_dropout': self.graph_dropout,
            'build_dynamic_graph': self.build_dynamic_graph,
            'exclude_special_token_map': self.exclude_special_token_map,
            'generate_unigram_node': self.generate_unigram_node,
            'use_custom_encoder': True,
        })

    def _get_lstm_dataset_params(self):
        return dict({
            'char_vocab': self.char_vocab,
            'node_vocab': self.node_vocab,
            'lang': self.lang,
            'normalize_unicode': self.normalize_unicode,
            'max_token_length': self.max_token_length,
            'trie_org': self.trie,
            'dataset_token': self.dataset_token,
            'include_lattice': self.include_lattice,
            'node_comp_type': self.node_comp_type,
            'graph_dropout': self.graph_dropout,
            'build_dynamic_graph': self.build_dynamic_graph,
            'exclude_special_token_map': self.exclude_special_token_map,
            'generate_unigram_node': self.generate_unigram_node,
        })

    def _get_vocab_params(self, model_type: str = 'bert'):
        if model_type == 'bert':
            return dict({
                'pad_token':
                '[PAD]',
                'unk_token':
                '[UNK]',
                'bos_token':
                '[BOS]',
                'eos_token':
                '[EOS]',
                'cls_token':
                '[CLS]',
                'sep_token':
                '[SEP]',
                'dataset_token':
                self.dataset_token,
                'padding_idx':
                0,
                'unk_idx':
                1,
                'bos_idx':
                2,
                'eos_idx':
                3,
                'cls_idx':
                4,
                'sep_idx':
                5,
                'sort_token':
                True,
            })
        elif model_type == 'lstm':
            return dict({
                'pad_token':
                '<pad>',
                'unk_token':
                '<unk>',
                'bos_token':
                '<bos>',
                'eos_token':
                '<eos>',
                'cls_token':
                '<cls>',
                'sep_token':
                '<sep>',
                'dataset_token':
                self.dataset_token,
                'padding_idx':
                0,
                'unk_idx':
                1,
                'bos_idx':
                2,
                'eos_idx':
                3,
                'cls_idx':
                4,
                'sep_idx':
                5,
                'sort_token':
                True,
            })
        else:
            raise ValueError
