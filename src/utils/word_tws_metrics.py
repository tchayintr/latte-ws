from collections import Counter
import torch
from torchmetrics import Metric
from typing import List


class TWS_WORD_F1(Metric):
    '''Micro F1 Score'''

    def __init__(self,
                 ignore_index: int = None,
                 constraint_type: str = 'BMES',
                 dist_sync_on_step=False):
        super().__init__(dist_sync_on_step=dist_sync_on_step)
        self.ignore_index = ignore_index
        if constraint_type.upper() == 'BMES':
            self.constraint_type = constraint_type
            self.ids = dict({0: 'B', 1: 'M', 2: 'E', 3: 'S'})
        else:
            raise NotImplementedError

        # (# w_ref ∩ w_hyp)
        self.add_state('correct_preds',
                       default=torch.tensor(0),
                       dist_reduce_fx='sum')
        # (# w_hyp)
        self.add_state('total_preds',
                       default=torch.tensor(0),
                       dist_reduce_fx='sum')
        # (# w_ref)
        self.add_state('total_correct',
                       default=torch.tensor(0),
                       dist_reduce_fx='sum')
        # (# w_ref + # w_hyp)
        self.add_state('total_tokens',
                       default=torch.tensor(0),
                       dist_reduce_fx='sum')

    def __repr__(self):
        rep = 'TWS_WORD_F1(ignore_index=`{}`, constraint_type=`{}`)'.format(
            self.ignore_index, self.constraint_type.upper())
        return rep

    def __state__(self):
        return 'correct_preds: {},\ntotal_preds: {},\n'.format(
            self.correct_preds,
            self.total_preds) + 'total_correct: {},\ntotal_tokens: {}'.format(
                self.total_correct, self.total_tokens)

    def update(self, preds: List[str], golds: List[str]):
        for _pred, _gold in zip(preds, golds):
            self._update_states(_pred.split(), _gold.split())

    def _update_states(self, pred: List, gold: List):
        if len(pred) > len(gold):
            pred, gold = gold, pred
        p_counter = Counter(pred)
        g_counter = Counter(gold)

        correct_preds = sum(
            min(p_counter[gk], gv) for gk, gv in g_counter.items())
        total_preds = len(pred)
        total_correct = len(gold)
        total_tokens = (len(gold) + len(pred))

        self.correct_preds += correct_preds
        self.total_preds += total_preds
        self.total_correct += total_correct
        self.total_tokens += total_tokens

    def compute(self):
        correct_preds = self.correct_preds
        total_preds = self.total_preds
        total_correct = self.total_correct

        tp, fp, fn = (correct_preds, total_preds - correct_preds,
                      total_correct - correct_preds)
        p = 0.0 if tp + fp == 0 else 1. * tp / (tp + fp)
        r = 0.0 if tp + fn == 0 else 1. * tp / (tp + fn)
        f = 0.0 if p + r == 0 else 2 * p * r / (p + r)
        return f
