from collections import Counter
from torchmetrics import Metric
from typing import List, Union
import sys
'''
    ACC = # correct_w_hyp * 2 / # w_hyp -> (tp + fp) / ...
    P = (# w_ref ∩ w_hyp) / # w_hyp
    R = (# w_ref ∩ w_hyp) / # w_ref
    F1 = 2PR / (P+R)
'''


class WORD_F1(Metric):

    def __init__(self,
                 average: str = 'micro',
                 delimiter: str = ' ',
                 dist_sync_on_step=False):
        super().__init__(dist_sync_on_step=dist_sync_on_step)
        if average not in ['micro', 'macro']:
            print('Invalid average: `average`: [micro, macro]')
            sys.exit()
        self.average = average
        self.delimiter = delimiter

        self.add_state('corrects', default=[])  # (# w_ref ∩ w_hyp)
        self.add_state('found_corrects', default=[])  # (# w_ref)
        self.add_state('found_guesseds', default=[])  # (# w_hyp)
        self.add_state('found_tokens', default=[])  # (# w_ref + w_hyp)

    def __repr__(self):
        rep = 'WORD_F1(average=`{}`, delimiter=`{}`)'.format(
            self.average, self.delimiter)
        return rep

    def __state__(self):
        return 'correct: {},\nfound_corrects: {},\n'.format(
            sum(self.corrects), sum(self.found_corrects)
        ) + 'found_guesseds: {},\nfound_tokens: {}'.format(
            sum(self.found_guesseds), sum(self.found_tokens))

    def update(self, preds: Union[str, List], golds: Union[str, List]):
        if isinstance(preds, list) and isinstance(golds, list):
            for _pred, _gold in zip(preds, golds):
                self._update_states(_pred.split(), _gold.split())
        elif isinstance(preds, str) and isinstance(golds, str):
            self._update_states(preds.split(), golds.split())
        else:
            print('Invalid inputs')
            sys.exit()
        return

    def _update_states(self, pred, gold):
        self.corrects.append(self._get_n_correct(pred, gold))
        self.found_corrects.append(len(gold))
        self.found_guesseds.append(len(pred))
        self.found_tokens.append(len(pred) + len(gold))

    def _get_n_correct(self, pred: List, gold: List) -> int:
        if len(pred) < len(gold):
            pred, gold = gold, pred
        p_counter = Counter(pred)
        g_counter = Counter(gold)
        return sum(min(p_counter[gk], gv) for gk, gv in g_counter.items())

    def compute(self):
        if self.average == 'micro':
            correct = sum(self.corrects)
            guessed = sum(self.found_guesseds)
            total = sum(self.found_corrects)
            tp, fp, fn = correct, guessed - correct, total - correct
            p = 0 if tp + fp == 0 else 1. * tp / (tp + fp)
            r = 0 if tp + fn == 0 else 1. * tp / (tp + fn)
            f = 0 if p + r == 0 else 2 * p * r / (p + r)
            return f
        elif self.average == 'macro':
            n_data = len(self.found_corrects)
            mp, mr, mf = 0.0, 0.0, 0.0
            mtp, mfp, mfn = 0, 0, 0
            for correct, guessed, total in zip(self.corrects,
                                               self.found_guesseds,
                                               self.found_corrects):
                tp, fp, fn = correct, guessed - correct, total - correct
                p = 0 if tp + fp == 0 else 1. * tp / (tp + fp)
                r = 0 if tp + fn == 0 else 1. * tp / (tp + fn)
                f = 0 if p + r == 0 else 2 * p * r / (p + r)
                mtp += tp
                mfp += fp
                mfn += fn
                mp += p
                mr += r
                mf += f
            tp = mtp / n_data
            fp = mfp / n_data
            fn = mfn / n_data
            p = mp / n_data
            r = mr / n_data
            f = mf / n_data
            return f

    def reset(self):
        super().reset()
        self.corrects = []
        self.found_guesseds = []
        self.found_corrects = []
        self.found_tokens = []
