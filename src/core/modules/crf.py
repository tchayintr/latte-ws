from allennlp_light.modules.conditional_random_field import (
    ConditionalRandomField, allowed_transitions)
from typing import Dict, Union


class CRF(ConditionalRandomField):

    def __init__(self,
                 constraint_type='BMES',
                 labels: Dict[str, int] = {
                     'B': 0,
                     'M': 1,
                     'E': 2,
                     'S': 3
                 },
                 ignore_index_for_mask: Union[int, bool] = 0):
        labels_for_transition = {i: label for label, i in labels.items()}
        constraints = allowed_transitions(constraint_type=constraint_type,
                                          labels=labels_for_transition)
        super(CRF, self).__init__(len(labels), constraints)
        self.labels = labels
        self.constraints = constraints
        self.ignore_index_for_mask = ignore_index_for_mask

    def __repr__(self):
        rep = 'CRF(labels={}, num_labels={}, constraints={}, ' \
            'ignore_index_for_mask={})'.format(
                self.labels, len(self.labels), (self.constraints is not None),
                self.ignore_index_for_mask)
        return rep


class _CRF(ConditionalRandomField):

    def __init__(self,
                 constraint_type='BMES',
                 labels: Dict[str, int] = {
                     '<pad>': 0,
                     'B': 1,
                     'M': 2,
                     'E': 3,
                     'S': 4
                 },
                 ignore_index: Union[int, bool] = 0):
        labels_for_transition = {i: label for label, i in labels.items()}
        constraints = allowed_transitions(constraint_type=constraint_type,
                                          labels=labels_for_transition)
        super(_CRF, self).__init__(len(labels), constraints)
        self.labels = labels
        self.constraints = constraints
        self.ignore_index = ignore_index

    def __repr__(self):
        rep = 'CRF(labels={}, num_labels={}, constraints={}, ' \
            'ignore_index={})'.format(
                self.labels, len(self.labels), (self.constraints is not None),
                self.ignore_index)
        return rep
