"""Regression test: the lattice path (BiGAT and word nodes) must receive gradients.

Runs one real batch of the bundled BEST2010 sample through BertTagger.
Set LATTE_TEST_PTM to a local BERT directory to avoid downloading
bert-base-multilingual-cased.

    python tests/test_gradient_flow.py        (or: python -m pytest tests/)
"""
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, 'src'))

SAMPLE_ARGS = [
    '--data-root-dir', '.',
    '--train-file', 'data/samples/best2010.train.sample20.seg.sl',
    '--valid-file', 'data/samples/best2010.valid.sample10.seg.sl',
    '--test-file', 'data/samples/best2010.test.sample10.seg.sl',
    '--batch-size', '4', '--bert-mode', 'sum', '--optimized-decay',
    '--scheduler', '--lang', 'th', '--normalize-unicode',
    '--criterion-type', 'crf', '--metric-type', 'word-bin-th',
    '--attn-comp-type', 'wavg', '--max-token-length', '4',
    '--node-comp-type', 'none',
    '--ext-dic-file', 'data/dict/samples/hse-thai-lexitron.vocab.sample.sl',
    '--graph-dropout', '0.1', '--seed', '112'
]


def test_lattice_path_receives_gradients():
    os.chdir(ROOT)
    os.cpu_count = lambda: 4  # one dataloader worker
    import train
    from taggers.bert_tagger import BertTagger

    ptm = os.environ.get('LATTE_TEST_PTM', 'bert-base-multilingual-cased')
    sys.argv = ['train.py', '--pretrained-model', ptm] + SAMPLE_ARGS
    args = train.get_args()
    train.set_seeds(args.seed)
    model = BertTagger(args)
    model.train()

    xs, ys = next(iter(model.train_dataloader()))
    loss = model._compute_active_loss(xs, ys, model.forward(xs))
    loss.backward()

    no_grad = [
        n for n, p in model.gnn.named_parameters()
        if p.grad is None or not bool(p.grad.abs().sum() > 0)
    ]
    assert not no_grad, f'GNN parameters without gradient: {no_grad}'

    word_ids = [
        d.token_id.view(-1)[[i for i, (s, e) in enumerate(d.span) if e - s > 1]]
        for d in xs['lattice'].to_data_list()
    ]
    word_ids = __import__('torch').cat(word_ids).unique()
    grad = model.bert.embeddings.word_embeddings.weight.grad
    assert grad is not None and bool(
        (grad[word_ids].abs().sum(-1) > 0).all()), \
        'word-node embeddings received no gradient'


if __name__ == '__main__':
    test_lattice_path_receives_gradients()
    print('ok: BiGAT and word nodes receive gradients')
