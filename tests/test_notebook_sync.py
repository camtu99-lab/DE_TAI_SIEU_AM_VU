import os, pytest


def test_notebook_embeds_identical_source():
    """Notebook và web dùng CÙNG mã: các cell %%writefile phải bằng đúng tệp trong src/ (không cần torch)."""
    import json
    nbp = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'notebooks', 'BreastVision_XAI_V4_BUS_VERIFY_Kaggle.ipynb')
    if not os.path.exists(nbp):
        pytest.skip('chưa dựng notebook')
    nb = json.load(open(nbp, encoding='utf-8'))
    emb = {''.join(c['source']).split('\n', 1)[0].replace('%%writefile bsv4/', ''): ''.join(c['source']).split('\n', 1)[1] for c in nb['cells'] if c['cell_type'] == 'code' and ''.join(c['source']).startswith('%%writefile')}
    root = os.path.dirname(os.path.dirname(__file__))
    for rel in ('src/pipeline.py', 'src/failure_risk.py', 'src/counterfactual_verifier.py', 'src/benchmark.py'):
        assert emb[rel].strip() == open(os.path.join(root, rel), encoding='utf-8').read().strip(), rel
