"""Tests for scripts/table1_seeds.py pure helpers (no torch / data needed)."""

import importlib.util
import math
import os

_PATH = os.path.join(os.path.dirname(__file__), '..', 'scripts', 'table1_seeds.py')
_spec = importlib.util.spec_from_file_location('table1_seeds', _PATH)
t1 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(t1)


def test_row_key():
    assert t1.row_key('runs/pinn_l12_gcn_mt') == 'gcn_mt'
    assert t1.row_key('runs/pinn_l12_lgsta_nodecls/') == 'lgsta_nodecls'
    assert t1.row_key('runs/other') == 'other'


def test_args_to_cli_only_non_defaults_and_skips_seed():
    defaults = {'arch': 'gat', 'epochs': 200, 'residual': False, 'seed': 42,
                'output_dir': 'runs', 'physics_lambda_eq': 0.0, 'pretrained': None}
    ref = {'arch': 'lgsta', 'epochs': 200, 'residual': True, 'seed': 7,
           'output_dir': 'runs/x', 'physics_lambda_eq': 0.1, 'pretrained': None,
           'removed_option': 1}
    cli = t1.args_to_cli(ref, defaults)
    assert cli == ['--arch', 'lgsta', '--physics_lambda_eq', '0.1', '--residual']


def test_task_label():
    assert t1.task_label({'task': 'node_cls', 'physics_lambda_stress': 0.1,
                          'physics_lambda_eq': 0.1}) == 'NC+L1+L2'
    assert t1.task_label({'task': 'multitask'}) == 'MT'


def test_summarize_sample_std_and_nan():
    m, s, n = t1.summarize([0.8, 0.9, float('nan'), None])
    assert n == 2 and math.isclose(m, 0.85)
    assert math.isclose(s, math.sqrt(0.005))
    assert t1.summarize([]) == (None, None, 0)
    assert t1.summarize([0.5]) == (0.5, 0.0, 1)


def test_latex_has_rows_in_draft_order():
    per_row = {
        'gcn_mt': [{'opt_f1': 0.82, 'auc': 0.99, 'reg_rmse_mm': 0.7, 'task_label': 'MT+L1+L2'},
                   {'opt_f1': 0.84, 'auc': 0.99, 'reg_rmse_mm': 0.9, 'task_label': 'MT+L1+L2'}],
        'lgsta_nodecls': [{'opt_f1': 0.9, 'auc': 0.999, 'task_label': 'NC+L1+L2'}],
    }
    tex = t1.to_latex(t1.aggregate_rows(per_row), '0, 1')
    assert tex.index('LGSTA (ours)') < tex.index('GCN')
    assert '0.830\\,$\\pm$\\,0.014' in tex
    assert 'seeds 0, 1' in tex
    assert tex.count('\\\\') >= 4
