"""
:model(ConvLSTM vs XGBoost).
 70/15/15  train/val/test,
 XGBoost model, ConvLSTM (EOD ).
"""

import sys
from pathlib import Path
import numpy as np
import pandas as pd
import torch

# 
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
code_dir = ROOT / 'code'
if str(code_dir) not in sys.path:
    sys.path.insert(0, str(code_dir))

#  fairness_analysis patch
import fairness_analysis  # noqa: E402
from fire_equality.datamodules.firetracks_loader import FireTracksDataset  # noqa: E402

fairness_analysis.patch_lightning_checkpoint_loading()


def get_sample_date(sample):
    """(for)."""
    d = sample.get('pixel_date')
    if d is not None:
        try:
            return pd.Timestamp(d).to_pydatetime()
        except Exception:
            pass
    meta = sample.get('metadata') or {}
    d = meta.get('start_date')
    if d is not None:
        try:
            return pd.Timestamp(d).to_pydatetime()
        except Exception:
            pass
    return None


def chronological_split(samples, train_ratio=0.7, val_ratio=0.15, test_ratio=0.15, seed=42):
    """
     train/val/test .
   , test.
    """
    n = len(samples)
    dates = [get_sample_date(s) for s in samples]
    has_date = np.array([d is not None for d in dates])
    order = np.arange(n)
    date_vals = np.array([dates[i] if has_date[i] else pd.Timestamp.max for i in range(n)])
    sort_idx = np.lexsort((order, date_vals))
    n_dated = int(has_date.sum())
    n_train = int(n_dated * train_ratio)
    n_val = int(n_dated * val_ratio)
    train_idx = sort_idx[:n_train]
    val_idx = sort_idx[n_train : n_train + n_val]
    # test =  + 
    test_idx = np.concatenate([
        sort_idx[n_train + n_val : n_dated],
        sort_idx[n_dated:],
    ]).astype(int)
    return train_idx, val_idx, test_idx


def extract_tabular_features(samples, use_mean_and_max=True):
    """
     features [T, H, W, C] .
    use_mean_and_max=True:  -> 16 ; -> 8 .
    """
    feats = []
    for s in samples:
        f = s.get('features')
        if f is None or f.size == 0:
            feats.append(np.full(16 if use_mean_and_max else 8, np.nan))
            continue
        f = np.asarray(f, dtype=np.float64)
        # f: (T, H, W, C)
        if f.ndim != 4:
            feats.append(np.full(16 if use_mean_and_max else 8, np.nan))
            continue
        flat = f.reshape(-1, f.shape[-1])
        mean_c = np.nanmean(flat, axis=0)
        if use_mean_and_max:
            max_c = np.nanmax(flat, axis=0)
            feats.append(np.concatenate([mean_c, max_c]))
        else:
            feats.append(mean_c)
    X = np.array(feats, dtype=np.float64)
    col_mean = np.nanmean(X, axis=0)
    col_mean = np.nan_to_num(col_mean, nan=0.0)
    X = np.where(np.isnan(X), col_mean, X)
    return X


def main():
    import argparse
    import pandas as pd
    from sklearn.metrics import roc_auc_score

    parser = argparse.ArgumentParser(description=':ConvLSTM vs XGBoost ')
    parser.add_argument('--checkpoint', type=str, default=None, help='ConvLSTM ')
    parser.add_argument('--data', type=str, nargs='+', default=None, help='.pth ()')
    parser.add_argument('--output', type=str, default='fairness_results_baselines', help='')
    parser.add_argument('--train-ratio', type=float, default=0.7)
    parser.add_argument('--val-ratio', type=float, default=0.15)
    parser.add_argument('--test-ratio', type=float, default=0.15)
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()

    if not args.checkpoint or not args.data:
        print(' --checkpoint  --data')
        sys.exit(1)

    try:
        import xgboost as xgb
    except ImportError:
        print(' xgboost: pip install xgboost')
        sys.exit(1)

    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)

    # 1) load( fairness_analysis )
    analyzer = fairness_analysis.FairnessAnalyzer(
        checkpoint_path=args.checkpoint,
        data_paths=args.data,
    )
    samples = analyzer.load_data()
    sensitive_attrs = analyzer.extract_sensitive_attributes(samples)
    pop_group = np.array(sensitive_attrs['pop_group'])
    y_all = np.array([s.get('target', 0) for s in samples])

    # 2) 
    train_idx, val_idx, test_idx = chronological_split(
        samples,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        test_ratio=args.test_ratio,
        seed=args.seed,
    )
    print(f': train={len(train_idx)}, val={len(val_idx)}, test={len(test_idx)}')

    # 3) 
    X_all = extract_tabular_features(samples, use_mean_and_max=True)
    X_train, y_train = X_all[train_idx], y_all[train_idx]
    X_val, y_val = X_all[val_idx], y_all[val_idx]
    X_test, y_test = X_all[test_idx], y_all[test_idx]
    pop_test = pop_group[test_idx]

    # 4)  XGBoost
    dtrain = xgb.DMatrix(X_train, label=y_train)
    dval = xgb.DMatrix(X_val, label=y_val)
    dtest = xgb.DMatrix(X_test, label=y_test)
    params = {
        'objective': 'binary:logistic',
        'eval_metric': 'auc',
        'max_depth': 6,
        'eta': 0.1,
        'seed': args.seed,
    }
    evals = [(dtrain, 'train'), (dval, 'val')]
    bst = xgb.train(
        params,
        dtrain,
        num_boost_round=200,
        evals=evals,
        early_stopping_rounds=20,
        verbose_eval=False,
    )
    xgb_proba = bst.predict(dtest)
    xgb_pred = (xgb_proba >= 0.5).astype(np.int64)
    print(f'XGBoost test AUC: {roc_auc_score(y_test, xgb_proba):.4f}')

    # 5) ConvLSTM 
    test_samples = [samples[i] for i in test_idx]
    test_dataset = FireTracksDataset(test_samples, target_type='binary_classification')
    analyzer.load_model_and_predict(test_dataset)
    conv_proba = analyzer.probabilities
    conv_pred = analyzer.predictions
    print(f'ConvLSTM test AUC: {roc_auc_score(y_test, conv_proba):.4f}')

    # 6) ( pop_group )
    valid = pop_test != 'Unknown'
    if valid.sum() < 10:
        print(' pop_group,')
        sys.exit(0)
    y_t = y_test[valid]
    pop_t = pop_test[valid]
    conv_pred_t = conv_pred[valid]
    conv_proba_t = conv_proba[valid]
    xgb_pred_t = xgb_pred[valid]
    xgb_proba_t = xgb_proba[valid]

    print('\n' + '=' * 60)
    print('(population density)')
    print('=' * 60)

    fairness_conv = analyzer.calculate_fairness_metrics(
        y_t, conv_pred_t, conv_proba_t, pop_t, cache_key=None, use_cache=False
    )
    #  XGBoost
    import io
    import contextlib
    fbuf = io.StringIO()
    with contextlib.redirect_stdout(fbuf):
        fairness_xgb = analyzer.calculate_fairness_metrics(
            y_t, xgb_pred_t, xgb_proba_t, pop_t, cache_key=None, use_cache=False
        )

    eod_conv = fairness_conv.get('equalized_odds_difference')
    eod_xgb = fairness_xgb.get('equalized_odds_difference')
    dpd_conv = fairness_conv.get('demographic_parity_difference')
    dpd_xgb = fairness_xgb.get('demographic_parity_difference')

    rows = [
        {'model': 'ConvLSTM', 'EOD': eod_conv, 'DPD': dpd_conv},
        {'model': 'XGBoost', 'EOD': eod_xgb, 'DPD': dpd_xgb},
    ]
    df = pd.DataFrame(rows)
    print(df.to_string(index=False))
    df.to_csv(out_dir / 'fairness_comparison_convlstm_xgboost.csv', index=False)

    # 7) :EOD 
    try:
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(5, 4))
        ax.bar(['ConvLSTM', 'XGBoost'], [eod_conv or 0, eod_xgb or 0], color=['#1f77b4', '#ff7f0e'])
        ax.axhline(0, color='gray', linestyle='-', linewidth=0.8)
        ax.set_ylabel('Equalized Odds Difference')
        ax.set_title('Fairness comparison: ConvLSTM vs XGBoost (population-density groups)')
        plt.tight_layout()
        plt.savefig(out_dir / 'fairness_eod_convlstm_vs_xgboost.png', dpi=150, bbox_inches='tight')
        plt.close()
        print(f': {out_dir / "fairness_eod_convlstm_vs_xgboost.png"}')
    except Exception as e:
        print(f': {e}')

    print(f'\n: {out_dir}')


if __name__ == '__main__':
    main()
