"""Independently check FD004 exported predictions and source labels."""
from pathlib import Path
import argparse
import json
import numpy as np
import pandas as pd
from src.prepare_data import load_trajectory, load_test_rul


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--raw-directory', type=Path, required=True)
    args = parser.parse_args()
    report = json.loads(Path('data/processed/fd004_results.json').read_text())
    predictions = pd.read_csv('data/processed/fd004_predictions.csv')
    assert not predictions.isna().any().any()
    assert not predictions.duplicated(['evaluation_split', 'arm', 'unit_id', 'cycle']).any()
    for split, rows in predictions.groupby('evaluation_split'):
        expected = 248 if split == 'official_test' else 200
        assert rows.groupby('arm').size().eq(expected).all()
        assert rows.groupby(['unit_id', 'cycle']).size().eq(4).all()
        assert rows.groupby(['unit_id', 'cycle']).true_rul.nunique().eq(1).all()
        for arm, arm_rows in rows.groupby('arm'):
            errors = arm_rows.predicted_rul - arm_rows.true_rul
            metrics = report[split]['arms'][arm]
            np.testing.assert_allclose(errors, arm_rows.error)
            np.testing.assert_allclose(errors.abs().mean(), metrics['mae'])
            np.testing.assert_allclose(np.sqrt((errors**2).mean()), metrics['rmse'])
    raw = args.raw_directory
    test = load_trajectory(raw / 'test_FD004.txt')
    labels = load_test_rul(raw / 'RUL_FD004.txt', test)
    final = predictions[predictions.evaluation_split == 'official_test']
    for _, rows in final.groupby('arm'):
        np.testing.assert_array_equal(rows.sort_values('unit_id').true_rul,
                                      labels.sort_values('unit_id').true_rul)
    history = report['tcn_validation_history']
    assert report['selected_tcn_epoch'] == min(history,
        key=lambda row: row['validation_mae_cycles'])['epoch']
    print('PASS: all four arms, source labels, paired endpoints and metrics agree.')


if __name__ == '__main__':
    main()
