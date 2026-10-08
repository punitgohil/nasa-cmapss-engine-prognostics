"""Frozen four-method replication with late access to FD004 test labels."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd
from src.audit_data import audit_trajectory_file, sha256
from src.prepare_data import (load_trajectory, add_training_rul,
    split_development_and_calibration, ConditionAwareTransformer,
    global_feature_columns, condition_aware_feature_columns, load_test_rul,
    final_test_observations)
from src.temporal_features import add_temporal_features, temporal_feature_columns
from src.evaluate_temporal_features import (split_phase2_training_validation,
    sample_validation_endpoints, engine_balanced_weights, make_point_model,
    prediction_frame, metric_summary)
from src.evaluate_tcn import (fit_scalers, train_with_early_stopping,
    train_fixed_epochs, predict_dataset, attach_test_labels)
from src.tcn_model import TrajectorySequenceDataset


def check_trajectory(path):
    report = audit_trajectory_file(path)
    for key in ('field_count_issue_count', 'numeric_parse_issue_count',
                'duplicate_unit_cycle_count', 'cycle_continuity_issue_count'):
        if report[key]:
            raise ValueError(f'{path.name}: {key}={report[key]}')
    return report


def compare(frame):
    metrics = {arm: metric_summary(rows) for arm, rows in frame.groupby('arm')}
    errors = frame.groupby(['unit_id', 'arm']).absolute_error.mean().unstack()
    comparisons = {}
    for baseline, candidate in [('global', 'condition_aware'),
                                ('condition_aware', 'engineered_history'),
                                ('engineered_history', 'temporal_cnn')]:
        differences = (errors[baseline] - errors[candidate]).to_numpy()
        rng = np.random.default_rng(2026)
        draws = [rng.choice(differences, len(differences), replace=True).mean()
                 for _ in range(10000)]
        comparisons[f'{candidate}_over_{baseline}'] = {
            'mae_improvement': float(differences.mean()),
            'bootstrap_95_interval': np.quantile(draws, [.025, .975]).tolist()}
    return {'arms': metrics, 'paired_comparisons': comparisons}


def run(raw, output):
    train_path = raw / 'train_FD004.txt'
    test_path = raw / 'test_FD004.txt'
    train_audit = check_trajectory(train_path)
    train = add_training_rul(load_trajectory(train_path))
    development, reserved = split_development_and_calibration(train)
    fitting, validation = split_phase2_training_validation(development)
    columns = condition_aware_feature_columns()
    arms = {'global': global_feature_columns(), 'condition_aware': columns,
            'engineered_history': columns + temporal_feature_columns()}
    transform = ConditionAwareTransformer().fit(fitting)
    fit_frame = add_temporal_features(transform.transform(fitting))
    val_frame = add_temporal_features(transform.transform(validation))
    val_endpoints = sample_validation_endpoints(val_frame)
    val_predictions = []
    for arm, features in arms.items():
        model = make_point_model().fit(fit_frame[features], fit_frame.rul,
                                     sample_weight=engine_balanced_weights(fit_frame))
        val_predictions.append(prediction_frame(arm, model, val_endpoints, features, 'rul'))
    scaler, mean, scale, weights = fit_scalers(fit_frame, columns, 'rul')
    fit_ds = TrajectorySequenceDataset(fit_frame, columns, scaler, 'rul', mean, scale,
                                      sample_weights=weights)
    val_ds = TrajectorySequenceDataset(val_frame, columns, scaler, 'rul', mean, scale,
                                      endpoint_indices=val_endpoints.index)
    print('Selecting TCN epoch using training-engine validation only.', flush=True)
    model, epoch, history = train_with_early_stopping(fit_ds, val_ds, mean, scale)
    val_predictions.append(predict_dataset(model, val_ds, mean, scale).drop(columns='row_index'))
    del fit_ds, val_ds, model
    print(f'Validation selected epoch {epoch}; fitting final models.', flush=True)
    transform = ConditionAwareTransformer().fit(development)
    final_frame = add_temporal_features(transform.transform(development))
    final_models = {arm: make_point_model().fit(final_frame[features], final_frame.rul,
                    sample_weight=engine_balanced_weights(final_frame))
                    for arm, features in arms.items()}
    scaler, mean, scale, weights = fit_scalers(final_frame, columns, 'rul')
    final_ds = TrajectorySequenceDataset(final_frame, columns, scaler, 'rul', mean,
                                        scale, sample_weights=weights)
    final_tcn, losses = train_fixed_epochs(final_ds, epoch)
    del final_ds
    print('All models fixed; opening official test labels for evaluation.', flush=True)
    test_audit = check_trajectory(test_path)
    test = add_temporal_features(transform.transform(load_trajectory(test_path)))
    rul = load_test_rul(raw / 'RUL_FD004.txt', test)
    endpoints = final_test_observations(test, rul)
    test_predictions = [prediction_frame(arm, final_models[arm], endpoints,
                        features, 'true_rul') for arm, features in arms.items()]
    labelled, indices = attach_test_labels(test, rul)
    test_ds = TrajectorySequenceDataset(labelled, columns, scaler, 'true_rul', mean,
                                       scale, endpoint_indices=indices)
    test_predictions.append(predict_dataset(final_tcn, test_ds, mean, scale)
                            .drop(columns='row_index'))
    val_all = pd.concat(val_predictions, ignore_index=True)
    test_all = pd.concat(test_predictions, ignore_index=True)
    report = {'dataset': 'NASA C-MAPSS FD004',
        'protocol': 'docs/fd004-protocol.md',
        'splits': {'development': int(development.unit_id.nunique()),
                   'reserved': int(reserved.unit_id.nunique()),
                   'fitting': int(fitting.unit_id.nunique()),
                   'validation': int(validation.unit_id.nunique())},
        'selected_tcn_epoch': epoch, 'tcn_validation_history': history,
        'tcn_final_training_losses': losses,
        'provenance': {'train': train_audit['sha256'], 'test': test_audit['sha256'],
                       'rul': sha256(raw / 'RUL_FD004.txt')},
        'validation': compare(val_all), 'official_test': compare(test_all)}
    all_predictions = pd.concat([val_all.assign(evaluation_split='validation'),
        test_all.assign(evaluation_split='official_test')], ignore_index=True)
    if all_predictions.isna().any().any():
        raise ValueError('Missing values in FD004 exported predictions.')
    output.mkdir(parents=True, exist_ok=True)
    (output / 'fd004_results.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    all_predictions.to_csv(output / 'fd004_predictions.csv', index=False)
    print(json.dumps(report['official_test'], indent=2), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--raw-directory', type=Path, required=True)
    parser.add_argument('--output-directory', type=Path, default=Path('data/processed'))
    args = parser.parse_args()
    run(args.raw_directory, args.output_directory)
