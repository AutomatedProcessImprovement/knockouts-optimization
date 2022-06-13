import logging
import math

from copy import deepcopy
from sys import stdout

import numpy as np
import pandas as pd
import shutup
from tqdm import tqdm

import wittgenstein as lw
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score, roc_auc_score, roc_curve, \
    balanced_accuracy_score, make_scorer
from sklearn.model_selection import train_test_split, GridSearchCV, TimeSeriesSplit, cross_val_score, KFold, \
    StratifiedKFold

from knockout_ios.utils.constants import globalColumnNames
from knockout_ios.utils.metrics import calc_knockout_ruleset_support, calc_knockout_ruleset_confidence


def find_ko_rulesets(log_df, ko_activities, available_cases_before_ko, columns_to_ignore=None, algorithm='IREP',
                     max_rules=None, max_rule_conds=None, max_total_conds=None, k=2, n_discretize_bins=7,
                     dl_allowance=2, prune_size=0.8, grid_search=True, param_grid=None, skip_temporal_holdout=False,
                     balance_classes=False,
                     ):
    if columns_to_ignore is None:
        columns_to_ignore = []

    rulesets = {}

    for activity in tqdm(ko_activities, desc='Finding rules of Knockout Activities'):
        # Bucketing approach: Keep all cases, apply mask to those not knocked out by current activity
        _by_case = deepcopy(log_df)
        _by_case["knockout_activity"] = np.where(_by_case["knockout_activity"] == activity, activity, False)
        _by_case["knocked_out_case"] = np.where(_by_case["knockout_activity"] == activity, True, False)

        # Workaround to avoid any confusion with attributes that sometimes have whitespaces and sometimes have _
        _by_case.columns = [c.replace(' ', '_') for c in _by_case.columns]
        columns_to_ignore = [c.replace(' ', '_') for c in columns_to_ignore]

        train, test = split_train_test(skip_temporal_holdout, balance_classes, _by_case)

        # Subset of log with all columns to be used in confidence & support metrics & avoid information leak
        _by_case_only_cases_in_test = test

        # After splitting and/or balancing as requested, drop all columns that are not needed for the analysis
        test = test.drop(columns=columns_to_ignore, errors='ignore')
        train = train.drop(columns=columns_to_ignore, errors='ignore')

        ruleset_model, ruleset_params = fit_ruleset_model(algorithm, max_rules, max_rule_conds, max_total_conds,
                                                          k, dl_allowance, n_discretize_bins, prune_size,
                                                          grid_search, activity, train, param_grid)

        # Performance metrics
        metrics = get_performance_metrics(test, _by_case, columns_to_ignore,
                                          ruleset_model, available_cases_before_ko, activity,
                                          _by_case_only_cases_in_test, skip_temporal_holdout)

        rulesets[activity] = (
            ruleset_model,
            ruleset_params,
            metrics
        )
        stdout.flush()

    return rulesets


def split_train_test(skip_temporal_holdout: bool, balance_classes: bool, _by_case: pd.DataFrame):
    # Data splitting techniques:
    # - Simple train/test split for comparison against Illya's paper (baseline) + optional class balance
    # - Temporal holdout for every other log (we need time-aware splits), no class balancing nor shuffling
    if skip_temporal_holdout:

        # Split with the same proportion as Illya, 20% for final metrics calculation
        _by_case, test = train_test_split(_by_case, test_size=.2)

        if balance_classes:
            # with the remaining 80%, balance ko'd/non ko'd cases (as in Illya's paper)
            train = _by_case.groupby("knocked_out_case")
            train = train.apply(lambda x: x.sample(train.size().min()).reset_index(drop=True))
            train = train.reset_index(drop=True)
        else:
            train = _by_case

    else:
        # Temporal holdout: take first 80% of cases for train set, so that cases in test set happen after those
        _by_case = _by_case.sort_values(by=[globalColumnNames.SIMOD_START_TIMESTAMP_COLUMN_NAME])
        train, test = train_test_split(_by_case, test_size=.2, shuffle=False)

        # make sure that all cases in the test set are after the last timestamp of the training set
        last_timestamp = train.iloc[-1][globalColumnNames.SIMOD_START_TIMESTAMP_COLUMN_NAME]
        assert test[globalColumnNames.SIMOD_START_TIMESTAMP_COLUMN_NAME].max() > last_timestamp

    return train, test


def fit_ruleset_model(algorithm: str, max_rules: int, max_rule_conds: int, max_total_conds: int, k: int,
                      dl_allowance: float, n_discretize_bins: int, prune_size: float, grid_search: bool,
                      activity: str, train: pd.DataFrame, param_grid: dict):
    if algorithm == "RIPPER":
        ruleset_params = {"max_rules": max_rules,
                          "max_rule_conds": max_rule_conds,
                          "max_total_conds": max_total_conds,
                          "k": k,
                          "dl_allowance": dl_allowance,
                          "n_discretize_bins": n_discretize_bins,
                          "prune_size": prune_size}

        if grid_search:
            try:
                optimized_params = do_grid_search(lw.RIPPER(max_rules=max_rules,
                                                            max_rule_conds=max_rule_conds,
                                                            max_total_conds=max_total_conds), train,
                                                  activity,
                                                  algorithm="RIPPER",
                                                  param_grid=param_grid)

                ruleset_params.update(optimized_params)
            except:
                logging.error(f"Grid search failed for {activity}. Using default parameters.")

        ruleset_model = lw.RIPPER(**ruleset_params)

    # fallback to "IREP"
    else:
        ruleset_params = {"max_rules": max_rules,
                          "max_rule_conds": max_rule_conds,
                          "max_total_conds": max_total_conds,
                          "n_discretize_bins": n_discretize_bins,
                          "prune_size": prune_size}

        if grid_search:
            try:
                optimized_params = do_grid_search(lw.IREP(max_rules=max_rules,
                                                          max_rule_conds=max_rule_conds,
                                                          max_total_conds=max_total_conds), train, activity,
                                                  algorithm="IREP",
                                                  param_grid=param_grid)
                ruleset_params.update(optimized_params)
            except:
                logging.error(f"Grid search failed for {activity}. Using default parameters.")

        ruleset_model = lw.IREP(**ruleset_params)

    # Fit "final model" with optimized parameters
    with shutup.mute_warnings:
        ruleset_model.fit(train, class_feat='knocked_out_case')

    return ruleset_model, ruleset_params


def do_grid_search(ruleset_model, dataset, activity, algorithm="RIPPER", quiet=True, param_grid=None,
                   skip_temporal_holdout=False):
    dataset = deepcopy(dataset)

    if not quiet:
        print(f"\nPerforming {algorithm} parameter grid search")

    # Dummify categorical features and booleanize class values for sklearn compatibility
    x_train = dataset.drop(['knocked_out_case'], axis=1)
    x_train = pd.get_dummies(x_train, columns=x_train.select_dtypes('object').columns)

    y_train = dataset['knocked_out_case']
    y_train = y_train.map(lambda x: 1 if x else 0)

    # Search best parameter combination; criteria: f1 score (balanced fbeta score / same importance to recall and
    # precision)
    # Source: https://scikit-learn.org/stable/modules/model_evaluation.html#scoring-parameter
    #
    # By default it uses the model's score() function (for classification this is sklearn.metrics.accuracy_score)
    # Source: https://scikit-learn.org/stable/modules/grid_search.html#tips-for-parameter-search

    # More about splitting techniques here:
    # https://scikit-learn.org/stable/modules/cross_validation.html#cross-validation

    if skip_temporal_holdout:
        # Incorrect approach: non-time-aware splits; only to be used for baseline comparison against Illya
        # sgkf = StratifiedGroupKFold(n_splits=5, shuffle=True)
        grid = GridSearchCV(estimator=ruleset_model, param_grid=param_grid, scoring="f1", n_jobs=-1, cv=5)
    else:
        # Temporal split is the correct way to handle time-series data
        # In this way, we can still use grid search + cross validation, with time-aware splits
        # Source: https://stackoverflow.com/a/46918197/8522453
        #         https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.TimeSeriesSplit.html
        tscv = TimeSeriesSplit(gap=0, max_train_size=None, test_size=None, n_splits=3)
        grid = GridSearchCV(estimator=ruleset_model, param_grid=param_grid, scoring="f1", n_jobs=-1, cv=tscv)

    with shutup.mute_warnings:
        grid.fit(x_train, y_train)

    if not quiet:
        print(f"Best {algorithm} parameters for \"{activity}\": {grid.best_params_}")

    return grid.best_params_


def get_performance_metrics(test: pd.DataFrame, _by_case: pd.DataFrame, columns_to_ignore: list[str],
                            ruleset_model: lw.RIPPER, available_cases_before_ko: dict, activity: str,
                            _by_case_only_cases_in_test: pd.DataFrame, skip_temporal_holdout: bool):
    # Performance metrics
    x_test = test.drop(['knocked_out_case'], axis=1)
    y_test = test['knocked_out_case']

    X = _by_case.drop(columns_to_ignore + ['knocked_out_case'], axis=1, errors='ignore')
    y = _by_case['knocked_out_case']

    support = calc_knockout_ruleset_support(ruleset_model, _by_case,
                                            available_cases_before_ko=available_cases_before_ko[activity])

    confidence = calc_knockout_ruleset_confidence(activity, ruleset_model, _by_case_only_cases_in_test)

    if skip_temporal_holdout:
        metrics = {
            'support': support,
            'confidence': confidence,
            'condition_count': ruleset_model.ruleset_.count_conds(),
            'rule_count': ruleset_model.ruleset_.count_rules(),
            'accuracy': cross_val_score(ruleset_model, x_test, y_test, cv=5, scoring="accuracy").mean(),
            'balanced_accuracy': cross_val_score(ruleset_model, x_test, y_test, cv=5,
                                                 scoring="balanced_accuracy").mean(),
            'precision': cross_val_score(ruleset_model, x_test, y_test, cv=5, scoring="precision").mean(),
            'recall': cross_val_score(ruleset_model, x_test, y_test, cv=5, scoring="recall").mean(),
            'f1_score': cross_val_score(ruleset_model, x_test, y_test, cv=5, scoring="f1").mean(),
        }

        # Get roc metrics with cross validation, as in Illya's paper
        # includes a workaround for greatly imbalanced datasets such as envpermit,
        # where roc_auc_score gets heavily biased. Fallback to values of random classifier.
        try:
            if math.isclose(metrics["confidence"], 0):
                raise Exception
            metrics['roc_auc_cv'] = cross_val_score(ruleset_model, X, y, cv=5, scoring=make_scorer(roc_auc_score),
                                                    error_score=0).mean()
        except Exception:
            metrics['roc_auc_cv'] = 0.5

        try:
            if math.isclose(metrics["confidence"], 0):
                raise Exception
            metrics['roc_curve_cv'] = get_roc_curve_cv(ruleset_model, X, y, cv=5)
        except Exception:
            metrics['roc_curve_cv'] = np.array([0, 0.5, 1]), np.array([0, 0.5, 1]), np.array([0, 0, 0])

    else:
        metrics = {
            'support': support,
            'confidence': confidence,
            'condition_count': ruleset_model.ruleset_.count_conds(),
            'rule_count': ruleset_model.ruleset_.count_rules(),
            'balanced_accuracy': ruleset_model.score(x_test, y_test, balanced_accuracy_score),
            'accuracy': ruleset_model.score(x_test, y_test, accuracy_score),
            'precision': ruleset_model.score(x_test, y_test, precision_score),
            'recall': ruleset_model.score(x_test, y_test, recall_score),
            'f1_score': ruleset_model.score(x_test, y_test, f1_score)
        }
        try:
            metrics['roc_auc_score'] = ruleset_model.score(x_test, y_test, roc_auc_score)
        except Exception:
            metrics['roc_auc_score'] = 0.5

        try:
            metrics['roc_curve'] = ruleset_model.score(x_test, y_test, roc_curve)
        except Exception:
            metrics['roc_curve'] = np.array([0, 0.5, 1]), np.array([0, 0.5, 1]), np.array([0, 0, 0])

    return metrics


def get_roc_curve_cv(ruleset_model, X, y, cv):
    # Splits & computes roc_curve per each configuration,
    # then returns mean of all (fpr, tpr, thresholds)

    # kf = KFold(n_splits=cv)
    # kf = TimeSeriesSplit(gap=0, max_train_size=None, test_size=None, n_splits=cv)
    kf = StratifiedKFold(n_splits=cv)
    curves: list[tuple] = []
    max_fpr_len, max_tpr_len, max_thresholds_len = 0, 0, 0

    for train, test in kf.split(X, y):
        X_train, X_test = X.iloc[train], X.iloc[test]
        y_train, y_test = y.iloc[train], y.iloc[test]
        try:
            if len(curves) > 0:
                max_fpr_len = max([len(curve[0]) for curve in curves])
                max_tpr_len = max([len(curve[1]) for curve in curves])
                max_thresholds_len = max([len(curve[2]) for curve in curves])

            ruleset_model.fit(X_train, y=y_train)
            if ruleset_model.ruleset_.isuniversal() or ruleset_model.ruleset_.isnull():
                raise Exception

            # TODO: investigate more about thresholds, predict proba / decisions, how to get more points in the curve...
            curve = ruleset_model.score(X_test, y_test, roc_curve)
            for e in curve:
                if np.isnan(e).any():
                    raise Exception

            if (len(curve[0]) < max_fpr_len) or (len(curve[1]) < max_tpr_len) or (len(curve[2]) < max_thresholds_len):
                raise Exception

            curves.append(curve)

        except Exception:
            pass

    # if all curves had an issue, raise Exception
    # to signal the caller to return default "random" classifier curve
    if len(curves) == 0:
        raise Exception

    # average all the elements of the roc curve tuples
    fprs = np.array([t[0] for t in curves])
    tprs = np.array([t[1] for t in curves])
    thresholds = np.array([t[2] for t in curves])

    avg = np.average(np.array(fprs), axis=0), \
          np.average(np.array(tprs), axis=0), \
          np.average(np.array(thresholds), axis=0)

    return avg
