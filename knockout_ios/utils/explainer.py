import logging

from copy import deepcopy
from sys import stdout

import numpy as np
import pandas as pd
import shutup
from tqdm import tqdm

# TODO: Excessive wittgenstein frame.append deprecation warnings
#  currently trying to suppress just with -Wignore
import wittgenstein as lw
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score, roc_auc_score, roc_curve
from sklearn.model_selection import train_test_split, GridSearchCV, TimeSeriesSplit

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

        # Replace blank spaces in _by_case column names with underscores and keep only 1 event per caseid
        # (attributes remain the same throughout the case)
        _by_case.columns = [c.replace(' ', '_') for c in _by_case.columns]

        if skip_temporal_holdout:
            # This simple train/test split is incorrect; need time-aware splits; only used for baseline comparison against Illya

            # split with the same proportion as Illya, 20% for final metrics calculation
            _by_case, test = train_test_split(_by_case, test_size=.2)
            test = test.drop(columns=columns_to_ignore, errors='ignore')

            if balance_classes:
                # with the remaining 80%, balance ko'd/non ko'd cases
                train = _by_case.groupby("knocked_out_case")
                train = train.apply(lambda x: x.sample(train.size().min()).reset_index(drop=True))
                train = train.reset_index(drop=True)
                train = train.drop(columns=columns_to_ignore, errors='ignore')
            else:
                train = _by_case.drop(columns=columns_to_ignore, errors='ignore')

        else:
            # Temporal holdout: take first 80% of cases, so that cases in test set happen after those in training set
            _by_case = _by_case.sort_values(by=[globalColumnNames.SIMOD_START_TIMESTAMP_COLUMN_NAME])
            train, test = train_test_split(_by_case, test_size=.2, shuffle=False)

            # make sure that all cases in the test set are after the last timestamp of the training set
            last_timestamp = train.iloc[-1][globalColumnNames.SIMOD_START_TIMESTAMP_COLUMN_NAME]
            assert test[globalColumnNames.SIMOD_START_TIMESTAMP_COLUMN_NAME].max() > last_timestamp

            test = test.drop(columns=columns_to_ignore, errors='ignore')
            train = train.drop(columns=columns_to_ignore, errors='ignore')

        if algorithm == "RIPPER":
            ruleset_model, ruleset_params = RIPPER_wrapper(train, activity, max_rules=max_rules,
                                                           max_rule_conds=max_rule_conds,
                                                           max_total_conds=max_total_conds,
                                                           k=k,
                                                           n_discretize_bins=n_discretize_bins,
                                                           dl_allowance=dl_allowance,
                                                           prune_size=prune_size,
                                                           grid_search=grid_search,
                                                           param_grid=param_grid)
        # fallback to "IREP"
        else:
            ruleset_model, ruleset_params = IREP_wrapper(train, activity, max_rules=max_rules,
                                                         max_rule_conds=max_rule_conds,
                                                         max_total_conds=max_total_conds,
                                                         n_discretize_bins=n_discretize_bins,
                                                         prune_size=prune_size,
                                                         grid_search=grid_search,
                                                         param_grid=param_grid)

        # Performance metrics
        x_test = test.drop(['knocked_out_case'], axis=1)
        y_test = test['knocked_out_case']

        support = calc_knockout_ruleset_support(ruleset_model, _by_case,
                                                available_cases_before_ko=available_cases_before_ko[activity])
        confidence = calc_knockout_ruleset_confidence(activity, ruleset_model, _by_case)

        metrics = {
            'support': support,
            'confidence': confidence,
            'condition_count': ruleset_model.ruleset_.count_conds(),
            'rule_count': ruleset_model.ruleset_.count_rules(),
            'accuracy': ruleset_model.score(x_test, y_test, accuracy_score),
            'precision': ruleset_model.score(x_test, y_test, precision_score),
            'recall': ruleset_model.score(x_test, y_test, recall_score),
            'f1_score': ruleset_model.score(x_test, y_test, f1_score)
        }
        try:
            metrics['roc_auc_score'] = ruleset_model.score(x_test, y_test, roc_auc_score)
            metrics['roc_curve'] = ruleset_model.score(x_test, y_test, roc_curve)
        except Exception:
            metrics['roc_auc_score'] = 0.5
            metrics['roc_curve'] = np.array([0, 0.5, 1]), np.array([0, 0.5, 1]), None

        rulesets[activity] = (
            ruleset_model,
            ruleset_params,
            metrics
        )
        stdout.flush()

    return rulesets


def RIPPER_wrapper(train, activity, max_rules=None,
                   max_rule_conds=None,
                   max_total_conds=None,
                   k=2,
                   n_discretize_bins=7,
                   dl_allowance=64,
                   prune_size=0.33,
                   grid_search=True,
                   param_grid=None
                   ):
    params = {"max_rules": max_rules,
              "max_rule_conds": max_rule_conds,
              "max_total_conds": max_total_conds,
              "k": k,
              "dl_allowance": dl_allowance,
              "n_discretize_bins": n_discretize_bins,
              "prune_size": prune_size}

    if grid_search:
        if param_grid is None:
            param_grid = {"prune_size": [0.2, 0.33, 0.5], "k": [1, 2, 4], "dl_allowance": [16, 32, 64],
                          "n_discretize_bins": [4, 8, 12]}
        try:
            optimized_params = do_grid_search(lw.RIPPER(max_rules=max_rules,
                                                        max_rule_conds=max_rule_conds,
                                                        max_total_conds=max_total_conds), train,
                                              activity,
                                              algorithm="RIPPER",
                                              param_grid=param_grid)

            params.update(optimized_params)
        except:
            logging.error(f"Grid search failed for {activity}. Using default parameters.")

    # Fit "final model" with optimized parameters
    ruleset_model = lw.RIPPER(**params)

    with shutup.mute_warnings:
        ruleset_model.fit(train, class_feat='knocked_out_case')

    return ruleset_model, params


def IREP_wrapper(train, activity, max_rules=None,
                 max_rule_conds=None,
                 max_total_conds=None,
                 n_discretize_bins=7,
                 prune_size=0.33,
                 grid_search=True,
                 param_grid=None
                 ):
    params = {"max_rules": max_rules,
              "max_rule_conds": max_rule_conds,
              "max_total_conds": max_total_conds,
              "n_discretize_bins": n_discretize_bins,
              "prune_size": prune_size}

    if grid_search:

        if param_grid is None:
            param_grid = {"prune_size": [0.2, 0.33, 0.5], "n_discretize_bins": [10, 20, 30]}
        else:
            # to keep only relevant parameters for IREP
            _param_grid = {}
            if "n_discretize_bins" in param_grid:
                _param_grid["n_discretize_bins"] = param_grid["n_discretize_bins"]
            if "prune_size" in param_grid:
                _param_grid["prune_size"] = param_grid["prune_size"]
            param_grid = _param_grid

        try:
            optimized_params = do_grid_search(lw.IREP(max_rules=max_rules,
                                                      max_rule_conds=max_rule_conds,
                                                      max_total_conds=max_total_conds), train, activity,
                                              algorithm="IREP",
                                              param_grid=param_grid)
            params.update(optimized_params)
        except:
            logging.error(f"Grid search failed for {activity}. Using default parameters.")

    # Fit "final model" with optimized parameters
    ruleset_model = lw.IREP(**params)

    with shutup.mute_warnings:
        ruleset_model.fit(train, class_feat='knocked_out_case')

    return ruleset_model, params


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
