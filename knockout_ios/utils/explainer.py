import pickle

from copy import deepcopy
from sys import stdout

import numpy as np
import pandas as pd
from tqdm import tqdm

from knockout_ios.utils.constants import *

# TODO: Excessive wittgenstein frame.append deprecation warnings
#  currently trying to suppress just with -Wignore
import wittgenstein as lw
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score, roc_auc_score
from sklearn.model_selection import train_test_split, GridSearchCV

from knockout_ios.utils.metrics import calc_knockout_ruleset_support, calc_knockout_ruleset_confidence


def read_rule_discovery_result(config_file_name, cache_dir):
    binary_file = open(f'{cache_dir}/{config_file_name}_clfs', 'rb')
    clfs = pickle.load(binary_file)
    binary_file.close()
    return clfs


def dump_rule_discovery_result(clfs, config_file_name, cache_dir):
    binary_file = open(f'{cache_dir}/{config_file_name}_clfs', 'wb')
    pickle.dump(clfs, binary_file)
    binary_file.close()


def find_ko_rulesets(log_df, ko_activities, config_file_name, cache_dir,
                     available_cases_before_ko,
                     force_recompute=True,
                     columns_to_ignore=None,
                     algorithm='IREP',
                     max_rules=None,
                     max_rule_conds=None,
                     max_total_conds=None,
                     k=2,
                     n_discretize_bins=7,
                     dl_allowance=2,
                     prune_size=0.8,
                     grid_search=True,
                     param_grid=None
                     ):
    if columns_to_ignore is None:
        columns_to_ignore = []

    try:
        if force_recompute:
            raise FileNotFoundError

        rulesets = read_rule_discovery_result(f"{config_file_name}_{algorithm}", cache_dir=cache_dir)
        return rulesets

    except FileNotFoundError:
        rulesets = {}

        for activity in tqdm(ko_activities, desc='Finding rules of Knockout Activities'):
            # Bucketing approach: Keep all cases, apply mask to those not knocked out by current activity
            _by_case = deepcopy(log_df)
            _by_case["knockout_activity"] = np.where(_by_case["knockout_activity"] == activity, activity, False)
            _by_case["knocked_out_case"] = np.where(_by_case["knockout_activity"] == activity, True, False)

            # Replace blank spaces in _by_case column names with underscores and keep only 1 event per caseid
            # (attributes remain the same throughout the case)
            _by_case.columns = [c.replace(' ', '_') for c in _by_case.columns]

            train, test = train_test_split(_by_case.drop(columns=columns_to_ignore, errors='ignore'),
                                           test_size=.33)

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
            # default to "IREP"
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

            if grid_search:
                # Pre-process to conform to sklearn required format
                x_test = pd.get_dummies(x_test, columns=x_test.select_dtypes('object').columns)
                y_test = y_test.map(lambda x: 1 if x else 0)

                _by_case = _by_case.drop(
                    columns=[PM4PY_CASE_ID_COLUMN_NAME, SIMOD_LOG_READER_CASE_ID_COLUMN_NAME],
                    errors='ignore')
                _by_case = pd.get_dummies(_by_case, columns=_by_case.select_dtypes('object').columns)

            support = calc_knockout_ruleset_support(activity, ruleset_model, _by_case,
                                                    available_cases_before_ko=available_cases_before_ko[activity],
                                                    processed_with_pandas_dummies=grid_search)
            confidence = calc_knockout_ruleset_confidence(activity, ruleset_model, _by_case,
                                                          processed_with_pandas_dummies=grid_search)

            rulesets[activity] = (
                ruleset_model,
                ruleset_params,
                {
                    'support': support,
                    'confidence': confidence,
                    'condition_count': ruleset_model.ruleset_.count_conds(),
                    'rule_count': ruleset_model.ruleset_.count_rules(),
                    'accuracy': ruleset_model.score(x_test, y_test, accuracy_score),
                    'precision': ruleset_model.score(x_test, y_test, precision_score),
                    'recall': ruleset_model.score(x_test, y_test, recall_score),
                    'f1_score': ruleset_model.score(x_test, y_test, f1_score),
                    'roc_auc_score': ruleset_model.score(x_test, y_test, roc_auc_score),
                }
            )

            stdout.flush()

        dump_rule_discovery_result(rulesets, f"{config_file_name}_{algorithm}", cache_dir=cache_dir)

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
    ruleset_model = lw.RIPPER(max_rules=max_rules,
                              max_rule_conds=max_rule_conds,
                              max_total_conds=max_total_conds,
                              k=k,
                              n_discretize_bins=n_discretize_bins,
                              dl_allowance=dl_allowance,
                              prune_size=prune_size
                              )

    ruleset_model.fit(train, class_feat='knocked_out_case')
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
        ruleset_model, optimized_params = do_grid_search(lw.RIPPER(max_rules=max_rules,
                                                                   max_rule_conds=max_rule_conds,
                                                                   max_total_conds=max_total_conds), train, activity,
                                                         algorithm="RIPPER",
                                                         param_grid=param_grid)
        params.update(optimized_params)

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

        ruleset_model, optimized_params = do_grid_search(lw.IREP(max_rules=max_rules,
                                                                 max_rule_conds=max_rule_conds,
                                                                 max_total_conds=max_total_conds), train, activity,
                                                         algorithm="IREP",
                                                         param_grid=param_grid)
        params.update(optimized_params)

    else:
        ruleset_model = lw.IREP(max_rules=max_rules,
                                max_rule_conds=max_rule_conds,
                                max_total_conds=max_total_conds,
                                n_discretize_bins=n_discretize_bins,
                                prune_size=prune_size,
                                )

        ruleset_model.fit(train, class_feat='knocked_out_case')

    return ruleset_model, params


def do_grid_search(ruleset_model, train, activity, algorithm="RIPPER", quiet=True, param_grid=None):
    train = deepcopy(train)

    if not quiet:
        print(f"\nPerforming {algorithm} parameter grid search")

    # Dummify categorical features and booleanize your class values for sklearn compatibility
    x_train = train.drop(['knocked_out_case'], axis=1)
    x_train = pd.get_dummies(x_train, columns=x_train.select_dtypes('object').columns)

    y_train = train['knocked_out_case']
    y_train = y_train.map(lambda x: 1 if x else 0)

    # Search best parameter combination; criteria: f1 score (balanced fbeta score / same importance to recall and
    # precision)
    # Source: https://scikit-learn.org/stable/modules/model_evaluation.html#scoring-parameter
    #
    # By default it uses the model's score() function (for classification this is sklearn.metrics.accuracy_score)
    # Source: https://scikit-learn.org/stable/modules/grid_search.html#tips-for-parameter-search

    # TODO: understand better the concept of this scoring function, it works much better than the previous but why
    grid = GridSearchCV(estimator=ruleset_model, param_grid=param_grid, scoring="roc_auc", n_jobs=-1)
    grid.fit(x_train, y_train)

    if not quiet:
        print(f"Best {algorithm} parameters for \"{activity}\": {grid.best_params_}")

    return grid.best_estimator_, grid.best_params_
