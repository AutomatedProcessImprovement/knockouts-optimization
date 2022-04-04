import pickle
from copy import deepcopy
from sys import stdout

# TODO: Excessive wittgenstein frame.append deprecation warnings
#  currently suppresed just with -Wignore
import numpy as np
import pandas as pd
import wittgenstein as lw
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score
from sklearn.model_selection import train_test_split, GridSearchCV
from wittgenstein.abstract_ruleset_classifier import AbstractRulesetClassifier

from knockout_ios.utils.constants import *


def get_rejected_cases_with_known_rejection_activity(log_df, rejection_activity,
                                                     activity_column=SIMOD_LOG_READER_ACTIVITY_COLUMN_NAME,
                                                     case_id_column=SIMOD_LOG_READER_CASE_ID_COLUMN_NAME):
    print(f"Filtering cases containing the activity: \"{rejection_activity}\"")

    rejected_case_ids = log_df.loc[log_df[activity_column] == rejection_activity]
    rejected_cases = log_df.loc[log_df[case_id_column].isin(rejected_case_ids[case_id_column])]
    grouped_rejected_cases = rejected_cases.groupby(case_id_column)

    rejected_cases_ids = list(grouped_rejected_cases.groups.keys())

    def is_rejected(case):
        return (case[activity_column] == rejection_activity).count() > 1

    for id in rejected_cases_ids:
        _case = grouped_rejected_cases.get_group(id)
        assert is_rejected(_case)

    return rejected_cases


def is_adjacent(e1, e2, _list):
    for x, y in zip(_list, _list[1:]):
        if (e1, e2) == (x, y):
            return True

    return False


def directly_follows_for_all_cases(a1, a2, _by_case):
    by_case = deepcopy(_by_case)
    directly_follows = False
    for _case in by_case.groups.keys():
        _case_df = by_case.get_group(_case)
        _sorted_case_activities = list(_case_df.sort_values("time:timestamp")['concept:name'])
        if (a1 not in _sorted_case_activities) or (a2 not in _sorted_case_activities):
            continue
        directly_follows = is_adjacent(a1, a2, _sorted_case_activities)
        if not directly_follows:
            break

    return directly_follows


def read_rule_discovery_result(config_file_name, cache_dir):
    binary_file = open(f'{cache_dir}/{config_file_name}_clfs', 'rb')
    clfs = pickle.load(binary_file)
    binary_file.close()
    return clfs


def dump_rule_discovery_result(clfs, config_file_name, cache_dir):
    binary_file = open(f'{cache_dir}/{config_file_name}_clfs', 'wb')
    pickle.dump(clfs, binary_file)
    binary_file.close()


def calc_support(activity: str, ruleset_model: AbstractRulesetClassifier, log: pd.DataFrame, dummies=False):
    predicted_ko = ruleset_model.predict(log)
    log['predicted_ko'] = predicted_ko

    if dummies:
        freq_x_and_y = \
            log[(log['predicted_ko']) & (log[f'knockout_activity_{activity}'])].shape[0]
    else:
        freq_x_and_y = log[(log['predicted_ko']) & (log['knockout_activity'] == activity)].shape[0]

    N = log.shape[0]

    support = freq_x_and_y / N

    return support


def calc_confidence(activity: str, ruleset_model: AbstractRulesetClassifier, log: pd.DataFrame, dummies=False):
    predicted_ko = ruleset_model.predict(log)
    log['predicted_ko'] = predicted_ko

    if dummies:
        freq_x_and_y = \
            log[(log['predicted_ko']) & (log[f'knockout_activity_{activity}'])].shape[0]
    else:
        freq_x_and_y = log[(log['predicted_ko']) & (log['knockout_activity'] == activity)].shape[0]
    freq_x = sum(predicted_ko)

    confidence = freq_x_and_y / freq_x

    return confidence


def find_ko_rulesets(log_df, ko_activities, config_file_name, cache_dir,
                     force_recompute=True,
                     columns_to_ignore=None,
                     algorithm='IREP',
                     max_rules=None,
                     max_rule_conds=None,
                     max_total_conds=None,
                     k=2,
                     n_discretize_bins=5,
                     dl_allowance=64,
                     prune_size=0.33,
                     grid_search=True,
                     param_grid=None,
                     bucketing_approach="A"
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

        for activity in ko_activities:
            try:
                if bucketing_approach == "A":
                    # Bucketing approach A: Keep only cases knocked out by current activity and non-knocked out ones
                    _by_case = log_df[log_df['knockout_activity'].isin([activity, False])]
                elif bucketing_approach == "B":
                    # Bucketing approach B: Keep all cases, apply mask to those not knocked out by current activity
                    _by_case = deepcopy(log_df)
                    _by_case["knockout_activity"] = np.where(_by_case["knockout_activity"] == activity, activity, False)
                    _by_case["knocked_out_case"] = np.where(_by_case["knockout_activity"] == activity, True, False)

                # Replace blank spaces in _by_case column names with underscores and keep only 1 event per caseid
                # (attributes remain the same throughout the case)
                _by_case.columns = [c.replace(' ', '_') for c in _by_case.columns]
                _by_case = _by_case.drop_duplicates(subset=[SIMOD_LOG_READER_CASE_ID_COLUMN_NAME])

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

                    support = calc_support(activity, ruleset_model, _by_case, dummies=True)
                    confidence = calc_confidence(activity, ruleset_model, _by_case, dummies=True)
                else:
                    support = calc_support(activity, ruleset_model, _by_case)
                    confidence = calc_confidence(activity, ruleset_model, _by_case)

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
                    }
                )

            except Exception as e:
                print(e)
                continue

            finally:
                stdout.flush()

        dump_rule_discovery_result(rulesets, f"{config_file_name}_{algorithm}", cache_dir=cache_dir)

    return rulesets


def RIPPER_wrapper(train, activity, max_rules=None,
                   max_rule_conds=None,
                   max_total_conds=10,
                   k=2,
                   n_discretize_bins=5,
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
                              prune_size=prune_size,
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
            param_grid = {"prune_size": [0.33, 0.5, 0.7], "k": [1, 2, 5, 10],
                          "n_discretize_bins": [3, 5, 8]}
        ruleset_model, optimized_params = do_grid_search(ruleset_model, train, activity, algorithm="RIPPER",
                                                         param_grid=param_grid)
        params.update(optimized_params)

    return ruleset_model, params


def IREP_wrapper(train, activity, max_rules=None,
                 max_rule_conds=None,
                 max_total_conds=10,
                 n_discretize_bins=5,
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
            param_grid = {"prune_size": [0.33, 0.5, 0.7], "n_discretize_bins": [3, 5, 8]}

        ruleset_model, optimized_params = do_grid_search(lw.IREP(), train, activity, algorithm="IREP",
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

    # TODO: decide if using balanced_accuracy alongside f1_score is better

    grid = GridSearchCV(estimator=ruleset_model, param_grid=param_grid, scoring=["f1", "balanced_accuracy"],
                        refit="f1", n_jobs=-1)
    grid.fit(x_train, y_train)

    if not quiet:
        print(f"Best {algorithm} parameters for \"{activity}\": {grid.best_params_}")

    return grid.best_estimator_, grid.best_params_
