import logging
import os
import time
from concurrent.futures import ProcessPoolExecutor

from copy import deepcopy
from math import log
from typing import Union

import numpy as np
import pandas as pd
import shutup
from catboost import CatBoostClassifier, Pool, cv
from catboost.utils import get_roc_curve, eval_metric
from matplotlib import pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from tqdm import tqdm

import wittgenstein as lw
from sklearn.metrics import precision_score, recall_score, f1_score, RocCurveDisplay, auc, roc_auc_score
from sklearn.model_selection import train_test_split, GridSearchCV, TimeSeriesSplit, StratifiedKFold
from wittgenstein.interpret import interpret_model

from knockout_ios.utils.constants import globalColumnNames
from knockout_ios.utils.metrics import calc_knockout_ruleset_support, calc_knockout_ruleset_confidence


def find_ko_rulesets(log_df, ko_activities, available_cases_before_ko, columns_to_ignore=None,
                     algorithm='IREP',
                     max_rules=None, max_rule_conds=None, max_total_conds=None, k=2, n_discretize_bins=7,
                     dl_allowance=2, prune_size=0.8, grid_search=True, param_grid=None,
                     skip_temporal_holdout=False,
                     balance_classes=False,
                     output_dir="."
                     ):
    if columns_to_ignore is None:
        columns_to_ignore = []

    rulesets = {}

    disable_parallelization = os.getenv('DISABLE_PARALLELIZATION', False)

    if disable_parallelization:

        for activity in tqdm(ko_activities, desc='Finding rules of Knockout Activities (sequential)'):
            result = do_find_rules(deepcopy(log_df), activity, columns_to_ignore, skip_temporal_holdout,
                                   balance_classes, algorithm, max_rules,
                                   max_rule_conds, max_total_conds,
                                   k, dl_allowance, n_discretize_bins, prune_size,
                                   grid_search, param_grid, available_cases_before_ko, output_dir)

            rulesets[result["activity"]] = result["rulesets_data"]

    else:
        with ProcessPoolExecutor() as executor:
            futures = []
            for activity in ko_activities:
                futures.append(
                    executor.submit(do_find_rules, deepcopy(log_df), activity, columns_to_ignore,
                                    skip_temporal_holdout,
                                    balance_classes, algorithm, max_rules,
                                    max_rule_conds, max_total_conds,
                                    k, dl_allowance, n_discretize_bins, prune_size,
                                    grid_search, param_grid, available_cases_before_ko))

            for future in tqdm(futures, desc='Finding rules of Knockout Activities (parallel)'):
                result = future.result()
                rulesets[result["activity"]] = result["rulesets_data"]

    return rulesets


def do_find_rules(_by_case, activity, columns_to_ignore, skip_temporal_holdout, balance_classes, algorithm, max_rules,
                  max_rule_conds, max_total_conds,
                  k, dl_allowance, n_discretize_bins, prune_size,
                  grid_search, param_grid, available_cases_before_ko, output_dir):
    # Bucketing approach: Keep all cases, apply mask to those not knocked out by current activity
    # _by_case = deepcopy(log_df)
    _by_case["knockout_activity"] = np.where(_by_case["knockout_activity"] == activity, activity, False)
    _by_case["knocked_out_case"] = np.where(_by_case["knockout_activity"] == activity, True, False)

    # Workaround to avoid any confusion with attributes that sometimes have whitespaces and sometimes have _
    _by_case.columns = [c.replace(' ', '_') for c in _by_case.columns]
    columns_to_ignore = [c.replace(' ', '_') for c in columns_to_ignore]

    try:
        train, test = split_train_test(skip_temporal_holdout, balance_classes, _by_case)
    except ValueError:
        # Impossible to split & stratify when dataset is too small
        blank_clf = lw.RIPPER()
        blank_clf.init_ruleset("[]", class_feat="", pos_class=0)
        return {"activity": activity, "rulesets_data": (
            blank_clf,
            {},
            {'support': 0,
             'confidence': 0,
             'condition_count': 0,
             'rule_count': 0,
             'precision': 0,
             'recall': 0,
             'f1_score': 0,
             'roc_auc_score': 0}
        )}

    # Subset of log with all columns to be used in confidence & support metrics & avoid information leak
    _by_case_only_cases_in_test = test

    # After splitting and/or balancing as requested, drop all columns that are not needed for the analysis
    test = test.drop(columns=columns_to_ignore, errors='ignore')
    train = train.drop(columns=columns_to_ignore, errors='ignore')

    if algorithm == 'CATBOOST-RIPPER':
        ruleset_model, ruleset_params, cb_classifier, \
        catboost_feature_importances, catboost_auc_score = get_ruleset_from_catboost(
            max_rules, max_rule_conds,
            max_total_conds, k,
            dl_allowance,
            n_discretize_bins, prune_size, train, test, _by_case, columns_to_ignore, activity, output_dir)
    else:
        cb_classifier = None
        ruleset_model, ruleset_params = fit_ruleset_model(algorithm, max_rules, max_rule_conds, max_total_conds,
                                                          k, dl_allowance, n_discretize_bins, prune_size,
                                                          grid_search, activity, train, param_grid)

    # Performance metrics
    metrics = get_performance_metrics(test, _by_case, columns_to_ignore,
                                      ruleset_model, available_cases_before_ko, activity,
                                      _by_case_only_cases_in_test, skip_temporal_holdout, output_dir)

    if not (cb_classifier is None):
        metrics['catboost_auc_score'] = catboost_auc_score
        metrics['catboost_feature_importances'] = catboost_feature_importances

    return {"activity": activity, "rulesets_data": (
        ruleset_model,
        ruleset_params,
        metrics
    )}


def split_train_test(skip_temporal_holdout: bool, balance_classes: bool, _by_case: pd.DataFrame):
    # Data splitting techniques:
    # - Simple train/test split for comparison against Illya's paper (baseline) + optional class balance
    # - Temporal holdout for every other log (we need time-aware splits), no class balancing nor shuffling
    if skip_temporal_holdout:

        # Split with the same proportion as Illya, 20% for final metrics calculation
        _by_case, test = train_test_split(_by_case, test_size=.2, stratify=_by_case["knocked_out_case"])

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
                # strip param_grid to keep only keys present in ruleset_params
                param_grid = {key: param_grid[key] for key in param_grid if key in ruleset_params}

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


def get_ruleset_from_catboost(max_rules, max_rule_conds, max_total_conds, k, dl_allowance, n_discretize_bins,
                              prune_size, train, test, full_dataset, columns_to_ignore, activity, output_dir):
    """
    Prototype of interpretable model extraction: CatBoost classifier then Ruleset
    - https://github.com/imoscovitz/wittgenstein#interpreter-models
    - https://catboost.ai/en/docs/concepts/python-usages-examples
    """

    ruleset_params = {"max_rules": max_rules}
    interpreter = lw.RIPPER(**ruleset_params)

    # Fit a 'complex' model - CatBoostClassifier
    X_train = train.drop(['knocked_out_case'], axis=1)
    y_train = train['knocked_out_case']
    X_test = test.drop(['knocked_out_case'], axis=1)
    y_test = test['knocked_out_case']
    categorical_features_indexes = [X_train.columns.get_loc(col) for col in X_train.columns if
                                    X_train[col].dtype == "object"]
    test_pool = Pool(X_test, y_test, cat_features=categorical_features_indexes, feature_names=list(X_test.columns))

    model = CatBoostClassifier(iterations=10, depth=16, loss_function='Logloss', eval_metric="AUC")
    model.fit(X_train, y_train, cat_features=categorical_features_indexes,
              use_best_model=True, eval_set=test_pool)

    # Get Catboost model metrics
    catboost_auc_score = get_catboost_roc_curve_cv(model, X_test, y_test, categorical_features_indexes, activity,
                                                   output_dir)

    # interpret with wittgenstein and get a ruleset
    def predict_fn(data, _):
        res = model.predict(data, prediction_type='Class')
        res = [x == 'True' for x in res]
        return res

    interpret_model(model=model, X=X_train, interpreter=interpreter, model_predict_function=predict_fn)
    ruleset_model = interpreter

    catboost_feature_importances = list(
        model.get_feature_importance(prettified=True).itertuples(index=False, name=None))

    ruleset_params['catboost'] = model.get_params()
    ruleset_params['catboost_total_trees'] = model.tree_count_

    return ruleset_model, ruleset_params, model, catboost_feature_importances, catboost_auc_score


def do_grid_search(ruleset_model, dataset, activity, algorithm="RIPPER", quiet=True, param_grid=None,
                   skip_temporal_holdout=False):
    dataset = deepcopy(dataset)

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
        sgkf = StratifiedKFold(n_splits=5)
        grid = GridSearchCV(estimator=ruleset_model, param_grid=param_grid, scoring="balanced_accuracy", n_jobs=1,
                            cv=sgkf)
    else:
        # Temporal split is the correct way to handle time-series data
        # In this way, we can still use grid search + cross validation, with time-aware splits
        # Source: https://stackoverflow.com/a/46918197/8522453
        #         https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.TimeSeriesSplit.html
        tscv = TimeSeriesSplit(gap=0, max_train_size=None, test_size=None, n_splits=3)
        grid = GridSearchCV(estimator=ruleset_model, param_grid=param_grid, scoring="f1", n_jobs=1, cv=tscv)

    with shutup.mute_warnings:
        grid.fit(x_train, y_train)

    return grid.best_params_


def get_performance_metrics(test: pd.DataFrame, _by_case: pd.DataFrame, columns_to_ignore: list[str],
                            ruleset_model: lw.RIPPER, available_cases_before_ko: dict, activity: str,
                            _by_case_only_cases_in_test: pd.DataFrame, skip_temporal_holdout: bool, output_dir: str):
    # Performance metrics
    x_test = test.drop(['knocked_out_case'], axis=1)
    y_test = test['knocked_out_case']

    X = _by_case.drop(columns_to_ignore + ['knocked_out_case'], axis=1, errors='ignore')
    y = _by_case['knocked_out_case']

    support = calc_knockout_ruleset_support(activity, ruleset_model, _by_case,
                                            available_cases_before_ko=available_cases_before_ko[activity])

    confidence = calc_knockout_ruleset_confidence(activity, ruleset_model, _by_case)

    metrics = {'support': support,
               'confidence': confidence,
               'condition_count': ruleset_model.ruleset_.count_conds(),
               'rule_count': ruleset_model.ruleset_.count_rules(),
               'precision': ruleset_model.score(x_test, y_test, precision_score),
               'recall': ruleset_model.score(x_test, y_test, recall_score),
               'f1_score': ruleset_model.score(x_test, y_test, f1_score),
               'roc_auc_score': get_roc_curve_cv(activity, ruleset_model, X, y, cv=5,
                                                 skip_temporal_holdout=skip_temporal_holdout, output_dir=output_dir)}

    return metrics


def get_roc_curve_cv(activity, model: Union[lw.RIPPER, lw.IREP, CatBoostClassifier], X, y, cv, skip_temporal_holdout,
                     output_dir):
    # Run classifier with cross-validation, plot ROC curves and return average AUC score
    # Source: https://scikit-learn.org/stable/auto_examples/model_selection/plot_roc_crossval.html

    if not os.getenv("ENABLE_ROC_PLOTS", False):
        return 0

    if skip_temporal_holdout:
        kf = StratifiedKFold(n_splits=cv)
    else:
        kf = TimeSeriesSplit(gap=0, max_train_size=None, test_size=None, n_splits=cv)

    tprs = []
    aucs = []
    mean_fpr = np.linspace(0, 1, 100)

    fig, ax = plt.subplots()
    for i, (train, test) in enumerate(kf.split(X, y)):
        X_train, X_test = X.iloc[train], X.iloc[test]
        y_train, y_test = y.iloc[train], y.iloc[test]

        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        viz = RocCurveDisplay.from_predictions(
            y_test,
            y_pred,
            name="ROC fold {}".format(i),
            alpha=0.3,
            lw=1,
            ax=ax,
        )
        interp_tpr = np.interp(mean_fpr, viz.fpr, viz.tpr)
        interp_tpr[0] = 0.0
        tprs.append(interp_tpr)
        auc_value = viz.roc_auc
        if np.isnan(auc_value):
            auc_value = 0
        aucs.append(auc_value)

    ax.plot([0, 1], [0, 1], linestyle="--", lw=2, color="r", label="Random classifier", alpha=0.8)

    mean_tpr = np.nanmean(tprs, axis=0)
    mean_tpr[-1] = 1.0
    mean_auc = auc(mean_fpr, mean_tpr)

    std_auc = np.std(aucs)
    ax.plot(
        mean_fpr,
        mean_tpr,
        color="b",
        label=r"Mean ROC (AUC = %0.2f $\pm$ %0.2f)" % (mean_auc, std_auc),
        lw=2,
        alpha=0.8,
    )

    std_tpr = np.nanstd(tprs, axis=0)
    tprs_upper = np.minimum(mean_tpr + std_tpr, 1)
    tprs_lower = np.maximum(mean_tpr - std_tpr, 0)
    ax.fill_between(
        mean_fpr,
        tprs_lower,
        tprs_upper,
        color="grey",
        alpha=0.2,
        label=r"$\pm$ 1 std. dev.",
    )

    ax.set(
        xlim=[-0.05, 1.05],
        ylim=[-0.05, 1.05],
        title=f"ROC curve for '{activity}'",
    )
    ax.legend(loc="lower right")
    plt.savefig(f'{output_dir}/{activity.replace(" ", "_")}_{int(time.time())}.png')

    plt.show()

    return np.mean(mean_auc)


def get_catboost_roc_curve_cv(model, X, y, categorical_features_indexes, activity, output_dir):
    try:
        # auc = model.best_score_['validation']['AUC']
        auc = roc_auc_score(y, model.predict_proba(X)[:, 1])
    except:
        true_count = sum(y)
        raise Exception(f"{true_count} positive example(s) for \'{activity}\' in test set")

    if not os.getenv("ENABLE_ROC_PLOTS", False):
        return auc

    (fpr, tpr, _) = get_roc_curve(model,
                                  Pool(X, y, cat_features=categorical_features_indexes, feature_names=list(X.columns)))

    ax = plt.gca()
    ax.plot(fpr, tpr, color="b",
            lw=2,
            alpha=0.8,
            label="ROC (AUC = %0.2f)" % auc)
    ax.plot([0, 1], [0, 1], linestyle="--", lw=2, color="r", label="Random classifier", alpha=0.8)

    ax.set(
        xlim=[-0.05, 1.05],
        ylim=[-0.05, 1.05],
        title=f"Catboost ROC curve for '{activity}'",
    )
    ax.legend(loc="lower right")

    plt.savefig(f'{output_dir}/{activity.replace(" ", "_")}_CB_{int(time.time())}.png')
    plt.show()

    return auc
