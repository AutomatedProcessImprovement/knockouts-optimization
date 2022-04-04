import pandas as pd
import numpy as np

import math

from copy import deepcopy

from matplotlib import pyplot as plt

from sklearn.tree import _tree, DecisionTreeClassifier
from sklearn.model_selection import train_test_split

from catboost import CatBoostClassifier, Pool

import shap

from dtreeviz.trees import *

from . import utils


def explainer_catboost(
        by_case,
        target,
        n_top_features=5,
        params=None,
        exclude=[],
        plot_imp=False,
        plot_tree=False,
        plot_shap=False,
        feature_boxplots=True,
):
    """
    ### Rule Extraction: CatBoost Decision Tree

    - CatBoost handles categorical variables without one-hot encoding, nor scaling (Source?)

    - [Usage](https://catboost.ai/en/docs/concepts/python-quickstart#classification-and-regression) / [Tutorial](https://github.com/catboost/tutorials/blob/master/python_tutorial.ipynb)
    """

    _by_case = deepcopy(by_case)

    # keep only labels appearing more than once
    v = _by_case[target].value_counts()
    _by_case = _by_case[_by_case[target].isin(v.index[v.gt(1)])]

    X = _by_case.drop(columns=[target] + exclude, axis=1)
    y = _by_case[target]

    is_cat = X.dtypes != np.number
    for feature, feat_is_cat in is_cat.to_dict().items():
        if feat_is_cat:
            X[feature].fillna("NAN", inplace=True)

    cat_features_index = np.where(is_cat)[0]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, stratify=y)

    if params == None:
        params = {
            "iterations": 5000,
            "learning_rate": 0.01,
            "cat_features": cat_features_index,
            "depth": 3,
            "eval_metric": "AUC",
            "verbose": False,
            "od_type": "Iter",  # overfit detector
            "od_wait": 500,  # most recent best iteration to wait before stopping
        }
    else:
        params["cat_features"] = cat_features_index

    model_clf = CatBoostClassifier(**params)

    # model = CatBoostClassifier(
    #    max_depth=5, verbose=True, max_ctr_complexity=1, iterations=10,).fit(pool)

    model_clf.fit(
        X_train,
        y_train,
        eval_set=(X_test, y_test),
        use_best_model=True,
        # True if we don't want to save trees created after iteration with the best validation score
        plot=False,
    )

    imp_df = pd.DataFrame(
        {
            "importance": model_clf.feature_importances_,
            "feature": X.columns[range(0, len(model_clf.feature_importances_))],
        }
    )
    imp_df = imp_df.sort_values("importance", ascending=False)
    imp_df = imp_df.set_index("feature")

    if plot_imp:
        imp_df.head(n_top_features + 5).plot(kind="bar", figsize=(12, 6))

    if plot_tree:
        model_clf.plot_tree(tree_idx=0)

    """
    #### SHAP values

    explain the prediction of an instance x by computing the contribution of each feature to the prediction. 

    Feature values of a data instance act as players in a coalition. Shapley values tell us how to fairly distribute the "payout" (= the prediction) among the features. 

    [About](https://christophm.github.io/interpretable-ml-book/shap.html#fn69)
    """
    if plot_shap:
        shap_values = model_clf.get_feature_importance(
            Pool(
                X_test,
                y_test,
                cat_features=cat_features_index,
                feature_names=list(X.columns),
            ),
            type="ShapValues",
        )

        expected_value = shap_values[0, -1]
        shap_values = shap_values[:, :-1]
        shap_values_transposed = shap_values.transpose(1, 0, 2)

        shap.initjs()
        # shap.summary_plot(shap_values=list(shap_values_transposed[:, :, :-1]), features=X_test, class_names=y_train.unique(), plot_type='bar')

        explainer = shap.TreeExplainer(model_clf)
        shap_values = explainer.shap_values(Pool(X, y, cat_features=cat_features_index))

        for i, c in enumerate(shap_values):
            shap.summary_plot(shap_values[i], X, show=False)
            plt.title(f"Cluster #{i}")
            plt.show()

    ## box-plot of top features

    if feature_boxplots:
        get_boxplots(imp_df, _by_case, target, n_top_features=n_top_features)

    return model_clf, imp_df


def explainer_sklearn_dt(
        by_case,
        target,
        max_depth=3,
        min_samples_leaf=10,
        min_samples_split=2,
        criterion="entropy",
        n_top_features=5,
        exclude=[],
        plot_imp=False,
        level="case",
):
    _by_case = deepcopy(by_case)

    X = _by_case.drop(columns=[target] + exclude, axis=1)
    y = _by_case[target]

    feature_names = list(X.columns.values)
    class_names = list(y.apply(str).unique())

    # Fit the classifier with max_depth=3
    clf = DecisionTreeClassifier(
        max_depth=max_depth,
        min_samples_leaf=min_samples_leaf,
        min_samples_split=min_samples_split,
        criterion=criterion,
    )
    model = clf.fit(X, y)

    # Plot tree with graphviz
    # dot_data = tree.export_graphviz(ruleset_model, out_file=None, feature_names=feature_names,class_names=class_names, filled=True, rounded=True,special_characters=True)
    # graph = graphviz.Source(dot_data)
    # graph

    # feature importances
    imp_df = pd.DataFrame(
        {
            "importance": clf.feature_importances_,
            "feature": X.columns[range(0, len(clf.feature_importances_))],
        }
    )
    imp_df = imp_df.sort_values("importance", ascending=False)
    imp_df = imp_df.set_index("feature")

    if plot_imp:
        imp_df.head(n_top_features).plot(kind="bar", figsize=(12, 6))

    if len(class_names) > utils.DTREEVIZ_LIMIT:
        colors = utils.dtreeviz_extended_colors
    else:
        colors = None

    if len(class_names) > 5:
        histtype = "bar"
    else:
        histtype = "barstacked"

    # Plot tree with dtreeviz
    viz = dtreeviz(
        clf,
        X,
        y,
        target_name=target,
        feature_names=feature_names,
        class_names=class_names,  # need class_names for classifier,
        colors=colors,
        histtype=histtype,
    )
    viz.view()

    # Print human-readable rules
    # text_representation = tree.export_text(ruleset_model, feature_names=feature_names)
    # print(text_representation)
    rules = get_dt_rules(clf, feature_names, class_names)

    def rule_printer():
        print(f"\nGrouping by {target}\n\n{len(rules)} Decision Rules found:\n")
        for r in rules:
            print(r)

    def path_visualizer(sample):
        return get_prediction_path(
            sample, X, y, clf, feature_names, class_names, target, histtype=histtype
        )

    def feat_box_plotter(
            _normalize=True,
            _n_top_features=n_top_features,
            _layout=None,
            _figsize=None,
            _cols=3,
    ):
        return get_boxplots(
            imp_df,
            _by_case,
            target,
            n_top_features=_n_top_features,
            layout=_layout,
            figsize=_figsize,
            normalize=_normalize,
            cols=_cols,
        )

    def duration_stats_getter(_layout=None, _figsize=None):
        if level == "activity":
            return get_duration_stats(
                _by_case,
                target,
                layout=_layout,
                figsize=_figsize,
                duration_features=["processing_time", "waiting_time"],
            )
        else:
            return get_duration_stats(
                _by_case, target, layout=_layout, figsize=_figsize
            )

    def top_feature_getter(_n_top_features=n_top_features):
        return list(imp_df.head(_n_top_features).index.values)

    return clf, rule_printer, path_visualizer, top_feature_getter, duration_stats_getter


def get_duration_stats(
        _by_case,
        target,
        layout=None,
        figsize=None,
        duration_features=["processing_time_sum", "waiting_time_sum"],
):
    if figsize == None:
        figsize = (len(_by_case[target].unique()) * 2, 5)

    if layout == None:
        layout = (1, len(_by_case[target].unique()))

    for feature in duration_features:
        _by_case.groupby(target).boxplot(
            column=feature,
            rot=45,
            fontsize=12,
            figsize=figsize,
            layout=layout,
            subplots=True,
        )

    return _by_case.groupby(target)[duration_features]


def get_boxplots(
        imp_df,
        _by_case,
        target,
        n_top_features=5,
        layout=None,
        figsize=None,
        normalize=True,
        cols=3,
):
    top_features = list(imp_df.head(n_top_features).index.values)

    if normalize:
        if figsize == None:
            figsize = (20, n_top_features * 2)

        _by_case.apply(
            lambda x: (x - x.min()) / (x.max() - x.min())
            if x.name in top_features
            else x
        ).boxplot(
            column=top_features,
            by=target,
            figsize=figsize,
            layout=(math.ceil(n_top_features / cols), cols),
        )

    else:
        if figsize == None:
            figsize = (len(_by_case[target].unique()), 5)

        if layout == None:
            layout = (1, len(_by_case[target].unique()))

        for feature in top_features:
            _by_case.groupby(target).boxplot(
                column=feature,
                rot=45,
                fontsize=12,
                figsize=figsize,
                layout=layout,
                subplots=True,
            )


def get_dt_rules(tree, feature_names, class_names):
    # Source: https://mljar.com/blog/extract-rules-decision-tree/

    tree_ = tree.tree_
    feature_name = [
        feature_names[i] if i != _tree.TREE_UNDEFINED else "undefined!"
        for i in tree_.feature
    ]

    paths = []
    path = []

    def recurse(node, path, paths):

        if tree_.feature[node] != _tree.TREE_UNDEFINED:
            name = feature_name[node]
            threshold = tree_.threshold[node]
            p1, p2 = list(path), list(path)
            p1 += [f"({name} <= {np.round(threshold, 3)})"]
            recurse(tree_.children_left[node], p1, paths)
            p2 += [f"({name} > {np.round(threshold, 3)})"]
            recurse(tree_.children_right[node], p2, paths)
        else:
            path += [(tree_.value[node], tree_.n_node_samples[node])]
            paths += [path]

    recurse(0, path, paths)

    # sort by samples count
    samples_count = [p[-1][1] for p in paths]
    ii = list(np.argsort(samples_count))
    paths = [paths[i] for i in reversed(ii)]

    rules = []
    for path in paths:
        rule = "if "

        for p in path[:-1]:
            if rule != "if ":
                rule += " and "
            rule += str(p)
        rule += " then "
        if class_names is None:
            rule += "response: " + str(np.round(path[-1][0][0][0], 3))
        else:
            classes = path[-1][0][0]
            l = np.argmax(classes)
            rule += f"class = {class_names[l]}"
        rule += f" | Prob: {np.round(100.0 * classes[l] / np.sum(classes), 2)}%, based on {path[-1][1]:,} samples"
        rules += [rule]

    return rules


def get_prediction_path(sample, X, y, clf, feature_names, class_names, target, histtype="stacked"):
    viz = dtreeviz(
        clf,
        X,
        y,
        target_name=target,
        orientation="TD",  # top-down orientation
        feature_names=feature_names,
        class_names=class_names,
        X=sample,
        fancy=True,
        histtype=histtype,
    )

    viz.view()

    print(f"\nCase class: {sample[target]}\nReasons:")
    print(
        explain_prediction_path(
            clf, sample, feature_names=feature_names, explanation_type="plain_english"
        )
    )
