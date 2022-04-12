import sys

import pytest

sys.path.append('..')

from copy import deepcopy
from pathlib import Path

import pandas as pd
import numpy as np

from .preprocessing import pt_cleaning as pt_cleaning

from .preprocessing.external.feature_extraction import intercase_and_context
from .preprocessing.external.predictive_monitoring_benchmark.experiments import BucketFactory

from .preprocessing.external import config_data_from_file, Configuration, ReadOptions, LogReader

from .preprocessing.grouping.clustering_pipelines import clust_kproto
from .preprocessing.grouping.explainer import explainer_sklearn_dt

from .preprocessing.activity_feature_extraction.activity_transformer import ActivityTransformer


# TODO: These tests are slow. Skip depending on slow/no-slow flag like Simod.

@pytest.mark.skip()
def test_intercase_case_level():
    # # Inter-case variability analysis

    # ##### Parse Log

    config_file = "./config/config_cons.yml"
    GROUPED_OUTPUT_DF_NAME = './outputs/consulta_intercase_and_kproto_by_case'
    UNGROUPED_OUTPUT_DF_NAME = './outputs/consulta_intercase_and_kproto'

    pd.set_option('display.float_format', lambda x: '%.2f' % x)

    config_path = Path(config_file)
    config_data = config_data_from_file(config_path)
    config = Configuration(**config_data)
    options = ReadOptions(column_names=ReadOptions.column_names_default())

    log = LogReader(config.log_path, options)

    # ##### Add inter-arrival features & clean processing times

    try:
        # to skip parsing on notebook resets
        log_df = pd.read_pickle("./outputs/" + config_file.split("./config/")[1] + "_case_grouping_dump.pkl")

    except:
        enriched_log_df, res_analyzer = intercase_and_context.extract(log, _model_type='dual_inter')
        log_df = pt_cleaning.clean_processing_times_with_calendar(enriched_log_df, config, res_analyzer)

        log_df.to_pickle("./outputs/" + config_file.split("./config/")[1] + "_case_grouping_dump.pkl")

    log_df.head()

    # ### Discover groups of cases

    df = deepcopy(log_df).drop(columns=['@@duration'])

    # Clean up: exclude cols, drop NAN
    excluded = ['@@duration', 'Activity', 'Resource', 'Role', '@@startevent_concept:name',
                '@@startevent_org:resource', '@@startevent_Activity',
                '@@startevent_Resource', 'n_role', 'role', 'n_task']  # , 'elementId', 'timetable_id', 'resourceId']

    df = df.drop(columns=excluded, errors='ignore').dropna()
    df = df[df.columns.drop(list(df.filter(regex='@@startevent_')))]

    # Categorical pre-proc
    cat_cols = ['user', 'st_weekday', 'end_weekday', 'st_month', 'end_month', "started_offtimetable",
                "ended_offtimetable"]
    df['user'] = df['user'].apply(str)

    # cat_cols = list(filter(lambda c : c != 'task', cat_cols))

    # Numerical pre-proc
    num_cols = list(df.select_dtypes(['number']).columns)

    # Aggregation-encoding approach - Irene Teinemaa's PPM Benchmarking
    num_aggregators = ['mean', 'max', 'min', 'sum', 'std']
    duration_aggregators = ['sum']

    case_aggregators = {
        # '@@duration': duration_aggregators,
        'waiting_time': duration_aggregators,
        'processing_time': duration_aggregators,
        'user': lambda x: x.iloc[0],
        'st_weekday': lambda x: x.iloc[0],
        'end_weekday': lambda x: x.iloc[-1],
        'st_month': lambda x: x.iloc[0],
        'end_month': lambda x: x.iloc[-1],
        "started_offtimetable": lambda x: x.iloc[0],  # first item of the trace
        "ended_offtimetable": lambda x: x.iloc[-1],  # last item of the trace
        "st_daytime": lambda x: x.iloc[0],
        "end_daytime": lambda x: x.iloc[-1]
    }

    for c in num_cols:
        if (c not in excluded) and (c not in case_aggregators.keys()):
            case_aggregators[c] = num_aggregators

    by_case = df.sort_values(['start_timestamp'], ascending=True).groupby('caseid').agg(case_aggregators)

    by_case = by_case.drop(columns=['end_timestamp', 'start_timestamp'], errors='ignore')

    by_case.columns = [a[0] if (a[1] == "<lambda>") else "_".join(a) for a in by_case.columns.to_flat_index()]

    # update with aggregated cols
    # num_cols = list(by_case.select_dtypes(['number']).columns)
    cat_cols = list(set(cat_cols + list(by_case.select_dtypes(['object']).columns)))

    categorical_indexes = [by_case.columns.get_loc(f"{c}") for c in cat_cols]
    num_cols = by_case.columns[list(set(list(range(0, len(by_case.columns)))) - set(categorical_indexes))]

    by_case = by_case.dropna()

    print(by_case.info())
    by_case.head()

    clusters, centers = clust_kproto(by_case, categorical_indexes, num_cols,
                                     K_MODES_ELBOW=0)  # elbow value 0 to force best k re-computation

    by_case['k_proto_cluster'] = clusters
    by_case.to_pickle(GROUPED_OUTPUT_DF_NAME)

    log_df_with_case_cluster = pd.merge(log_df, by_case[['k_proto_cluster']], on="caseid")
    log_df_with_case_cluster.to_pickle(UNGROUPED_OUTPUT_DF_NAME)

    # #### Methods from Irene's PPM Benchmarking: K-Means clustering & Prefix-Length Buckets

    df = pd.read_pickle(UNGROUPED_OUTPUT_DF_NAME)
    by_case = pd.read_pickle(GROUPED_OUTPUT_DF_NAME)

    cat_cols = ['user', 'st_weekday', "end_weekday", 'st_month', "end_month", "started_offtimetable",
                "ended_offtimetable"]
    num_cols = list(df.select_dtypes(['number']).columns)

    # irene clusters
    cluster_agg = BucketFactory.get_bucketer('cluster', n_clusters=5, encoding_method="agg", case_id_col="caseid",
                                             num_cols=num_cols, cat_cols=cat_cols)
    clusters = cluster_agg.fit_predict(df)
    by_case['k_means_cluster'] = clusters

    # irene prefix-length buckets
    bucketer = BucketFactory.get_bucketer(
        "prefix", case_id_col="caseid", num_cols=num_cols, cat_cols=cat_cols
    )
    buckets = bucketer.fit_predict(df)
    by_case["prefix_length_bucket"] = buckets

    # ### Explain differences between  groups

    # #### By K-Prototypes Cluster

    # One-hot encoding necessary because sk-learn dec. tree classifier does not handle categorical vars.
    for c in cat_cols:
        by_case[c] = by_case[c].apply(str)
    one_hot_data = pd.get_dummies(by_case[cat_cols], drop_first=True)
    enc_by_case_df = pd.merge(by_case.drop(columns=cat_cols), one_hot_data, left_index=True, right_index=True)

    clf, rule_printer, path_visualizer, top_feature_getter, duration_stats_getter = explainer_sklearn_dt(enc_by_case_df,
                                                                                                         "k_proto_cluster",
                                                                                                         max_depth=3,
                                                                                                         min_samples_leaf=1,
                                                                                                         min_samples_split=0.10,
                                                                                                         exclude=[
                                                                                                             'k_means_cluster',
                                                                                                             'prefix_length_bucket',
                                                                                                             'processing_time_sum',
                                                                                                             'waiting_time_sum'])

    rule_printer()

    stats = duration_stats_getter(_figsize=(10, 5))
    stats.describe()

    # #### By K-Means Cluster

    _, rule_printer_km, _, _, duration_stats_getter_km = explainer_sklearn_dt(enc_by_case_df, "k_means_cluster",
                                                                              max_depth=5, min_samples_leaf=1,
                                                                              min_samples_split=0.2,
                                                                              exclude=['k_proto_cluster',
                                                                                       'prefix_length_bucket'])

    rule_printer_km()

    stats_km = duration_stats_getter_km(_figsize=(12, 5))
    stats_km.describe()

    assert True  # Simply testing everything runs


@pytest.mark.skip()
def test_intracase_case_level():
    # # Intra-case type variability analysis
    INPUT_DF_NAME = './outputs/consulta_intercase_and_kproto_by_case'
    OUTPUT_DF_NAME = './outputs/consulta_intracase_case_level'

    pd.set_option('display.float_format', lambda x: '%.2f' % x)

    log_df = pd.read_pickle(INPUT_DF_NAME)

    # transformer = ActivityTransformer(log_df)
    # log_df = transformer.transform()

    df = deepcopy(log_df)

    # Clean up: exclude cols, drop NAN
    excluded = ['user', 'task', '@@duration', 'end_timestamp', 'start_timestamp', 'Activity', 'Resource', 'Role',
                '@@startevent_concept:name',
                '@@startevent_org:resource', '@@startevent_Activity',
                '@@startevent_Resource', 'n_role', 'role', 'n_task']

    df = df.drop(columns=["caseid"] + excluded, errors='ignore').dropna()
    df = df[df.columns.drop(list(df.filter(regex='@@startevent_')))]

    # Categorical pre-proc
    cat_cols = ['st_weekday', 'end_weekday', 'st_month', 'end_month', "started_offtimetable", "ended_offtimetable"]

    cat_cols = list(set(cat_cols + list(df.select_dtypes(['object']).columns)))
    cat_cols = list(filter(lambda c: c != 'task', cat_cols))

    for c in cat_cols:
        df[c] = df[c].apply(str)

    # Numerical pre-proc
    num_cols = list(df.select_dtypes(['number']).columns)

    by_case_type = df.groupby('k_proto_cluster')

    # TO-DO extend to all cases (or a subset of most relevant)

    # activity_name = activities_srt_by_duration_std.head(1).index[0]
    case_type = 2

    case_df = by_case_type.get_group(case_type).drop(columns=["k_proto_cluster"])

    case_df['processing_time_quartile'] = pd.cut(case_df['processing_time_sum'], bins=4,
                                                 labels=list(range(0, 4)))

    # cat_cols.append('duration_quartile')

    categorical_indexes = [case_df.columns.get_loc(f"{c}") for c in cat_cols]
    num_cols = case_df.columns[list(set(list(range(0, len(case_df.columns)))) - set(categorical_indexes))]

    clusters, centers = clust_kproto(case_df, categorical_indexes, num_cols,
                                     K_MODES_ELBOW=5)  # elbow value 0 to force best k re-computation

    case_df['intra_case_cluster'] = clusters
    case_df.to_pickle(OUTPUT_DF_NAME)

    # #### Explainability: Intra-case clusters

    # One-hot encoding necessary because sk-learn dec. tree classifier does not handle categorical vars.
    for c in cat_cols:
        case_df[c] = case_df[c].apply(str)
    one_hot_data = pd.get_dummies(case_df[cat_cols], drop_first=True)
    enc_case_df = pd.merge(case_df.drop(columns=cat_cols), one_hot_data, left_index=True, right_index=True)

    clf, rule_printer, path_visualizer, top_feature_getter, duration_stats_getter = explainer_sklearn_dt(enc_case_df,
                                                                                                         "intra_case_cluster",
                                                                                                         exclude=[
                                                                                                             "processing_time_quartile",
                                                                                                             "processing_time_sum",
                                                                                                             "waiting_time_sum"],
                                                                                                         max_depth=3,
                                                                                                         min_samples_leaf=1,
                                                                                                         min_samples_split=0.1,
                                                                                                         criterion="entropy")

    rule_printer()

    stats = duration_stats_getter(_figsize=(10, 5))
    stats.describe()

    # ### Explainability: Processing Time quartiles

    # One-hot encoding necessary because sk-learn dec. tree classifier does not handle categorical vars.
    for c in cat_cols:
        case_df[c] = case_df[c].apply(str)
    one_hot_data = pd.get_dummies(case_df[cat_cols], drop_first=True)
    enc_case_df = pd.merge(case_df.drop(columns=cat_cols), one_hot_data, left_index=True, right_index=True)

    enc_case_df = enc_case_df.groupby('processing_time_quartile')
    enc_case_df = pd.DataFrame(enc_case_df.apply(lambda x: x.sample(enc_case_df.size().min()).reset_index(drop=True)))
    enc_case_df = enc_case_df.reset_index(drop=True)

    clf, rule_printer, path_visualizer, top_feature_getter, duration_stats_getter = explainer_sklearn_dt(enc_case_df,
                                                                                                         "processing_time_quartile",
                                                                                                         exclude=[
                                                                                                             "intra_case_cluster",
                                                                                                             "processing_time_sum",
                                                                                                             "waiting_time_sum"],
                                                                                                         max_depth=4,
                                                                                                         min_samples_leaf=1,
                                                                                                         min_samples_split=0.1,
                                                                                                         criterion="entropy")

    rule_printer()

    stats = duration_stats_getter(_figsize=(10, 5))
    stats.describe()

    assert True  # Simply testing everything runs


@pytest.mark.skip()
def test_intercase_activity_level():
    # # Activity level variability analysis (w/o case type grouping)

    # #### Discover activity clusters (k-prototypes)

    INPUT_DF_NAME = './outputs/consulta_intercase_and_kproto'
    OUTPUT_DF_NAME = './outputs/consulta_activity_no_case_group'

    pd.set_option('display.float_format', lambda x: '%.2f' % x)

    log_df = pd.read_pickle(INPUT_DF_NAME).drop(columns=["k_proto_cluster"])

    transformer = ActivityTransformer(log_df)
    log_df = transformer.transform()

    df = deepcopy(log_df)

    # Clean up: exclude cols, drop NAN
    excluded = ['@@duration', 'end_timestamp', 'start_timestamp', 'Activity', 'Resource', 'Role',
                '@@startevent_concept:name',
                '@@startevent_org:resource', '@@startevent_Activity',
                '@@startevent_Resource', 'n_role', 'role', 'n_task']

    df = df.drop(columns=["caseid"] + excluded, errors='ignore').dropna()
    df = df[df.columns.drop(list(df.filter(regex='@@startevent_')))]

    # Categorical pre-proc
    cat_cols = ['user', 'timetable_id', 'st_weekday', 'end_weekday', 'st_month', 'end_month', "started_offtimetable",
                "ended_offtimetable"]

    cat_cols = list(set(cat_cols + list(df.select_dtypes(['object']).columns)))
    cat_cols = list(filter(lambda c: c != 'task', cat_cols))

    for c in cat_cols:
        df[c] = df[c].apply(str)

    # Numerical pre-proc
    num_cols = list(df.select_dtypes(['number']).columns)

    by_activity = df.groupby('task')

    activities_srt_by_duration_std = by_activity \
        .agg({'processing_time': np.std, 'task': np.size}) \
        .rename(columns={'task': 'count', 'processing_time': 'processing_time_std'}) \
        .sort_values(by="processing_time_std", ascending=False)

    # TO-DO extend to all activities (or a subset of most relevant)

    # activity_name = activities_srt_by_duration_std.head(1).index[0]
    activity_name = "Homologacion por grupo de cursos"

    activity_df = by_activity.get_group(activity_name).drop(columns=["task"])

    activity_df['processing_time_quartile'] = pd.cut(activity_df['processing_time'], bins=4,
                                                     labels=list(range(0, 4)))

    # cat_cols.append('processing_time_quartile')

    categorical_indexes = [activity_df.columns.get_loc(f"{c}") for c in cat_cols]
    num_cols = activity_df.columns[list(set(list(range(0, len(activity_df.columns)))) - set(categorical_indexes))]

    clusters, centers = clust_kproto(activity_df, categorical_indexes, num_cols,
                                     K_MODES_ELBOW=5)  # elbow value 0 to force best k re-computation

    activity_df['activity_cluster'] = clusters
    activity_df.to_pickle(OUTPUT_DF_NAME)

    # #### Explainability: Activity Clusters

    # One-hot encoding necessary because sk-learn dec. tree classifier does not handle categorical vars.
    for c in cat_cols:
        activity_df[c] = activity_df[c].apply(str)
    one_hot_data = pd.get_dummies(activity_df[cat_cols], drop_first=True)
    enc_activity_df = pd.merge(activity_df.drop(columns=cat_cols), one_hot_data, left_index=True, right_index=True)

    clf, rule_printer, path_visualizer, top_feature_getter, duration_stats_getter = explainer_sklearn_dt(
        enc_activity_df, "activity_cluster",
        exclude=["processing_time_quartile", "processing_time", "waiting_time", "accumulated_activity_time"],
        max_depth=3, min_samples_leaf=1, min_samples_split=0.1, criterion="entropy", level='activity')

    rule_printer()

    stats = duration_stats_getter(_figsize=(10, 5))
    stats.describe()

    # ### Explainability: Processing Time quartiles

    # One-hot encoding necessary because sk-learn dec. tree classifier does not handle categorical vars.
    for c in cat_cols:
        activity_df[c] = activity_df[c].apply(str)
    one_hot_data = pd.get_dummies(activity_df[cat_cols], drop_first=True)
    enc_activity_df = pd.merge(activity_df.drop(columns=cat_cols), one_hot_data, left_index=True, right_index=True)

    enc_activity_df = enc_activity_df.groupby('processing_time_quartile')
    enc_activity_df = pd.DataFrame(
        enc_activity_df.apply(lambda x: x.sample(enc_activity_df.size().min()).reset_index(drop=True)))
    enc_activity_df = enc_activity_df.reset_index(drop=True)

    clf, rule_printer, path_visualizer, top_feature_getter, duration_stats_getter = explainer_sklearn_dt(
        enc_activity_df, "processing_time_quartile",
        exclude=["activity_cluster", "processing_time", "waiting_time", "accumulated_activity_time"], max_depth=4,
        min_samples_leaf=1, min_samples_split=0.1, criterion="entropy", level='activity')

    rule_printer()

    stats = duration_stats_getter(_figsize=(10, 5))
    stats.describe()

    assert True  # Simply testing everything runs
