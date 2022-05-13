from collections import Counter
from copy import deepcopy
from typing import List

import numpy as np
import pandas as pd
import pm4py
from scipy.stats import t

from pm4py.algo.filtering.pandas import ltl
from ruleset.base import Ruleset
from tqdm import tqdm

from knockout_ios.knockout_analyzer import KnockoutAnalyzer
from knockout_ios.utils.constants import *


def chained_eventually_follows(log, activities):
    if len(activities) < 2:
        return log

    if len(activities) == 2:
        return ltl.ltl_checker.A_eventually_B(log, activities[0], activities[1])

    for i in range(0, len(activities) - 1):
        log = ltl.ltl_checker.A_eventually_B(log, activities[i], activities[i + 1])

    return log


def get_observed_ko_checks_order(log, ko_activities):
    observed_ko_order = {ko: [] for ko in ko_activities}

    # Idea: for every ko activity, get a list of the index of appearance in every case of the log

    cases = log.groupby(PM4PY_CASE_ID_COLUMN_NAME)
    for ko_activity in observed_ko_order.keys():

        for group in cases.groups.keys():
            case = cases.get_group(group)
            case_ko = case['knockout_activity']

            if not (len(case_ko.values) > 0):
                continue

            if case_ko.values[0] != ko_activity:
                continue

            # Necessary to get the index of the activity in the case (starting from 0 at the beggining of THIS case)
            case.set_index(pd.Index([i for i in range(len(case[PM4PY_ACTIVITY_COLUMN_NAME]))]),
                           PM4PY_ACTIVITY_COLUMN_NAME,
                           inplace=True)

            idx = case[case[PM4PY_ACTIVITY_COLUMN_NAME] == ko_activity].index.item()
            observed_ko_order[ko_activity].append(idx)

        observed_ko_order[ko_activity] = int(np.mean(observed_ko_order[ko_activity]))

    # transform observed_ko_order into a list of its keys, sorted by the value of every key
    observed_ko_order = list(map(lambda x: x[0], sorted(observed_ko_order.items(), key=lambda x: x[1])))

    return observed_ko_order


def get_attribute_names_from_ruleset(ruleset: Ruleset):
    res = set()
    for rule in ruleset.ruleset_:
        for cond in rule.conds:
            res.add(cond.feature.replace("_", " "))

    return list(res)


def get_sorted_with_dependencies(ko_activities: List[str], dependencies: dict[str, List[tuple[str, str]]],
                                 current_activity_order: List[str],
                                 efforts=None):
    optimal_order_names = current_activity_order.copy()

    if (not (efforts is None)) and (REPORT_COLUMN_KNOCKOUT_CHECK in efforts.columns):
        efforts.set_index(REPORT_COLUMN_KNOCKOUT_CHECK, inplace=True)

    for knockout_activity in ko_activities:
        if knockout_activity not in optimal_order_names:
            continue

        _dependencies = dependencies[knockout_activity]
        if not all(map(lambda x: x[1] in optimal_order_names, _dependencies)):
            continue

        while len(_dependencies) > 0:
            # Sort deps by the index of every second element of the tuples in current_activity_order
            _dependencies = sorted(_dependencies, key=lambda x: optimal_order_names.index(x[1]))
            _, attribute_producer = _dependencies.pop(0)

            # Remove knockout_activity from current_activity_order to insert it in the right place
            optimal_order_names.remove(knockout_activity)

            # Find where is attribute_value_producer in current_activity_order,
            # then insert knockout_activity after attribute_value_producer
            idx = optimal_order_names.index(attribute_producer)
            optimal_order_names.insert(idx + 1, knockout_activity)

            # Sort by effort everything after the producer
            # def position_rejection(x):
            #     try:
            #         return efforts.loc[x].values[1]
            #     except KeyError:
            #         return len(optimal_order_names) - 1

            def position_effort(x):
                try:
                    return efforts.loc[x].values[0], (100 - efforts.loc[x].values[1])
                except KeyError:
                    return len(optimal_order_names) - 1, len(optimal_order_names) - 1

            if not (efforts is None):
                # Sort the rest of the list by effort (ascending)
                # TODO: by rejection rate (descending)?
                idx = optimal_order_names.index(attribute_producer)
                optimal_order_names[idx + 1:-1] = sorted(optimal_order_names[idx + 1:-1],
                                                         key=position_effort)

    return optimal_order_names


def find_producers(attribute: str, log: pd.DataFrame):
    """ Assumes log has case id as index and is sorted by end timestamp"""

    producers = []
    log = log.sort_values(by=['case_id_idx', SIMOD_END_TIMESTAMP_COLUMN_NAME],
                          ascending=[True, True], inplace=False)

    for caseid in log.index.unique():
        case = log.loc[caseid]

        activities_where_unavailable = case[pd.isnull(case[attribute].values)][
            SIMOD_LOG_READER_ACTIVITY_COLUMN_NAME].values

        # if attribute is never null, it means it's available from the start
        if not (len(activities_where_unavailable) > 0):
            continue

        try:
            log.loc[caseid, "next_attribute_value"] = (case[attribute]).shift(-1)
            # refresh view after adding column to log
            case = log.loc[caseid]
            # added ffill to handle the case where attribute is null in any subsequent events
            case = case.fillna(method="ffill")

            # find after which row, the attribute value stopped changing
            last_valid_value = case[attribute].values[-1]
            producer_event = case[case["next_attribute_value"] == last_valid_value].values[0]

            # extract the activity name
            idx = case.columns.get_loc(SIMOD_LOG_READER_ACTIVITY_COLUMN_NAME)
            producer_activity = producer_event[idx]

            producers.append(producer_activity)

        except Exception as e:
            print(e)
            continue

    return producers


def bootstrap_ci(
        data,
        statfunction=np.average,
        alpha=0.05,
        n_samples=100):
    # source: https://stackoverflow.com/a/66008548/8522453
    import warnings

    def bootstrap_ids(data, n_samples=100):
        for _ in range(n_samples):
            yield np.random.randint(data.shape[0], size=(data.shape[0],))

    alphas = np.array([alpha / 2, 1 - alpha / 2])
    nvals = np.round((n_samples - 1) * alphas).astype(int)
    if np.any(nvals < 10) or np.any(nvals >= n_samples - 10):
        warnings.warn("Some values used extremal samples; results are probably unstable. "
                      "Try to increase n_samples")

    data = np.array(data)
    if np.prod(data.shape) != max(data.shape):
        raise ValueError("Data must be 1D")
    data = data.ravel()

    boot_indexes = bootstrap_ids(data, n_samples)
    stat = np.asarray([statfunction(data[_ids]) for _ids in boot_indexes])
    stat.sort(axis=0)

    return stat[nvals]


def confidence_intervals_t_student(x, confidence=0.95):
    # source: https://towardsdatascience.com/how-to-calculate-confidence-intervals-in-python-a8625a48e62b

    m = x.mean()
    s = x.std()
    dof = len(x) - 1

    t_crit = np.abs(t.ppf((1 - confidence) / 2, dof))

    return m - s * t_crit / np.sqrt(len(x)), m + s * t_crit / np.sqrt(len(x))


def get_relocated_kos(current_order_all_activities, ko_activities, dependencies, start_activity_constraint=None,
                      optimal_ko_order_constraint=None, efforts=None):
    trace_start_slice = []
    if not (start_activity_constraint is None):
        try:
            # get a slice of the list right before the start activity, so that re-location does not interfere with the process semantics
            trace_start_slice = current_order_all_activities[
                                0:current_order_all_activities.index(start_activity_constraint) + 1]
            current_order_all_activities = current_order_all_activities[
                                           current_order_all_activities.index(start_activity_constraint) + 1:]
        except Exception:
            return current_order_all_activities

    if not (optimal_ko_order_constraint is None):
        # reorder current_order_all_activities to match the relative orders in optimal_ko_order_constraint (potentially shorter list, as it only contains ko activities)
        def position(value):
            # source: https://stackoverflow.com/a/52545309/8522453
            try:
                return optimal_ko_order_constraint.index(value)
            except ValueError:
                return len(optimal_ko_order_constraint)

        current_order_all_activities.sort(key=position)

    relocated = get_sorted_with_dependencies(ko_activities=ko_activities, dependencies=dependencies,
                                             current_activity_order=current_order_all_activities, efforts=efforts)

    trace_start_slice.extend(relocated)
    return trace_start_slice


def find_ko_activity_dependencies(analyzer: KnockoutAnalyzer) -> dict[str, List[tuple[str, str]]]:
    """
    - Returns dependencies between log ko_activities and attributes required by knockout checks
    """

    if analyzer.ruleset_algorithm == "IREP":
        rule_discovery_dict = analyzer.IREP_rulesets
    elif analyzer.ruleset_algorithm == "RIPPER":
        rule_discovery_dict = analyzer.RIPPER_rulesets
    else:
        raise ValueError("Unknown ruleset algorithm")

    # for every knockout activity, there will be a list of tuples (attribute of KO rule, name of producer activity)
    dependencies = {k: [] for k in rule_discovery_dict.keys()}

    log = deepcopy(analyzer.discoverer.log_df)
    log.set_index(SIMOD_LOG_READER_CASE_ID_COLUMN_NAME, inplace=True)
    log = log.rename_axis('case_id_idx')

    for ko_activity in tqdm(rule_discovery_dict.keys(), desc="Searching KO activity dependencies"):
        ruleset = rule_discovery_dict[ko_activity][0]

        if len(ruleset.ruleset_) == 0:
            continue

        required_attributes = get_attribute_names_from_ruleset(ruleset)

        for attribute in required_attributes:
            # Find after which activity the attribute is available in the log
            # (a list is returned; we then consider the most frequent activity as the producer)
            producers = find_producers(attribute, log[log["knockout_activity"] == ko_activity])

            if len(producers) > 0:
                # get most frequent producer activity
                producers = Counter(producers).most_common(1)[0][0]
                if producers == ko_activity:
                    continue
                dependencies[ko_activity].append((attribute, producers))

    return dependencies


def evaluate_knockout_relocation_io(analyzer: KnockoutAnalyzer, dependencies: dict[str, List[tuple[str, str]]],
                                    optimal_ko_order=None, efforts: pd.DataFrame = None) -> dict[tuple[str], List[str]]:
    # TODO: get min_coverage_percentage or K as a parameter in config file
    log = deepcopy(analyzer.discoverer.log_df)
    log.sort_values(
        by=[SIMOD_LOG_READER_CASE_ID_COLUMN_NAME, SIMOD_END_TIMESTAMP_COLUMN_NAME],
        inplace=True)

    # flt = pm4py.filter_variants_by_coverage_percentage(analyzer.discoverer.log_df, min_coverage_percentage=0.01)
    flt = pm4py.filter_variants_top_k(analyzer.discoverer.log_df, k=10)
    variants = pm4py.get_variants_as_tuples(flt)

    proposed_relocations = {}
    for variant in variants.keys():
        proposed_relocations[variant] = get_relocated_kos(current_order_all_activities=list(variant),
                                                          ko_activities=analyzer.discoverer.ko_activities,
                                                          dependencies=dependencies,
                                                          start_activity_constraint=analyzer.start_activity,
                                                          optimal_ko_order_constraint=optimal_ko_order,
                                                          efforts=efforts)

    return proposed_relocations


def evaluate_knockout_reordering_io(analyzer: KnockoutAnalyzer,
                                    dependencies: dict[str, List[tuple[str, str]]] = None) -> tuple[dict, pd.DataFrame]:
    '''
    - Returns the observed ko-checks ordering (AS-IS)
    - Returns optimal ordering by KO effort
    - If a dependencies dictionary is provided, it will take it into account for the optimal ordering
    '''

    log = deepcopy(analyzer.discoverer.log_df)
    log.sort_values(
        by=[SIMOD_LOG_READER_CASE_ID_COLUMN_NAME, SIMOD_END_TIMESTAMP_COLUMN_NAME],
        inplace=True)

    report_df = deepcopy(analyzer.report_df)
    report_df[REPORT_COLUMN_REJECTION_RATE] = report_df[REPORT_COLUMN_REJECTION_RATE].str.replace('%', '')
    report_df[REPORT_COLUMN_REJECTION_RATE] = report_df[REPORT_COLUMN_REJECTION_RATE].astype(float)

    sorted_by_effort = report_df.sort_values(by=[REPORT_COLUMN_EFFORT_PER_KO, REPORT_COLUMN_REJECTION_RATE],
                                             ascending=[True, False], inplace=False)

    optimal_order_names = sorted_by_effort[REPORT_COLUMN_KNOCKOUT_CHECK].values
    efforts = sorted_by_effort[
        [REPORT_COLUMN_KNOCKOUT_CHECK, REPORT_COLUMN_EFFORT_PER_KO, REPORT_COLUMN_REJECTION_RATE]]

    # Determine how many cases respect this order in the log
    # TODO: for the moment, keeping only non knocked out cases to analyze order. Could be useful to see also partial (ko-d) cases
    filtered = log[log['knocked_out_case'] == False]
    total_cases = filtered.groupby([PM4PY_CASE_ID_COLUMN_NAME]).ngroups

    if not (dependencies is None):
        # TODO: make it more flexible / generic, to include other sorting criteria
        optimal_order_names = get_sorted_with_dependencies(
            ko_activities=list(optimal_order_names), dependencies=dependencies,
            current_activity_order=list(optimal_order_names), efforts=efforts)

    cases_respecting_order = chained_eventually_follows(filtered, optimal_order_names) \
        .groupby([PM4PY_CASE_ID_COLUMN_NAME])

    efforts.reset_index(inplace=True)
    return {"optimal_ko_order": list(optimal_order_names),
            "cases_respecting_order": cases_respecting_order.ngroups,
            "total_cases": total_cases}, efforts


def evaluate_knockout_rule_change_io(analyzer: KnockoutAnalyzer, confidence=0.95):
    """
    # TODO: consider this, for parsing the rules https://stackoverflow.com/a/6405461/8522453
    """
    if analyzer.ruleset_algorithm == "IREP":
        rule_discovery_dict = analyzer.IREP_rulesets
    elif analyzer.ruleset_algorithm == "RIPPER":
        rule_discovery_dict = analyzer.RIPPER_rulesets
    else:
        raise ValueError("Unknown ruleset algorithm")

    adjusted_values = {k: [] for k in rule_discovery_dict.keys()}
    raw_rulesets = {k: [] for k in rule_discovery_dict.keys()}
    log = analyzer.rule_discovery_log_df

    for ko_activity in tqdm(rule_discovery_dict.keys(), desc="Analyzing KO rule value ranges"):
        log_subset = log[log['knockout_activity'] == ko_activity]

        ruleset = rule_discovery_dict[ko_activity][0]
        required_attributes = get_attribute_names_from_ruleset(ruleset)

        adjusted_values[ko_activity] = {}
        raw_rulesets[ko_activity] = ruleset.ruleset_
        for attribute in required_attributes:
            column = log_subset[attribute]
            if column.dtype.kind not in ['i', 'f']:
                continue
            column = column.dropna()
            adjusted_values[ko_activity][attribute] = (np.min(column), np.max(column))

    return adjusted_values, raw_rulesets
