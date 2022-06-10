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
from knockout_ios.utils.constants import globalColumnNames


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

    cases = log.groupby(globalColumnNames.PM4PY_CASE_ID_COLUMN_NAME)
    for ko_activity in observed_ko_order.keys():

        for group in cases.groups.keys():
            case = cases.get_group(group)
            case_ko = case['knockout_activity']

            if not (len(case_ko.values) > 0):
                continue

            if case_ko.values[0] != ko_activity:
                continue

            # Necessary to get the index of the activity in the case (starting from 0 at the beggining of THIS case)
            case.set_index(pd.Index([i for i in range(len(case[globalColumnNames.PM4PY_ACTIVITY_COLUMN_NAME]))]),
                           globalColumnNames.PM4PY_ACTIVITY_COLUMN_NAME,
                           inplace=True)

            idx = case[case[globalColumnNames.PM4PY_ACTIVITY_COLUMN_NAME] == ko_activity].index.item()
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


def get_ko_activities_sorted_with_dependencies(dependencies: dict[str, List[tuple[str, str]]],
                                               current_activity_order: List[str],
                                               efforts: pd.DataFrame = None):
    """ Assumes ko_activities are already sorted by effort """

    # TODO: Works but needs refactor. Consider Topological Sort:
    # https://stackoverflow.com/questions/4106862/how-to-sort-depended-objects-by-dependency
    # https://www.geeksforgeeks.org/topological-sorting/
    # https://networkx.org/documentation/stable/reference/algorithms/generated/networkx.algorithms.dag.topological_sort.html

    optimal_order_names = deepcopy(current_activity_order)

    # First, sort by index of the dependency that appears last in the current "optimal ko order
    max_dependency_indexes = []
    for i, activity in enumerate(current_activity_order):
        filtered_deps = list(filter(lambda a: a[1] in current_activity_order, dependencies[activity]))
        dependency_indexes = [optimal_order_names.index(t[1]) for t in filtered_deps]

        if len(dependency_indexes) > 0:
            max_dependency_indexes.append((activity, max(dependency_indexes)))
        else:
            max_dependency_indexes.append((activity, 0))

    optimal_order_names = sorted(max_dependency_indexes, key=lambda x: x[1])
    optimal_order_names = [t[0] for t in optimal_order_names]

    # Then rearrange the list if needed, making sure no activity is before its dependency
    if (not (efforts is None)) and (globalColumnNames.REPORT_COLUMN_KNOCKOUT_CHECK in efforts.columns):
        efforts.set_index(globalColumnNames.REPORT_COLUMN_KNOCKOUT_CHECK, inplace=True)

    for i, activity in enumerate(current_activity_order):
        filtered_deps = list(filter(lambda a: a[1] in current_activity_order, dependencies[activity]))
        dependency_indexes = [optimal_order_names.index(t[1]) for t in filtered_deps]

        if len(dependency_indexes) > 0:
            last_dependency = optimal_order_names[max(dependency_indexes)]

            if (activity in optimal_order_names) and (last_dependency in optimal_order_names):
                optimal_order_names.remove(activity)
                optimal_order_names.insert(optimal_order_names.index(last_dependency) + 1, activity)

                # Sort the rest of the list by effort and 100 - rejection rate
                if not (efforts is None):
                    def position_effort(x):
                        return efforts.loc[x].values[0], (100 - efforts.loc[x].values[1])

                    idx = optimal_order_names.index(last_dependency)
                    optimal_order_names[idx + 1:] = sorted(optimal_order_names[idx + 1:],
                                                           key=position_effort)

    return optimal_order_names


def find_producers(attribute: str, log: pd.DataFrame):
    """ Assumes log has case id as index and is sorted by end timestamp"""

    producers = []
    log = log.sort_values(by=['case_id_idx', globalColumnNames.SIMOD_END_TIMESTAMP_COLUMN_NAME],
                          ascending=[True, True], inplace=False)

    for caseid in log.index.unique():
        case = log.loc[caseid]

        activities_where_unavailable = case[pd.isnull(case[attribute].values)][
            globalColumnNames.SIMOD_LOG_READER_ACTIVITY_COLUMN_NAME].values

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
            idx = case.columns.get_loc(globalColumnNames.SIMOD_LOG_READER_ACTIVITY_COLUMN_NAME)
            producer_activity = producer_event[idx]

            producers.append(producer_activity)

        except Exception:
            continue

    return producers


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
    log.set_index(globalColumnNames.SIMOD_LOG_READER_CASE_ID_COLUMN_NAME, inplace=True)
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


def get_relocated_kos(current_order_all_activities, optimal_ko_order, dependencies, start_activity_constraint=None):
    """
    NOTE: This assumes optimal_ko_order is already sorted taking into account attribute dependencies BETWEEN ko activities!
    """
    # drop all activities that are in the optimal order
    ko_activities_in_trace = [x for x in optimal_ko_order if x in current_order_all_activities]
    current_order_all_activities = [x for x in current_order_all_activities if x not in optimal_ko_order]
    previous_knockout_activity = None

    # insert ko activities after their dependencies, and respecting the optimal order
    for i, knockout_activity in enumerate(optimal_ko_order):
        if knockout_activity not in ko_activities_in_trace:
            continue

        try:
            dependency_indexes = [current_order_all_activities.index(t[1]) for t in dependencies[knockout_activity]]
        except ValueError:
            continue

        if len(dependency_indexes) > 0:
            idx = max(dependency_indexes)
        else:
            # To allow for potentially inserting at the very begginning (insertion index gets added 1 later)
            idx = -1

        if previous_knockout_activity in current_order_all_activities:
            idx = max(idx, current_order_all_activities.index(previous_knockout_activity))

        # To accomodate additional information about the start activity; an activity that MUST be performed before any Ko checks,
        # but may not be reflected in case attribute dependencies
        if not (start_activity_constraint is None):
            idx = max(idx, current_order_all_activities.index(start_activity_constraint))

        current_order_all_activities.insert(idx + 1, knockout_activity)

        previous_knockout_activity = knockout_activity

    return current_order_all_activities


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


def evaluate_knockout_relocation_io(analyzer: KnockoutAnalyzer, dependencies: dict[str, List[tuple[str, str]]],
                                    optimal_ko_order=None, start_activity_constraint=None) -> tuple[
    dict[tuple[str], List[str]], dict]:
    log = deepcopy(analyzer.discoverer.log_df)
    log.sort_values(
        by=[globalColumnNames.SIMOD_LOG_READER_CASE_ID_COLUMN_NAME,
            globalColumnNames.SIMOD_END_TIMESTAMP_COLUMN_NAME],
        inplace=True)

    flt = pm4py.filter_variants_by_coverage_percentage(log,
                                                       min_coverage_percentage=analyzer.config.relocation_variants_min_coverage_percentage)
    variants = pm4py.get_variants_as_tuples(flt)

    proposed_relocations = {}
    for variant in variants.keys():
        proposed_relocations[variant] = get_relocated_kos(current_order_all_activities=list(variant),
                                                          optimal_ko_order=optimal_ko_order,
                                                          dependencies=dependencies,
                                                          start_activity_constraint=start_activity_constraint)

    return proposed_relocations, variants


def evaluate_knockout_reordering_io(analyzer: KnockoutAnalyzer,
                                    dependencies: dict[str, List[tuple[str, str]]] = None) -> dict:
    '''
    - Returns the observed ko-checks ordering (AS-IS)
    - Returns optimal ordering by KO effort
    - If a dependencies dictionary is provided, it will take it into account for the optimal ordering
    '''

    log = deepcopy(analyzer.discoverer.log_df)
    log.sort_values(
        by=[globalColumnNames.SIMOD_LOG_READER_CASE_ID_COLUMN_NAME,
            globalColumnNames.SIMOD_START_TIMESTAMP_COLUMN_NAME],
        inplace=True)

    report_df = deepcopy(analyzer.report_df)
    report_df[globalColumnNames.REPORT_COLUMN_REJECTION_RATE] = report_df[
        globalColumnNames.REPORT_COLUMN_REJECTION_RATE].str.replace('%', '')
    report_df[globalColumnNames.REPORT_COLUMN_REJECTION_RATE] = report_df[
        globalColumnNames.REPORT_COLUMN_REJECTION_RATE].astype(float)

    sorted_by_effort_and_rejection_rate = report_df.sort_values(
        by=[globalColumnNames.REPORT_COLUMN_EFFORT_PER_KO, globalColumnNames.REPORT_COLUMN_REJECTION_RATE],
        ascending=[True, False], inplace=False)

    optimal_order_names = sorted_by_effort_and_rejection_rate[globalColumnNames.REPORT_COLUMN_KNOCKOUT_CHECK].values
    sorted_efforts = sorted_by_effort_and_rejection_rate[
        [globalColumnNames.REPORT_COLUMN_KNOCKOUT_CHECK, globalColumnNames.REPORT_COLUMN_EFFORT_PER_KO,
         globalColumnNames.REPORT_COLUMN_REJECTION_RATE]]

    # Determine how many cases respect this order in the log
    # TODO: for the moment, keeping only non knocked out cases to analyze order. Could be useful to see also partial (ko-d) cases
    filtered = log[log['knocked_out_case'] == False]
    total_cases = filtered.groupby([globalColumnNames.PM4PY_CASE_ID_COLUMN_NAME]).ngroups

    if not (dependencies is None):
        # TODO: make it more flexible / generic, to include other sorting criteria
        optimal_order_names = get_ko_activities_sorted_with_dependencies(dependencies=dependencies,
                                                                         current_activity_order=list(
                                                                             optimal_order_names),
                                                                         efforts=sorted_efforts)

    cases_respecting_order = chained_eventually_follows(filtered, optimal_order_names) \
        .groupby([globalColumnNames.PM4PY_CASE_ID_COLUMN_NAME])

    sorted_efforts.reset_index(inplace=True)
    return {"optimal_ko_order": list(optimal_order_names),
            "cases_respecting_order": cases_respecting_order.ngroups,
            "total_cases": total_cases,
            "sorted_efforts": sorted_efforts}


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


def simplify_rule(ruleset: str) -> str:
    """
    intended to simplify redundant rule intervals, for example:

    1) [[Loan_Ammount=11693.71-16840.45] V [Loan_Ammount=>16840.45]]
        to:
       [[Loan_Ammount=>11693.71]]

    2) [[Monthly_Income=555.77-830.79] V [Monthly_Income=<555.77] V [Monthly_Income=830.79-1019.68]]
        to:
        [[Monthly_Income=<1019.68]]

    # TODO: First version: skips rules with ^ (more complex)
    """
    return ""
