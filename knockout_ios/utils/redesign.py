from collections import Counter
from typing import List

import numpy as np
import pandas as pd
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


def get_sorted_with_dependencies(dependencies: dict[str, List[tuple[str, str]]], optimal_order_names: List[str]):
    # TODO: test with multiple dependencies

    for knockout_activity in optimal_order_names:
        _dependencies = dependencies[knockout_activity]
        if not (len(_dependencies) > 0):
            continue

        # sort deps by the index of every second element of the tuples in optimal_order_names
        _dependencies = sorted(_dependencies, key=lambda x: optimal_order_names.index(x[1]))

        # Remove knockout_activity from optimal_order_names,
        # find where is attribute_value_producer in optimal_order_names,
        # then insert knockout_activity after attribute_value_producer

        for pair in _dependencies:
            attribute_value_producer = pair[1]
            optimal_order_names.remove(knockout_activity)
            idx = optimal_order_names.index(attribute_value_producer)
            optimal_order_names.insert(idx + 1, knockout_activity)

    return optimal_order_names


def evaluate_knockout_relocation_io(analyzer: KnockoutAnalyzer) -> dict[str, List[tuple[str, str]]]:
    """
    - Returns dependencies between log activities and attributes required by knockout checks
    """
    if analyzer.ruleset_algorithm == "IREP":
        rule_discovery_dict = analyzer.IREP_rulesets
    elif analyzer.ruleset_algorithm == "RIPPER":
        rule_discovery_dict = analyzer.RIPPER_rulesets
    else:
        raise ValueError("Unknown ruleset algorithm")

    # for every knockout activity, there will be a list of tuples (attribute of KO rule, name of producer activity)
    dependencies = {k: [] for k in rule_discovery_dict.keys()}

    log = analyzer.discoverer.log_df
    log.set_index(SIMOD_LOG_READER_CASE_ID_COLUMN_NAME, inplace=True)

    for ko_activity in tqdm(rule_discovery_dict.keys(), desc="Searching KO activity dependencies"):
        ruleset = rule_discovery_dict[ko_activity][0]
        required_attributes = get_attribute_names_from_ruleset(ruleset)

        for attribute in required_attributes:
            # Find after which activity the attribute is available in the log
            producers = []
            for caseid in log.index.unique():
                case = log.loc[caseid]
                knockout_activity_column_values = case['knockout_activity']
                if not (len(knockout_activity_column_values.values) > 0):
                    continue
                if knockout_activity_column_values.values[0] != ko_activity:
                    continue

                activities_where_unavailable = case[np.isnan(case[attribute].values)][PM4PY_ACTIVITY_COLUMN_NAME].values
                if len(activities_where_unavailable) > 0:
                    producer_activity = activities_where_unavailable[-1]
                    producers.append(producer_activity)

            if len(producers) > 0:
                # get most frequent producer activity
                producers = Counter(producers).most_common(1)[0][0]
                if producers == ko_activity:
                    continue
                dependencies[ko_activity].append((attribute, producers))

    return dependencies


def evaluate_knockout_reordering_io(analyzer: KnockoutAnalyzer,
                                    dependencies: dict[str, List[tuple[str, str]]] = None) -> dict:
    '''
    - Returns the observed ko-checks ordering (AS-IS)
    - Returns optimal ordering by KO effort
    - If a dependencies dictionary is provided, it will take it into account for the optimal ordering
    '''

    log = analyzer.discoverer.pm4py_formatted_df

    sorted_by_effort = analyzer.report_df.sort_values(by=[REPORT_COLUMN_EFFORT_PER_KO], ascending=True, inplace=False)
    optimal_order_names = sorted_by_effort[REPORT_COLUMN_KNOCKOUT_CHECK].values

    # Determine how many cases respect this order in the log
    # TODO: for the moment, keeping only non knocked out cases to analyze order. Could be useful to see also partial (ko-d) cases
    filtered = log[log['knocked_out_case'] == False]
    total_cases = filtered.groupby([PM4PY_CASE_ID_COLUMN_NAME]).ngroups

    if dependencies is not None:
        # TODO: make it more flexible / generic, to include other sorting criteria
        optimal_order_names = get_sorted_with_dependencies(dependencies, list(optimal_order_names))
        pass

    cases_respecting_order = chained_eventually_follows(filtered, optimal_order_names) \
        .groupby([PM4PY_CASE_ID_COLUMN_NAME])

    observed_ko_order = get_observed_ko_checks_order(log, analyzer.discoverer.ko_activities)

    return {"optimal_order_names": list(optimal_order_names), "cases_respecting_order": cases_respecting_order.ngroups,
            "total_cases": total_cases,
            "observed_ko_order": observed_ko_order}


def evaluate_knockout_rule_change_io(analyzer: KnockoutAnalyzer):
    return []
