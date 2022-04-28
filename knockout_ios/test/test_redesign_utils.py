from collections import Counter

import pandas as pd
import pytest

from knockout_ios.utils.constants import SIMOD_LOG_READER_CASE_ID_COLUMN_NAME, SIMOD_END_TIMESTAMP_COLUMN_NAME
from knockout_ios.utils.redesign import get_sorted_with_dependencies, find_producers


def test_get_sorted_with_dependencies_1():
    order = ["A", "C", "B"]
    dependencies = {k: [] for k in order}
    dependencies["A"].append(("attr_from_C", "C"))
    dependencies["A"].append(("attr_from_B", "B"))

    optimal_order = get_sorted_with_dependencies(dependencies=dependencies, optimal_order_names=order)

    assert optimal_order == ["C", "B", "A"]


def test_get_sorted_with_dependencies_2():
    order = ["A", "C", "B"]
    dependencies = {k: [] for k in order}
    dependencies["C"].append(("attr_from_B", "B"))
    dependencies["A"].append(("attr_from_C", "C"))
    dependencies["A"].append(("attr_from_B", "B"))

    optimal_order = get_sorted_with_dependencies(dependencies=dependencies, optimal_order_names=order)

    assert optimal_order == ["B", "C", "A"]


@pytest.mark.skip(reason="TODO: create traces for test")
def test_find_producer_activity_simple():
    attribute = "attr_produced_by_B"
    ko_activity = "D"

    # in this case, the attribute stops being null from activity B, and never changes since then.
    # therefore, B is considered the producer.
    # TODO: create traces for test
    log = []
    log = pd.DataFrame(log)
    log.set_index(SIMOD_LOG_READER_CASE_ID_COLUMN_NAME, inplace=True)
    log.sort_values(by=SIMOD_END_TIMESTAMP_COLUMN_NAME)

    producers = find_producers(attribute, ko_activity, log)

    assert len(producers) > 0
    assert Counter(producers).most_common(1)[0][0] == ["B"]


@pytest.mark.skip(reason="TODO: create traces for test")
def test_find_producer_activity_advanced():
    attribute = "attr_produced_by_D"
    ko_activity = "F"

    # in this case, the attribute stops being null from activity B, but keeps changing until activity D.
    # therefore, D is considered the producer.
    # TODO: create traces for test
    log = []
    log = pd.DataFrame(log)
    log.set_index(SIMOD_LOG_READER_CASE_ID_COLUMN_NAME, inplace=True)
    log.sort_values(by=SIMOD_END_TIMESTAMP_COLUMN_NAME)

    producers = find_producers(attribute, ko_activity, log)

    assert len(producers) > 0
    assert Counter(producers).most_common(1)[0][0] == ["D"]
