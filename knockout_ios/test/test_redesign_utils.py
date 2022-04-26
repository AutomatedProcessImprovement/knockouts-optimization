from knockout_ios.utils.redesign import get_sorted_with_dependencies


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
