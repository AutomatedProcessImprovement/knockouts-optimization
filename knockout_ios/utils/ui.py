from typing import Union

from matplotlib import patches
from matplotlib.patches import Rectangle
from matplotlib.transforms import Bbox
from wittgenstein.base import Cond, Rule


def get_rectangle_from_condition(condition: Cond, datalims: Bbox, viewlims: Bbox) -> Union[Rectangle, None]:
    # bottom always 0
    # max height always as high as chart
    bottom = 0
    height = viewlims.y1

    # left and width are determined by condition
    # handle ranges, .e.g val = "555.77-830.79"
    if "^" in condition.val:
        # For the moment, skipping complex conditions
        return None
    elif "-" in condition.val:
        left, right = condition.val.split("-")
        left = float(left)
        right = float(right)
        width = right - left
    elif "<" in condition.val:
        left = datalims.x0
        width = float(condition.val.replace("<", ""))
    elif ">" in condition.val:
        left = float(condition.val.replace(">", ""))
        width = datalims.x1
    else:
        return None

    return patches.Rectangle((left, bottom), width, height,
                             alpha=0.1,
                             facecolor="red")


def get_rectangle_from_rule(rule: Rule, datalims: Bbox, viewlims: Bbox, attribute) -> list[Rectangle]:
    rectangles = []
    for cond in rule.conds:
        if attribute.replace(" ", "_") != cond.feature:
            continue
        rect = get_rectangle_from_condition(cond, datalims, viewlims)

        if not (rect is None):
            rectangles.append(rect)

    return rectangles
