import pickle
from typing import Union

import streamlit as st

import pandas as pd
import seaborn as sns

from matplotlib import pyplot as plt, patches
from matplotlib.patches import Rectangle
from matplotlib.transforms import Bbox
from wittgenstein.base import Cond, Rule

sns.set()

log = pd.read_pickle("log.pkl")

with open("rule_change.pkl", "rb") as f:
    rule_change = pickle.load(f)

with open("rulesets.pkl", "rb") as f:
    rulesets = pickle.load(f)


def get_rectangle_from_condition(condition: Cond, datalims: Bbox, viewlims: Bbox) -> Union[Rectangle, None]:
    # bottom always 0
    # max height always as high as chart
    bottom = 0
    height = viewlims.y1

    # left and width are determined by condition
    # handle ranges, .e.g val = "555.77-830.79"
    if "-" in condition.val:
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


for ko_activity in rule_change:
    fig, ax = plt.subplots(len(rule_change[ko_activity]), 1)
    fig.suptitle(f'{ko_activity}')

    for attribute in rule_change[ko_activity]:
        f = sns.histplot(log, x=attribute, ax=ax)

        # Overlay the range of current attribute captured by the rules of current ko activity
        for rule in rulesets[ko_activity].rules:
            rectangles = get_rectangle_from_rule(rule, f.dataLim, f.viewLim, attribute)
            for rect in rectangles:
                f.add_patch(rect)

    # plt.show()
    st.pyplot(fig)
