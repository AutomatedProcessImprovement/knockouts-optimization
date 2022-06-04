import difflib

import matplotlib.pyplot as plt
import seaborn as sns

from knockout_ios.utils.constants import globalColumnNames


def format_null_activity(activity):
    if activity is False:
        return "Not Knocked Out"
    return activity


def format_for_post_proc(df, id_col_name=globalColumnNames.PM4PY_CASE_ID_COLUMN_NAME,
                         duration_col_name=globalColumnNames.DURATION_COLUMN_NAME):
    by_case = df.groupby(id_col_name).agg({duration_col_name: ['sum'],
                                           'knockout_activity': lambda x: format_null_activity(x.iloc[0]),
                                           'knocked_out_case': lambda x: x.iloc[0]})

    by_case.columns = [a[0] if (a[1] == "<lambda>") else "_".join(a) for a in by_case.columns.to_flat_index()]
    by_case = by_case.rename(columns={f"{duration_col_name}_sum": globalColumnNames.PROCESSING_TIME})

    return by_case


def plot_cycle_times_per_ko_activity(by_case, ko_activities, show_outliers=False):
    plt.figure()
    by_case.groupby('knockout_activity').boxplot(
        column=globalColumnNames.PROCESSING_TIME,
        fontsize=12,
        subplots=True,
        layout=(1, len(ko_activities) + 1),
        figsize=(12, 6),
        showfliers=show_outliers
    )

    plt.suptitle(f'{globalColumnNames.PROCESSING_TIME} by KO Activity')


def plot_ko_activities_count(by_case):
    # Visualize knocked-out cases per ko activity
    plt.figure()
    sns.set(rc={'figure.facecolor': 'white'})
    ax = sns.countplot(y=by_case['knockout_activity'],
                       order=by_case['knockout_activity'].value_counts(ascending=False).index)
    abs_values = by_case['knockout_activity'].value_counts(ascending=False).values
    ax.bar_label(label_type="center", container=ax.containers[0], labels=abs_values)

    ax.figure.tight_layout()

    plt.show()


def seconds_to_hms(seconds):
    seconds = int(seconds)
    m, s = divmod(seconds, 60)
    h, m = divmod(m, 60)
    days, h = divmod(h, 24)
    if (days > 0) or (h > 24):
        return f'{days:d} days, {h:d}:{m:02d}:{s:02d}'

    return f'{h:d}:{m:02d}:{s:02d}'


# red = lambda text: f"\033[38;2;255;0;0m{text}\033[38;2;255;255;255m"
# green = lambda text: f"\033[38;2;0;255;0m{text}\033[38;2;255;255;255m"
# blue = lambda text: f"\033[38;2;0;0;255m{text}\033[38;2;255;255;255m"
# white = lambda text: f"\033[38;2;255;255;255m{text}\033[38;2;255;255;255m"

# TODO: find way to output colored for UI but not for .txt
green = lambda text: f'<span style="color:Green;">{text}</span>'
white = lambda text: text


def get_edits_string(old, new):
    # Source: https://stackoverflow.com/a/64404008/8522453

    result = ""
    codes = difflib.SequenceMatcher(a=old, b=new).get_opcodes()
    for code in codes:
        if code[0] == "equal":
            result += white(old[code[1]:code[2]])
        elif code[0] == "insert":
            result += green(new[code[3]:code[4]])
        elif code[0] == "replace":
            result += (green(new[code[3]:code[4]]))
    return result


def out_pretty(ruleset):
    """Print Ruleset line-by-line."""
    ruleset_str = (
        str([str(rule) for rule in ruleset])
            .replace(" ", "")
            .replace(",", " V\n")
            .replace("'", "")
            .replace("^", " ^ ")
    )
    return ruleset_str
