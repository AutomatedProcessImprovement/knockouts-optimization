import matplotlib.pyplot as plt
import seaborn as sns

from knockout_ios.utils.constants import *


def format_null_activity(activity):
    if activity is False:
        return "N/A"
    return activity


def format_for_post_proc(df, id_col_name=PM4PY_CASE_ID_COLUMN_NAME, duration_col_name=DURATION_COLUMN_NAME):
    by_case = df.groupby(id_col_name).agg({duration_col_name: ['sum'],
                                           'knockout_activity': lambda x: format_null_activity(x.iloc[0]),
                                           'knocked_out_case': lambda x: x.iloc[0]})

    by_case.columns = [a[0] if (a[1] == "<lambda>") else "_".join(a) for a in by_case.columns.to_flat_index()]
    by_case = by_case.rename(columns={f"{duration_col_name}_sum": 'cycle_time'})

    return by_case


def plot_cycle_times_per_ko_activity(by_case, ko_activities, show_outliers=False):
    plt.figure()
    by_case.groupby('knockout_activity').boxplot(
        column='cycle_time',
        fontsize=12,
        subplots=True,
        layout=(1, len(ko_activities) + 1),
        figsize=(12, 6),
        showfliers=show_outliers
    )

    plt.suptitle('Cycle Times by KO Activity')


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
