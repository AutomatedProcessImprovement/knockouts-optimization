from knockout_ios.utils.constants import *


def find_rejection_rates(log_df, ko_activities):

    # for every ko_activity,
    # P = how many cases were knocked out by it / how many cases contain it
    knock_out_counts_by_activity = log_df.drop_duplicates(subset=SIMOD_LOG_READER_CASE_ID_COLUMN_NAME) \
        .groupby('knockout_activity') \
        .count()[SIMOD_LOG_READER_CASE_ID_COLUMN_NAME]

    rates = {}
    for activity in ko_activities:
        cases_knocked_out_by_activity = knock_out_counts_by_activity.get(activity)
        cases_containing_activity = log_df[log_df[SIMOD_LOG_READER_ACTIVITY_COLUMN_NAME] == activity]\
            .drop_duplicates(subset=SIMOD_LOG_READER_CASE_ID_COLUMN_NAME)\
            .count()[SIMOD_LOG_READER_CASE_ID_COLUMN_NAME]

        rates[activity] = round(cases_knocked_out_by_activity / cases_containing_activity, 3)

    return rates
