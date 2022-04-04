import pandas as pd
import numpy as np

from copy import deepcopy

# import sys
# sys.path.append('..')

# from pt_cleaning import clean_processing_times_with_calendar


class ActivityTransformer:
    def __init__(
        self,
        log_df,
        case_id_col="caseid",
        activity_col="task",
        duration_col="processing_time",
        st_timestamp_col="start_timestamp",
        end_timestamp_col="end_timestamp",
        resource_col="user",
    ):
        self.log_df = log_df
        self.case_id_col = case_id_col
        self.activity_col = activity_col
        self.duration_col = duration_col
        self.st_timestamp_col = st_timestamp_col
        self.end_timestamp_col = end_timestamp_col
        self.resource_col = resource_col

        self.repetitions_column = "ocurrences_in_trace"
        self.cumulative_time_column = "accumulated_activity_time"
        self.index_in_trace_column = "index_in_trace"
        self.resource_occupation_st_column = "res_st_oc"
        self.resource_occupation_end_column = "res_end_oc"

    def transform(self):
        self.add_repetitions_and_time()
        self.add_activity_positions()
        self.add_resource_occupations()

        for col in [
            self.repetitions_column,
            self.cumulative_time_column,
            self.index_in_trace_column,
            self.resource_occupation_st_column,
            self.resource_occupation_end_column,
        ]:
            self.log_df[col] = self.log_df[col].astype(int)

        return deepcopy(self.log_df)

    def add_repetitions_and_time(self):

        grouped = (
            self.log_df.groupby([self.case_id_col, self.activity_col])[
                self.duration_col
            ]
            .agg(["size", "sum"])
            .reset_index()
            .rename(
                columns={
                    "size": self.repetitions_column,
                    "sum": self.cumulative_time_column,
                }
            )
        )

        self.log_df = pd.merge(
            self.log_df, grouped, on=[self.case_id_col, self.activity_col]
        )

    def add_activity_positions(self):
        grouped = self.log_df.groupby([self.case_id_col])

        for key in grouped.groups.keys():
            srt = (
                grouped.get_group(key)
                .sort_values(by=self.st_timestamp_col)
                .reset_index()
            )

            for i, r in srt.iterrows():
                self.log_df.loc[r["index"], self.index_in_trace_column] = i

    def add_resource_occupations(self):
        for index, row in self.log_df.iterrows():
            resource = row[self.resource_col]
            start = row[self.st_timestamp_col]
            end = row[self.end_timestamp_col]
            filt_st = self.log_df[
                (self.log_df[self.st_timestamp_col] <= start)
                & (self.log_df[self.end_timestamp_col] > start)
                & (self.log_df[self.resource_col] == resource)
            ]
            filt_end = self.log_df[
                (self.log_df[self.st_timestamp_col] <= end)
                & (self.log_df[self.end_timestamp_col] > end)
                & (self.log_df[self.resource_col] == resource)
            ]

            self.log_df.loc[index, self.resource_occupation_st_column] = len(filt_st)
            self.log_df.loc[index, self.resource_occupation_end_column] = len(filt_end)

    def get_activity_count(self, case_id, activity):

        act = self.get_activity(case_id, activity)

        try:
            return act[self.repetitions_column]
        except:
            return None

    def get_activity_cumulative_time(self, case_id, activity):

        act = self.get_activity(case_id, activity)

        try:
            return act[self.cumulative_time_column]
        except:
            return None

    def get_activity_positions(self, case_id, activity):

        activities = self.get_activity(case_id, activity, single=False)

        try:
            return activities[self.index_in_trace_column].values
        except:
            return None

    def get_activity(self, case_id, activity, index_in_trace=-1, single=True):

        try:
            if index_in_trace != -1:
                match = self.log_df.loc[
                    (self.log_df[self.case_id_col] == case_id)
                    & (self.log_df[self.activity_col] == activity)
                    & (self.log_df[self.index_in_trace_column] == index_in_trace)
                ]
                return match.iloc[0]
            else:
                match = self.log_df.loc[
                    (self.log_df[self.case_id_col] == case_id)
                    & (self.log_df[self.activity_col] == activity)
                ]
                if single:
                    return match.iloc[0]
                else:
                    return match
        except:
            return None

    def get_resource_occupations(self, case_id, activity, index_in_trace=-1):

        act = self.get_activity(case_id, activity, index_in_trace)

        try:
            return (
                act[self.resource_occupation_st_column],
                act[self.resource_occupation_end_column],
            )

        except:
            return None
