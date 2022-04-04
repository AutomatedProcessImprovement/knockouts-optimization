import os

import numpy as np
import pandas as pd

import datetime
import pprint

import itertools

from typing import Union


from ..external.Simod.src.simod import support_utils as sup
from ..external.Simod.src.simod.configuration import CalculationMethod
from ..external.Simod.src.simod.configuration import Configuration
from ..external.Simod.src.simod.extraction.role_discovery import ResourcePoolAnalyser
from ..external.Simod.src.simod.extraction.schedule_tables import TimeTablesCreator
from ..external.Simod.src.simod.readers.log_reader import LogReader

from lxml import etree
from ..external.Simod.src.simod.writers.xml_writer import (
    xml_template,
    QBP_NAMESPACE_URI,
    create_file,
)

TIMESTAMP_FORMAT = "%Y-%m-%d %H:%M:%S.%f"


def mine_resources_with_resource_table(
    settings: Configuration, log: LogReader = None, res_analyzer=None
):
    def create_resource_pool(resource_table, table_name) -> list:
        """Creates resource pools and associate them the default timetable in BIMP format"""
        resource_pool = [
            {
                "id": "QBP_DEFAULT_RESOURCE",
                "name": "SYSTEM",
                "total_amount": "20",
                "costxhour": "20",
                "timetable_id": table_name["arrival"],
            }
        ]
        data = sorted(resource_table, key=lambda x: x["role"])
        for key, group in itertools.groupby(data, key=lambda x: x["role"]):
            res_group = [x["resource"] for x in list(group)]
            r_pool_size = str(len(res_group))
            name = (
                table_name["resources"]
                if "resources" in table_name.keys()
                else table_name[key]
            )
            resource_pool.append(
                {
                    "id": sup.gen_id(),
                    "name": key,
                    "total_amount": r_pool_size,
                    "costxhour": "20",
                    "timetable_id": name,
                }
            )
        return resource_pool

    if res_analyzer == None:
        if log == None:
            raise Exception("No Log provided!")
        res_analyzer = ResourcePoolAnalyser(log, sim_threshold=settings.rp_similarity)
    ttcreator = TimeTablesCreator(settings)

    args = {
        "res_cal_met": settings.res_cal_met,
        "arr_cal_met": settings.arr_cal_met,
        "resource_table": res_analyzer.resource_table,
    }

    if not isinstance(args["res_cal_met"], CalculationMethod):
        args["res_cal_met"] = CalculationMethod.from_str(settings.res_cal_met)
    if not isinstance(args["arr_cal_met"], CalculationMethod):
        args["arr_cal_met"] = CalculationMethod.from_str(settings.arr_cal_met)

    ttcreator.create_timetables(args)
    resource_pool = create_resource_pool(
        res_analyzer.resource_table, ttcreator.res_ttable_name
    )
    resource_table = pd.DataFrame.from_records(res_analyzer.resource_table)

    return ttcreator.time_table, resource_pool, resource_table


def write_timetables_to_xml(timetable, resource_pool, out_file):
    my_doc = xml_template(resource_pool=resource_pool)
    # insert timetable
    if timetable is not None:
        ns = {"qbp": QBP_NAMESPACE_URI}
        childs = timetable.findall("qbp:timetable", namespaces=ns)
        node = my_doc.find("qbp:timetables", namespaces=ns)
        for i, child in enumerate(childs):
            node.insert((i + 1), child)

    create_file(out_file, etree.tostring(my_doc, pretty_print=True))


def remove_outliers(log: Union[LogReader, pd.DataFrame]) -> pd.DataFrame:
    if isinstance(log, LogReader):
        event_log = pd.DataFrame(log.data)
    else:
        event_log = log

    # calculating case durations
    cases_durations = list()
    for id, trace in event_log.groupby("caseid"):
        duration = (
            trace["end_timestamp"].max() - trace["start_timestamp"].min()
        ).total_seconds()
        cases_durations.append({"caseid": id, "duration_seconds": duration})
    cases_durations = pd.DataFrame(cases_durations)

    # merging data
    event_log = event_log.merge(cases_durations, how="left", on="caseid")

    # filtering rare events
    unique_cases_durations = event_log[["caseid", "duration_seconds"]].drop_duplicates()
    first_quantile = unique_cases_durations.quantile(0.1)
    last_quantile = unique_cases_durations.quantile(0.9)
    event_log = event_log[
        (event_log.duration_seconds <= last_quantile.duration_seconds)
        & (event_log.duration_seconds >= first_quantile.duration_seconds)
    ]
    event_log = event_log.drop(columns=["duration_seconds"])

    return event_log


def contextualize_timetable_item_start(activity_start: np.datetime64, timetable_item):
    act_dt = pd.to_datetime(activity_start)
    tt_dt = pd.to_datetime(timetable_item["fromTime"]).replace(
        year=act_dt.year, month=act_dt.month, day=act_dt.day
    )

    # move timetable dt to the first day of the week in which activity started
    # then traspose to its weekday
    start = tt_dt - datetime.timedelta(days=tt_dt.weekday())
    tt_dt = start + datetime.timedelta(days=timetable_item["fromWeekDay"])

    return tt_dt


def contextualize_timetable_item_end(activity_end: np.datetime64, timetable_item):
    act_dt = pd.to_datetime(activity_end)
    tt_dt = pd.to_datetime(timetable_item["toTime"]).replace(
        year=act_dt.year, month=act_dt.month, day=act_dt.day
    )

    # move timetable dt to the first day of the week in which activity started
    # then traspose to its weekday
    start = tt_dt - datetime.timedelta(days=tt_dt.weekday())
    tt_dt = start + datetime.timedelta(days=timetable_item["toWeekDay"])

    return tt_dt


def get_raw_duration(activity_start: np.datetime64, activity_end: np.datetime64):
    return (
        pd.to_datetime(activity_end) - pd.to_datetime(activity_start)
    ).total_seconds()


def delta_in_secs_ignore_date(end: np.datetime64, start: np.datetime64):
    dt = pd.to_datetime(end) - pd.to_datetime(start)
    return dt.seconds


# find time slot that minimizes difference with act. start
def find_closest(act_start, timetable):
    best_fit = 0

    cont = contextualize_timetable_item_start(act_start, timetable[0])
    m = abs((cont - pd.to_datetime(act_start)).total_seconds())

    for i, day in enumerate(timetable):
        cont = contextualize_timetable_item_start(act_start, day)
        diff = abs((pd.to_datetime(act_start) - cont).total_seconds())
        # print(f"{pd.to_datetime(act_start)} - {cont} = {diff}")
        if diff <= m:
            m = diff
            best_fit = i

    return best_fit


# find time slot that minimizes difference with act. end
def find_closest_to_end(act_end, timetable):
    best_fit = 0

    cont = contextualize_timetable_item_end(act_end, timetable[best_fit])
    m = abs((cont - pd.to_datetime(act_end)).total_seconds())

    for i, day in enumerate(timetable):
        cont = contextualize_timetable_item_end(act_end, day)
        diff = abs((cont - pd.to_datetime(act_end)).total_seconds())
        if diff <= m:
            m = diff
            best_fit = i

    return best_fit


def get_processing_time(
    activity_start: np.datetime64,
    activity_end: np.datetime64,
    role_timetable,
    raw_duration=0,
):

    if raw_duration <= 0:
        raw_duration = (
            pd.to_datetime(activity_end) - pd.to_datetime(activity_start)
        ).total_seconds()

    # Find starting time-slot and possible off-timetable times
    ACTIVITY_START_OFFTIMETABLE = False
    ACTIVITY_END_OFFTIMETABLE = False

    K = find_closest(activity_start, role_timetable)
    K_end = find_closest_to_end(activity_end, role_timetable)

    adj_activity_start = contextualize_timetable_item_start(
        activity_start, role_timetable[K]
    )
    adj_activity_end = contextualize_timetable_item_end(
        activity_end, role_timetable[K_end]
    )

    # Calculate offsets and adjust activity start/end if needed
    start_offset = adj_activity_start - pd.to_datetime(activity_start)
    end_offset = pd.to_datetime(activity_end) - adj_activity_end

    if start_offset.total_seconds() > 0:
        ACTIVITY_START_OFFTIMETABLE = True
        activity_start = adj_activity_start

    if end_offset.total_seconds() > 0:
        ACTIVITY_END_OFFTIMETABLE = True
        activity_end = adj_activity_end

    # fill an array of effective days
    effective_days = []
    date_counter = None
    wrap = 0

    while True:

        idx = K % len(role_timetable)
        item = role_timetable[idx]

        if len(effective_days) >= 1:
            if item["fromWeekDay"] < effective_days[-1]["fromWeekDay"]:
                wrap += 1
                # print("WRAP")

        curr_day = {
            "fromWeekDay": item["fromWeekDay"],
            "fromTime": contextualize_timetable_item_start(activity_start, item)
            + datetime.timedelta(days=int(wrap * 7)),
            "toWeekDay": item["toWeekDay"],
            "toTime": contextualize_timetable_item_end(activity_start, item)
            + datetime.timedelta(days=int(wrap * 7)),
        }

        effective_days.append(curr_day)

        date_counter = curr_day["toTime"]

        # print(
        #    f"{date_counter} >= {pd.to_datetime(activity_end)} : {date_counter >= pd.to_datetime(activity_end)}"
        # )
        # date counter "moves" in time along the effective days list, breaking the loop when we reach a point in time
        # that equals or surpasses the activity lifetime
        if date_counter >= pd.to_datetime(activity_end):
            break

        K += 1

    # pprint.pprint(effective_days)

    net_time = 0

    for i, day in enumerate(effective_days):
        if i == 0:
            # trim time before activity started
            net_time += delta_in_secs_ignore_date(day["toTime"], activity_start)

            # if activity is contained in same time-slot, trim time after activity ended
            if day["toTime"] > pd.to_datetime(activity_end):
                net_time -= delta_in_secs_ignore_date(
                    day["toTime"], pd.to_datetime(activity_end)
                )
        elif (i == len(effective_days) - 1) & (
            day["toTime"] > pd.to_datetime(activity_end)
        ):
            net_time += delta_in_secs_ignore_date(activity_end, day["fromTime"])
        else:
            net_time += delta_in_secs_ignore_date(day["toTime"], day["fromTime"])

        # print(net_time)

    if ACTIVITY_START_OFFTIMETABLE:
        net_time += start_offset.total_seconds()

    if ACTIVITY_END_OFFTIMETABLE:
        net_time += end_offset.total_seconds()

    # print(
    #    f"ACTIVITY_START_OFFTIMETABLE: {ACTIVITY_START_OFFTIMETABLE} // ACTIVITY_END_OFFTIMETABLE: {ACTIVITY_END_OFFTIMETABLE}"
    # )

    net_time = min(raw_duration, net_time)
    waiting_time = raw_duration - net_time

    return (
        net_time,
        waiting_time,
        ACTIVITY_START_OFFTIMETABLE,
        ACTIVITY_END_OFFTIMETABLE,
    )


def get_log_name(log_path):
    return os.path.basename(os.path.normpath(log_path))
