import pandas as pd
import numpy as np

import time

from copy import deepcopy

from ..external.Simod.src.simod.configuration import (
    QBP_NAMESPACE_URI,
)
from ..external.Simod.src.simod.extraction.role_discovery import ResourcePoolAnalyser

from .pt_cleaning_util import (
    mine_resources_with_resource_table,
    write_timetables_to_xml,
    get_processing_time,
    get_log_name,
)


# TODO: this code is really low; it's from when I was starting to even familiarize with the project.
#  May be worth to revisit if it will be used again

def clean_processing_times_with_calendar(log, config, _res_analyzer=None):
    # # Processing Times Pre-Processing w.r.t Resource Calendar
    #
    # Calendar Discovery: [thesis](https://comserv.cs.ut.ee/ati_thesis/datasheet.php?language=en) / [source code](https://bitbucket.org/Ibrahim_Mahdy/calendar/src/master/)
    #
    try:
        log_df = pd.DataFrame(log.data)
    except AttributeError:
        log_df = log

    # ### Discover Resource Pool & Timetables

    # uses role_discovery & schedule_tables modules,
    # calls external calendar discovery impl. in .jar

    if _res_analyzer:
        res_analyzer = _res_analyzer
    else:
        res_analyzer = ResourcePoolAnalyser(log, sim_threshold=config.rp_similarity)

    time_table, resource_pool, resource_table = mine_resources_with_resource_table(
        config, res_analyzer=res_analyzer
    )

    write_timetables_to_xml(
        deepcopy(time_table),
        deepcopy(resource_pool),
        f"{config.output}/{get_log_name(config.log_path)}_cal.xml",
    )

    # ### Clean-up Activity Durations

    # - associate each activity to a role
    def getRole(resource):
        if pd.isna(resource):
            return pd.NA
        if resource_table.loc[resource_table["resource"] == resource]["role"].size:
            return resource_table.loc[resource_table["resource"] == resource][
                "role"
            ].values[0]
        else:
            return pd.NA

    def getTimetableId(role):
        if pd.isna(role):
            return pd.NA
        return resource_pool_df.loc[resource_pool_df["name"] == role][
            "timetable_id"
        ].values[0]

    def weekdayToNum(daystr):
        return time.strptime(daystr.lower(), "%A").tm_wday

    def hourToNpDatetime(hourstr):
        datestr = f"2021-01-01 {hourstr.split('+')[0]}"
        return np.datetime64(datestr)

    def extract_processing_time(activity):
        if pd.isna(activity["timetable_id"]):
            return activity["@@duration"], activity["@@duration"], False, False

        role_timetable = tt_dict.get(activity["timetable_id"])
        net, idle, start_offtimetable, end_offtimetable = get_processing_time(
            np.datetime64(activity["start_timestamp"]),
            np.datetime64(activity["end_timestamp"]),
            role_timetable,
            activity["@@duration"],
        )
        return net, idle, start_offtimetable, end_offtimetable

    log_df["Role"] = log_df.apply(lambda row: getRole(row.user), axis=1)

    # - for each activity, lookup associated resource's timetable
    resource_pool_df = pd.DataFrame(resource_pool)

    tt_list = time_table.findall("qbp:timetable", namespaces={"qbp": QBP_NAMESPACE_URI})
    tt_dict = {}
    for e in tt_list:
        rules = list(e.findall("qbp:rules", namespaces={"qbp": QBP_NAMESPACE_URI})[0])
        rules = map(
            lambda elem: {
                "fromTime": hourToNpDatetime(elem.get("fromTime")),
                "toTime": hourToNpDatetime(elem.get("toTime")),
                "fromWeekDay": weekdayToNum(elem.get("fromWeekDay")),
                "toWeekDay": weekdayToNum(elem.get("toWeekDay")),
            },
            rules,
        )

        tt_dict[e.get("id")] = list(rules)

    log_df["timetable_id"] = log_df.apply(lambda row: getTimetableId(row.Role), axis=1)

    log_df[
        [
            "processing_time",
            "waiting_time",
            "started_offtimetable",
            "ended_offtimetable",
        ]
    ] = log_df.apply(
        lambda row: extract_processing_time(row), axis=1, result_type="expand"
    )

    return log_df
