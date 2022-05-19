import os
from multiprocessing import Process

from knockout_ios.utils.metrics import calc_overprocessing_waste, calc_processing_waste, calc_waiting_time_waste_v2, \
    read_metric_cache


def parallel_metrics_calc(ko_activities, log_df):
    # Fall back to sequential version when testing
    if os.environ.get('RUNNING_TESTS'):
        overprocessing_waste = calc_overprocessing_waste(ko_activities, log_df)
        processing_waste = calc_processing_waste(ko_activities, log_df)
        waiting_time_waste = calc_waiting_time_waste_v2(ko_activities, log_df)

        return overprocessing_waste, processing_waste, waiting_time_waste

    tasks = [calc_overprocessing_waste, calc_processing_waste,
             calc_waiting_time_waste_v2]
    running_tasks = [Process(target=task, args=(ko_activities, log_df)) for task in tasks]
    for running_task in running_tasks:
        running_task.start()
    for running_task in running_tasks:
        running_task.join()

    overprocessing_waste = read_metric_cache("overprocessing_waste.pkl")
    processing_waste = read_metric_cache("processing_waste.pkl")
    waiting_time_waste = read_metric_cache("waiting_time_waste.pkl")

    return overprocessing_waste, processing_waste, waiting_time_waste
