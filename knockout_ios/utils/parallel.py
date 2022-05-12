from multiprocessing import Process

from knockout_ios.utils.metrics import calc_overprocessing_waste, calc_processing_waste, calc_waiting_time_waste_v2, \
    read_metric_cache, calc_available_cases_before_ko


def parallel_metrics_calc(ko_activities, log_df):
    tasks = [calc_available_cases_before_ko, calc_overprocessing_waste, calc_processing_waste,
             calc_waiting_time_waste_v2]
    running_tasks = [Process(target=task, args=(ko_activities, log_df)) for task in tasks]
    for running_task in running_tasks:
        running_task.start()
    for running_task in running_tasks:
        running_task.join()

    freqs = read_metric_cache("available_cases_before_ko.pkl")
    overprocessing_waste = read_metric_cache("overprocessing_waste.pkl")
    processing_waste = read_metric_cache("processing_waste.pkl")
    waiting_time_waste = read_metric_cache("waiting_time_waste.pkl")

    return freqs, overprocessing_waste, processing_waste, waiting_time_waste
