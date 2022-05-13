from knockout_ios.pipeline_wrapper import run_pipeline


def ko_reorder_io():
    ko_redesign_adviser = run_pipeline(config_dir="config",
                                       config_file_name="synthetic_example_ko_order_io_pipeline_test.json",
                                       cache_dir="cache/synthetic_example")

    assert ko_redesign_adviser.redesign_options['reordering']['optimal_ko_order'] == ["Check Monthly Income",
                                                                                      "Check Risk",
                                                                                      "Assess application",
                                                                                      "Check Liability"]


def real_log_1():
    ko_redesign_adviser = run_pipeline(config_dir="config",
                                       config_file_name="real_log_1.json",
                                       cache_dir="cache/real_log_1")


def real_log_2():
    ko_redesign_adviser = run_pipeline(config_dir="config",
                                       config_file_name="real_log_2.json",
                                       cache_dir="cache/real_log_2")


if __name__ == "__main__":
    # ko_reorder_io()
    real_log_1()
    # real_log_2()
