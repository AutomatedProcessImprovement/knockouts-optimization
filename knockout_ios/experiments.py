from knockout_ios.pipeline_wrapper import run_pipeline


def synthetic_example():
    ko_redesign_adviser = run_pipeline(config_dir="config",
                                       config_file_name="synthetic_example_ko_order_io_pipeline_test.json",
                                       cache_dir="cache/synthetic_example")

    assert ko_redesign_adviser.redesign_options['reordering']['optimal_ko_order'] == ["Check Monthly Income",
                                                                                      "Check Risk",
                                                                                      "Assess application",
                                                                                      "Check Liability"]


def synthetic_example_1_timest():
    ko_redesign_adviser = run_pipeline(config_dir="config",
                                       config_file_name="synthetic_example_ko_order_io_pipeline_test_1_timest.json",
                                       cache_dir="cache/synthetic_example")


def bpi_2017_1k():
    ko_redesign_adviser = run_pipeline(config_dir="config",
                                       config_file_name="bpi_2017_1k.json",
                                       cache_dir="cache/bpi_2017_1k")


def bpi_2017_8k():
    ko_redesign_adviser = run_pipeline(config_dir="config",
                                       config_file_name="bpi_2017_8k.json",
                                       cache_dir="cache/bpi_2017_8k")


def bpi_2017_21k():
    ko_redesign_adviser = run_pipeline(config_dir="config",
                                       config_file_name="bpi_2017_21k.json",
                                       cache_dir="cache/bpi_2017_21k")


def bpi_2013():
    ko_redesign_adviser = run_pipeline(config_dir="config",
                                       config_file_name="bpi_2013.json",
                                       cache_dir="cache/bpi_2013")


def bpi_2017_1k_W():
    ko_redesign_adviser = run_pipeline(config_dir="config",
                                       config_file_name="bpi_2017_1k_W.json",
                                       cache_dir="cache/bpi_2017_1k_W")


def envpermit():
    ko_redesign_adviser = run_pipeline(config_dir="config",
                                       config_file_name="envpermit.json",
                                       cache_dir="cache/envpermit")


if __name__ == "__main__":
    # synthetic_example()
    synthetic_example_1_timest()
    # bpi_2017_1k()
    # bpi_2017_8k()
    # bpi_2017_21k()
    # bpi_2013()
    # bpi_2017_1k_W()
    # envpermit()
