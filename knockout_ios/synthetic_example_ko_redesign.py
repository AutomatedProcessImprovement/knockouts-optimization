from knockout_ios.pipeline_wrapper import run_pipeline


def ko_reorder_io():
    ko_redesign_adviser = run_pipeline(config_dir="config",
                                       config_file_name="synthetic_example_ko_order_io_pipeline_test.json",
                                       cache_dir="cache/synthetic_example")

    # assert ko_redesign_adviser.redesign_options['reordering']['current_activity_order'].index("Check Risk") \
    #        < ko_redesign_adviser.redesign_options['reordering']['current_activity_order'].index("Assess application")
    #
    # assert ko_redesign_adviser.redesign_options['reordering']['current_activity_order'].index("Check Liability") \
    #        == len(ko_redesign_adviser.redesign_options['reordering']['current_activity_order']) - 1


if __name__ == "__main__":
    ko_reorder_io()
