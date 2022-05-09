from knockout_ios.pipeline_wrapper import run_pipeline


def test_ko_reorder_io():
    ko_redesign_adviser = run_pipeline(config_dir="config",
                                       config_file_name="synthetic_example_ko_order_io_pipeline_pytest.json",
                                       cache_dir="cache/synthetic_example",
                                       )

    assert ko_redesign_adviser.redesign_options['reordering']['optimal_order_names'].index("Check Risk") \
           < ko_redesign_adviser.redesign_options['reordering']['optimal_order_names'].index("Assess application")

    assert ko_redesign_adviser.redesign_options['reordering']['optimal_order_names'].index("Check Liability") \
           == len(ko_redesign_adviser.redesign_options['reordering']['optimal_order_names']) - 1
