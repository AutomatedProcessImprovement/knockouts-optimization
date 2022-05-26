from knockout_ios.pipeline_wrapper import Pipeline


def synthetic_example():
    ko_redesign_adviser = Pipeline(config_dir="config",
                                   config_file_name="synthetic_example_enriched.json",
                                   cache_dir="cache/synthetic_example_enriched").run_pipeline()

    assert ko_redesign_adviser.redesign_options['reordering']['optimal_ko_order'] == ["Check Monthly Income",
                                                                                      "Check Risk",
                                                                                      "Assess application",
                                                                                      "Check Liability"]


def bpi_2017_1k_W():
    ko_redesign_adviser = Pipeline(config_dir="config",
                                   config_file_name="bpi_2017_1k_W.json",
                                   cache_dir="cache/bpi_2017_1k_W").run_pipeline()


def envpermit():
    ko_redesign_adviser = Pipeline(config_dir="config",
                                   config_file_name="envpermit.json",
                                   cache_dir="cache/envpermit").run_pipeline()


if __name__ == "__main__":
    # synthetic_example()
    # bpi_2017_1k_W()
    envpermit()
