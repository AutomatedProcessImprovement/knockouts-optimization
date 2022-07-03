import contextlib
import datetime
import logging
import pickle
import pprint
import time

from dataclasses import asdict

from knockout_ios.knockout_analyzer import KnockoutAnalyzer
from knockout_ios.knockout_redesign_adviser import KnockoutRedesignAdviser

from knockout_ios.utils.preprocessing.configuration import read_log_and_config
from knockout_ios.utils.synthetic_example.preprocessors import preprocessors_dict


class Pipeline:

    def __init__(self, config_file_name, cache_dir=None, config_dir="config", skip_cache_dump=False):
        if cache_dir is None:
            self.cache_dir = f"cache/{config_file_name}_{int(time.time())}"
        else:
            self.cache_dir = cache_dir

        self.skip_cache_dump = skip_cache_dump
        self.config_file_name = config_file_name
        self.config_dir = config_dir
        self.log_df = None
        self.config = None
        self.adviser = None

    def load_cache(self):
        pipeline_cache_file = open(f"{self.cache_dir}/{self.config_file_name}_pipeline_cache.pkl", "rb")
        data = pickle.load(pipeline_cache_file)
        pipeline_cache_file.close()
        return data

    def save_cache(self, adviser: KnockoutRedesignAdviser, analyzer: KnockoutAnalyzer, ko_redesign_reports,
                   ko_rule_discovery_stats, redesign_options,
                   low_confidence_warnings):
        if self.skip_cache_dump:
            return
        pipeline_cache_file = open(f"{self.cache_dir}/{self.config_file_name}_pipeline_cache.pkl", "wb")
        pickle.dump((adviser, analyzer, ko_redesign_reports, ko_rule_discovery_stats, redesign_options,
                     low_confidence_warnings), pipeline_cache_file)
        pipeline_cache_file.close()

    def read_log_and_config(self):
        self.log_df, self.config = read_log_and_config(self.config_dir, self.config_file_name, self.cache_dir)

    def run_analysis(self, pipeline_config=None, log_df=None):

        if pipeline_config is None:
            pipeline_config = self.config

        if log_df is None:
            log_df = self.log_df

        preprocessor_name = pipeline_config.custom_log_preprocessing_function
        if preprocessor_name is not None:
            pipeline_config.custom_log_preprocessing_function = preprocessors_dict[preprocessor_name]

        file_path = pipeline_config.redesign_results_file_path

        with open(file_path, "w", encoding="utf-8") as o:
            with contextlib.redirect_stdout(o):
                print(f"Knockouts Redesign Pipeline started @ {datetime.datetime.now()}")
                print("\nInput parameters:\n")
                pprint.pprint(asdict(pipeline_config))
                print("\n")
                tic = time.perf_counter()

                try:
                    if pipeline_config.always_force_recompute:
                        raise FileNotFoundError

                    adviser, analyzer, ko_redesign_reports, \
                    ko_rule_discovery_stats, redesign_options, low_confidence_warnings = self.load_cache()

                    analyzer.print_report()
                    adviser.print_report()

                except FileNotFoundError:
                    analyzer = KnockoutAnalyzer(log_df=log_df,
                                                config=pipeline_config,
                                                config_file_name=self.config_file_name,
                                                config_dir=self.config_dir,
                                                cache_dir=self.cache_dir,
                                                always_force_recompute=pipeline_config.always_force_recompute,
                                                quiet=True,
                                                custom_log_preprocessing_function=pipeline_config.custom_log_preprocessing_function)

                    analyzer.discover_knockouts()

                    _, _, ko_rule_discovery_stats, low_confidence_warnings = analyzer.compute_ko_rules(
                        algorithm=pipeline_config.rule_discovery_algorithm,
                        confidence_threshold=pipeline_config.confidence_threshold,
                        support_threshold=pipeline_config.support_threshold,
                        print_rule_discovery_stats=pipeline_config.print_rule_discovery_stats,
                        max_rules=pipeline_config.max_rules,
                        max_rule_conds=pipeline_config.max_rule_conds,
                        grid_search=pipeline_config.grid_search,
                        dl_allowance=pipeline_config.dl_allowance,
                        k=pipeline_config.k,
                        n_discretize_bins=pipeline_config.n_discretize_bins,
                        prune_size=pipeline_config.prune_size,
                        param_grid=pipeline_config.param_grid,
                    )

                    adviser = KnockoutRedesignAdviser(analyzer)
                    redesign_options, ko_redesign_reports = adviser.compute_redesign_options()

                    self.save_cache(adviser, analyzer, ko_redesign_reports, ko_rule_discovery_stats, redesign_options,
                                    low_confidence_warnings)

                toc = time.perf_counter()
                print("\n" + f"Knockouts Redesign Pipeline ended @ {datetime.datetime.now()}")
                print("\n" + f"Wall-clock execution time:  {str(datetime.timedelta(seconds=toc - tic))}")

                return adviser, analyzer.report_df, ko_redesign_reports, ko_rule_discovery_stats, redesign_options, low_confidence_warnings

    def run_pipeline(self):
        self.read_log_and_config()
        self.adviser, _, _, _, _, _ = self.run_analysis()
        return self.adviser

    def update_known_ko_activities(self, known_ko_activities):
        self.config.known_ko_activities = known_ko_activities

    def update_post_ko_activities(self, post_ko_activities):
        self.config.post_knockout_activities = post_ko_activities
        logging.info(f"Updated post-knockout activities: {self.config.post_knockout_activities}")

    def update_success_activities(self, success_activities):
        self.config.success_activities = success_activities
        logging.info(f"Updated success activities: {self.config.success_activities}")

    def update_log_attributes(self, attributes_to_consider):
        all_attributes = self.log_df.columns.values.tolist()
        attributes_to_ignore = [a for a in all_attributes if a not in attributes_to_consider]
        self.config.attributes_to_ignore = attributes_to_ignore
        logging.info(f"Updated attributes to ignore: {self.config.attributes_to_ignore}")
