import pandas as pd

from knockout_ios.utils.synthetic_example.attribute_enricher import enrich_log_df, enrich_log_df_with_value_providers


def enrich_synthetic_example_log_v1(ko_analyzer: 'KnockoutAnalyzer', log: pd.DataFrame,
                                    cache_file_path: str = None) -> pd.DataFrame:
    '''
    Returns the log enriched with attributes whose values are known from the begging of the case
    '''

    # TODO: remove this if the .xes exporting problem is ever fixed - not critical anyway.
    if "synthetic_example" in ko_analyzer.config_file_name:
        try:
            if ko_analyzer.always_force_recompute:
                raise FileNotFoundError

            return pd.read_pickle(cache_file_path)

        except FileNotFoundError:
            log = enrich_log_df(ko_analyzer.discoverer.log_df)
            log.to_pickle(cache_file_path)

            return log


def enrich_synthetic_example_log_v2(ko_analyzer: 'KnockoutAnalyzer', log: pd.DataFrame,
                                    cache_file_path: str = None) -> pd.DataFrame:
    '''
    Returns the log enriched with attributes whose values are known only after a certain activity has been performed
    '''

    # TODO: remove this if the .xes exporting problem is ever fixed - not critical anyway.
    if "synthetic_example" in ko_analyzer.config_file_name:
        try:
            if ko_analyzer.always_force_recompute:
                raise FileNotFoundError

            return pd.read_pickle(cache_file_path)

        except FileNotFoundError:
            log = enrich_log_df_with_value_providers(ko_analyzer.discoverer.log_df, [])
            log.to_pickle(cache_file_path)

            return log
