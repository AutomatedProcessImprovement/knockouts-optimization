import pandas as pd

from knockout_ios.utils.synthetic_example.attribute_enricher import enrich_log_df, enrich_log_df_with_masked_attributes, \
    RuntimeAttribute, enrich_log_df_fixed_values


def enrich_log_with_fully_known_attributes(ko_analyzer: 'KnockoutAnalyzer', log: pd.DataFrame,
                                           cache_file_path: str = None, fixed_values=False) -> pd.DataFrame:
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
            if fixed_values:
                enriched_log = enrich_log_df_fixed_values(log)
            else:
                enriched_log = enrich_log_df(log)
            enriched_log.to_pickle(cache_file_path)

            return enriched_log


def enrich_log_with_fully_known_attributes_fixed_values_wrapper(ko_analyzer: 'KnockoutAnalyzer', log: pd.DataFrame,
                                                                cache_file_path: str = None) -> pd.DataFrame:
    return enrich_log_with_fully_known_attributes(ko_analyzer, log, cache_file_path, fixed_values=True)


def enrich_log_for_ko_order_advanced_test(ko_analyzer: 'KnockoutAnalyzer',
                                          log: pd.DataFrame,
                                          cache_file_path: str = None, fixed_values=False) -> pd.DataFrame:
    '''
    Returns the log enriched with attributes whose values are known only after a certain activity has been performed,
    in particular, the 'Aggregated Risk Score' can be computed after activity 'Check Risk' and is needed by the activity
    'Aggregated Risk Score Check'
    '''

    # TODO: remove this if the .xes exporting problem is ever fixed - not critical anyway.
    if "synthetic_example" in ko_analyzer.config_file_name:
        try:
            if ko_analyzer.always_force_recompute:
                raise FileNotFoundError

            return pd.read_pickle(cache_file_path)

        except FileNotFoundError:
            enriched_log = enrich_log_df_with_masked_attributes(log,
                                                                [RuntimeAttribute(
                                                                    attribute_name='Aggregated Risk Score',
                                                                    value_provider_activity='Check Risk')],
                                                                fixed_values=fixed_values)
            enriched_log.to_pickle(cache_file_path)

            return enriched_log


def enrich_log_for_ko_order_advanced_test_fixed_values_wrapper(ko_analyzer: 'KnockoutAnalyzer', log: pd.DataFrame,
                                                               cache_file_path: str = None) -> pd.DataFrame:
    return enrich_log_for_ko_order_advanced_test(ko_analyzer, log, cache_file_path, fixed_values=True)
