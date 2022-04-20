import pandas as pd

from knockout_ios.utils.synthetic_example.attribute_enricher import enrich_log_df, enrich_log_df_with_value_providers, \
    RuntimeAttribute


def enrich_log_with_fully_known_attributes(ko_analyzer: 'KnockoutAnalyzer', log: pd.DataFrame,
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
            enriched_log = enrich_log_df(log)
            enriched_log.to_pickle(cache_file_path)

            return enriched_log


def enrich_log_for_ko_order_advanced_test(ko_analyzer: 'KnockoutAnalyzer',
                                          log: pd.DataFrame,
                                          cache_file_path: str = None) -> pd.DataFrame:
    '''
    Returns the log enriched with attributes whose values are known only after a certain activity has been performed,
    in particular, the 'External Risk Score' is computed by activity 'Get External Risk Score' and is required by
    the activity 'Assess application'
    '''

    # TODO: remove this if the .xes exporting problem is ever fixed - not critical anyway.
    if "synthetic_example" in ko_analyzer.config_file_name:
        try:
            if ko_analyzer.always_force_recompute:
                raise FileNotFoundError

            return pd.read_pickle(cache_file_path)

        except FileNotFoundError:
            enriched_log = enrich_log_df_with_value_providers(log,
                                                              [RuntimeAttribute(attribute_name='External Risk Score',
                                                                                value_provider_activity='Get External Risk Score')])
            enriched_log.to_pickle(cache_file_path)

            return enriched_log


def enrich_log_for_ko_relocation_test(ko_analyzer: 'KnockoutAnalyzer',
                                      log: pd.DataFrame,
                                      cache_file_path: str = None) -> pd.DataFrame:
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
            enriched_log = enrich_log_df_with_value_providers(log,
                                                              [RuntimeAttribute(
                                                                  attribute_name='Aggregated Risk Score',
                                                                  value_provider_activity='Check Risk')])
            enriched_log.to_pickle(cache_file_path)

            return enriched_log
