import pytest

from knockout_ios.knockout_analyzer import KnockoutAnalyzer
from knockout_ios.utils.constants import *
from knockout_ios.utils.synthetic_example.preprocessors import enrich_log_with_fully_known_attributes


@pytest.mark.parametrize("algorithm", ["RIPPER", "IREP"])
def test_report_creation(algorithm):
    analyzer = KnockoutAnalyzer(config_file_name="synthetic_example.json",
                                config_dir="./config",
                                cache_dir="./cache/synthetic_example",
                                always_force_recompute=True,
                                quiet=True,
                                custom_log_preprocessing_function=enrich_log_with_fully_known_attributes)

    analyzer.discover_knockouts()

    df, _ = analyzer.get_ko_rules(grid_search=False, algorithm=algorithm, omit_report=True, confidence_threshold=0.5,
                                  support_threshold=0.5)

    expected_kos = ['Check Liability', 'Check Risk', 'Check Monthly Income', 'Assess application']

    # assert all 4 knockouts are in the report
    assert df.shape[0] == len(expected_kos)
    assert sorted(analyzer.discoverer.ko_activities) == sorted(expected_kos)

    # assert all columns are in the report
    assert sorted([REPORT_COLUMN_WT_WASTE,
                   REPORT_COLUMN_TOTAL_PT_WASTE,
                   REPORT_COLUMN_TOTAL_OVERPROCESSING_WASTE,
                   REPORT_COLUMN_EFFORT_PER_KO,
                   REPORT_COLUMN_REJECTION_RATE,
                   REPORT_COLUMN_MEAN_PT,
                   REPORT_COLUMN_CASE_FREQ,
                   REPORT_COLUMN_TOTAL_FREQ,
                   REPORT_COLUMN_KNOCKOUT_CHECK,
                   f'{REPORT_COLUMN_REJECTION_RULE} ({algorithm})']) \
           == sorted(df.columns.tolist())

    # assert that there are no rows with NaN values
    assert df.isnull().sum().sum() == 0

    # assert that there are no rows with negative values
    assert df.loc[df['Effort per rejection'] < 0].shape[0] == 0

    # assert that there are no rows where 'Rejection rule' is empty
    assert df.loc[df[f'Rejection rule ({algorithm})'] == "[]", f'Rejection rule ({algorithm})'].shape[0] == 0
