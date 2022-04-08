import pandas as pd

from knockout_ios.utils.synthetic_example.attribute_enricher import enrich_log_df


def test_attribute_enricher():
    # Verify that conditions hold

    original = pd.read_pickle('./synthetic_example_raw_df.pkl')

    log_df = enrich_log_df(original)

    # verify random state is fixed
    assert log_df.equals(enrich_log_df(original))

    assert log_df[log_df['knockout_activity'] == "Check Monthly Income"]['Monthly Income'].min() <= 1000
    assert log_df[log_df['knockout_activity'] == "Check Risk"]['Loan Ammount'].min() >= 10000
    assert log_df[log_df['knockout_activity'] == "Assess application"]['External Risk Score'].min() >= 0.3

    subset_low_debt = log_df[(log_df['knockout_activity'] == "Check Liability") & (log_df['Total Debt'] <= 5000)]
    assert subset_low_debt['Vehicle Owned'].isna().all()

    subset_with_vehicle = log_df[
        (log_df['knockout_activity'] == "Check Liability") & (log_df['Vehicle Owned'].isna())]
    assert subset_with_vehicle[subset_with_vehicle['Total Debt'] >= 5000].shape[0] == 0
