# knockouts-optimization

This project is about discovering improvement opportunities in Knock-out checks performed within Business Processes.

## Set-up

Clone the repo, then run:

```
source venv/Scripts/activate
pip install -r requirements.txt
```

Depending on your platform, install graphviz (v 3.0.0+) separately, and make sure it's in your `PATH`:
[graphviz downloads page](https://graphviz.org/download/#windows).

## Usage example

(Same code is found in [`main.py`](./main.py)):

```python
from knockout_ios.knockout_analyzer import KnockoutAnalyzer

#  Synthetic Example Ground Truth
#  (K.O. checks and their rejection rules):
#
# 'Check Liability':        'Total Debt'     > 5000 ||  'Vehicle Owned' = "available_cases_before_ko/A"
# 'Check Risk':             'Loan Ammount'   > 10000
# 'Check Monthly Income':   'Monthly Income' < 1000
# 'Assess application':     'External Risk Score' < 0.3

analyzer = KnockoutAnalyzer(config_file_name="synthetic_example.json",
                            config_dir="config",
                            cache_dir="knockout_ios/cache/synthetic_example",
                            always_force_recompute=True)

analyzer.discover_knockouts()

analyzer.get_ko_rules(grid_search=True, algorithm="IREP", confidence_threshold=0.5, support_threshold=0.5)
```

## Running tests

A shell script is provided, which runs pytest in the relevant modules.

On Mac/Linux or Windows with git bash, it can be launched with `bash ./test.sh`.

The flag `-n auto` is used by pytest-xdist to run tests in parallel.

## Notes:

### About wittgenstein

Excessive warnings are printed due to `frame.append` deprecation warnings.

Currently, they can just be suppressed by passing the `-Wignore` flag.

Another situation where excessive warnings are printed, is during the parameter grid search, as some combinations may
yield too small training sets or sets with no positive examples.

This can be safely ignored as long as the grid contains at least one parameter combination that results in a valid
dataset, and the rule discovery module is able to return meaningful rules.

### About pm4py

A pull request has been submitted to pm4py about an inappropriate type-checking technique used in
the `pm4py/filtering.py` module.

If you get an error about certain methods expecting an Event Log subclass, or a DataFrame, even though you are passing
one, refer to this PR:

https://github.com/pm4py/pm4py-core/pull/323