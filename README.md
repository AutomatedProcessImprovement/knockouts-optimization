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
from knockout_ios.knockout_redesign_adviser import KnockoutRedesignAdviser
from knockout_ios.utils.synthetic_example.preprocessors import *

analyzer = KnockoutAnalyzer(config_file_name="synthetic_example_ko_order_io_advanced.json",
                            config_dir="config",
                            cache_dir="test/knockout_ios/cache/synthetic_example",
                            always_force_recompute=True,
                            quiet=True,
                            custom_log_preprocessing_function=enrich_log_for_ko_order_advanced_test)

analyzer.discover_knockouts()

analyzer.compute_ko_rules(algorithm="IREP", confidence_threshold=0.5, support_threshold=0.1,
                          print_rule_discovery_stats=False, omit_report=False,
                          max_rules=2, grid_search=True
                          )

adviser = KnockoutRedesignAdviser(analyzer)

adviser.compute_redesign_options()
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

A bug was found in the `pm4py/filtering.py` module and the maintainers were informed.

At the time of writing, the bugfix is not yet available in the pm4py version published on pypy.

If you get an error along the lines of "`Unbound local variable EventLog...`", refer
to [this commit](https://github.com/pm4py/pm4py-core/commit/65e1f1b0bbd0747fe81eb049780874608a395d6e) (done by a pm4py
maintainer).

