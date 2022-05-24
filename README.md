# knockouts-redesign

This project is about discovering improvement opportunities in Knock-out checks performed within Business Processes.

## Set-up

- Clone the repo
- [Create a virtual environment with venv](https://packaging.python.org/en/latest/guides/installing-using-pip-and-virtual-environments/#creating-a-virtual-environment)
- In the directory that contains the newly created `venv` folder, run:
    ```
    source venv/Scripts/activate
    pip install -r requirements.txt
    ```
- Depending on your platform, install graphviz (v 3.0.0+) separately, and make sure it's in your `PATH`:
  [graphviz downloads page](https://graphviz.org/download/#windows).

## Usage

Create a config file following the schema defined in `config_schema.json` (see [`config_examples`](./config_examples)
directory), and then
simply run:

  ```python
from knockout_ios.pipeline_wrapper import run_pipeline

ko_redesign_adviser = run_pipeline(config_dir="config_examples",
                                   config_file_name="synthetic_example.json",
                                   cache_dir="cache/synthetic_example")
  ```

## Running tests

A shell script is provided, which runs pytest in the relevant modules.

On Mac/Linux or Windows with git bash, it can be launched with `bash ./test.sh`.

The flag `-n auto` is used by pytest-xdist to run tests in parallel.

## Notes

### About wittgenstein

Excessive warnings are printed due to `frame.append` deprecation warnings.

Currently, they can just be suppressed by passing the `-Wignore` flag.

Another situation where excessive warnings are printed, is during the parameter grid search, as some combinations may
yield too small training sets or sets with no positive examples.

This can be safely ignored as long as the grid contains at least one parameter combination that results in a valid
dataset, and the rule discovery module is able to return meaningful rules.
