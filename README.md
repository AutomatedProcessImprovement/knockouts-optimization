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

## Usage
Check out [`main.py`](./main.py) for an end-to-end usage example.

## Running tests
A shell script is provided, which runs pytest in the relevant modules.

It can be launched with `./run tests.sh`

The flag `-n auto` is used by pytest-xdist to run tests in parallel.

## Notes:

### About wittgenstein
Excessive warnings are printed due to `frame.append` deprecation warnings.

Currently, they can just be suppressed by passing the `-Wignore` flag.

### About pm4py
A pull request has been submitted to pm4py about an inappropriate type-checking technique used in the `pm4py/filtering.py` module.

If you get an error about certain methods expecting an Event Log subclass, or a DataFrame, even though you are passing one, refer to this PR:

https://github.com/pm4py/pm4py-core/pull/323