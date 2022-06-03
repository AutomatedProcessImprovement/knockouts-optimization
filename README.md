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
- Depending on your platform, install graphviz (v 3.0.0+) separately, and make sure it's in your `PATH`
  (see also: [graphviz downloads page](https://graphviz.org/download/#windows)):
    - Windows: `choco install graphviz`
    - Linux: `sudo apt install graphviz`
    - Mac: `brew install graphviz`

## Usage

Create a config file following the schema defined in `config/config_schema.json` (the directory also contains examples).
Then you can choose whether to:

- Launch the tool as a web app (powered by [Streamlit](https://streamlit.io/)):

  ```bash
  bash ./start.sh
  ```

- Or programmatically:

  ```python
  
  from knockout_ios.pipeline_wrapper import Pipeline
  
  ko_redesign_adviser = Pipeline(config_file_name="synthetic_example.json").run_pipeline()

  ```
  In both cases, if using the default settings, output will be written to `.csv` and `txt` files in
  the `data/outputs` folder.

## Running tests

A shell script is provided, which runs the tests in parallel (thanks to pytest-xdist).

On Mac/Linux or Windows with git bash:

```bash
bash ./test.sh
```

---

### Notes

#### About wittgenstein & grid search

- Excessive warnings are printed due to `frame.append` deprecation warnings in the `wittgenstein` package.

- Currently, they can just be suppressed by passing the `-Wignore` flag.

- Another situation where excessive warnings are printed, is during the parameter grid search, as some combinations may
  yield too small training sets or sets with no positive examples.

- This can be safely ignored as long as the grid contains at least one parameter combination that results in a valid
  dataset, and the rule discovery module is able to return meaningful rules.
