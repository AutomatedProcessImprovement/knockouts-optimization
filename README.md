# knockouts-redesign

[![CI](https://github.com/AutomatedProcessImprovement/knockouts-redesign/actions/workflows/build-test.yml/badge.svg?branch=main)](https://github.com/AutomatedProcessImprovement/knockouts-redesign/actions/workflows/build-test.yml)

## About 

This is the implementation of a data-driven approach for discovering improvement opportunities related to the execution of knock-out checks within business processes. 

The tool comes with a browser-based UI (powered by [Streamlit](https://streamlit.io/)), but it can also be invoked via CLI or imported as a module in Python scripts.

<img src="https://user-images.githubusercontent.com/40581019/182252516-36b2fbe2-7d61-4502-b1f4-0c760d669d8f.PNG" width="100%">

## Resources
- [Download the full paper here](https://www.researchgate.net/publication/370973958_Discovery_of_Improvement_Opportunities_in_Knock-Out_Checks_of_Business_Processes)
- [Download the full thesis here](https://comserv.cs.ut.ee/ati_thesis/datasheet.php?id=75186&year=2022)
- [Event logs used for the paper experiments](https://github.com/AutomatedProcessImprovement/knockouts-redesign/tree/main/data/inputs)
- [Results of baseline experiments](https://zenodo.org/record/7615774)

## Index

1. [Set-up](https://github.com/AutomatedProcessImprovement/knockouts-redesign/tree/main#set-up)
2. [Usage](https://github.com/AutomatedProcessImprovement/knockouts-redesign/tree/main#usage)
3. [Running tests](https://github.com/AutomatedProcessImprovement/knockouts-redesign/tree/main#running-tests)
4. [Notes](https://github.com/AutomatedProcessImprovement/knockouts-redesign/tree/main#notes)

## Set-up

### Option A: Using a `setup` script

- Setup scripts are provided for Windows & Linux, which perform the steps described in the "Manual steps"
  section.
- Requirements:
    - Having cloned the repository
    - Python 3.9+ installed
- Commands:
  ```bash
  # Windows: 
  setup.bat
       
  # Linux: 
  bash setup.sh
   ```
### Option B: Manual steps 
  
<details>
<summary>Expand</summary>
  
- Clone the repo
- [Create a virtual environment with venv](https://packaging.python.org/en/latest/guides/installing-using-pip-and-virtual-environments/#creating-a-virtual-environment)
- In the directory that contains the newly created `venv` folder, run:
    ```
    source venv/Scripts/activate
    pip install -r requirements.txt
    ```
- Depending on your platform, install graphviz (v 3.0.0+) separately, and make sure it's in your `PATH`
  (see also: [graphviz downloads page](https://graphviz.org/download/#windows)):

     ```bash
      # Windows: 
      choco install graphviz
      
      # Linux: 
      sudo apt install graphviz
      
      # Mac:
      brew install graphviz
    ```

</details>

## Usage

Create a config file following the schema defined in [config_schema.json](https://github.com/AutomatedProcessImprovement/knockouts-redesign/blob/main/config/config_schema.json) (the directory also contains examples).
Then you can choose whether to:

- Launch the tool as a web app (powered by [Streamlit](https://streamlit.io/)):

  ```bash
     # Windows: 
     start.bat
     
     # Linux: 
     bash start.sh
  ```


- Launch from CLI:
  ```bash
     # Windows:
     python main.py config/synthetic_example.json
  
     # Linux: 
     python3 main.py config/synthetic_example.json
  ```

- Or programmatically:

  ```python
  
  from knockout_ios.pipeline_wrapper import Pipeline
  
  ko_redesign_adviser = Pipeline(config_file_name="synthetic_example.json").run_pipeline()

  ```
  In all these cases, if using the default settings, output will be written to `.csv` and `txt` files in
  the `data/outputs` folder.

## Running tests

Shell scripts are provided, which run the tests in parallel (thanks to pytest-xdist).

```bash
 # Windows: 
 test.bat
 
 # Linux: 
 bash test.sh
```


## Notes

#### About wittgenstein & grid search
<details>
<summary>Expand</summary>
  
- Excessive warnings are printed due to `frame.append` deprecation warnings in the `wittgenstein` package.

- Currently, they can just be suppressed by passing the `-Wignore` flag.

- Another situation where excessive warnings are printed, is during the parameter grid search, as some combinations may
  yield too small training sets or sets with no positive examples.

- This can be safely ignored as long as the grid contains at least one parameter combination that results in a valid
  dataset, and the rule discovery module is able to return meaningful rules.
  
</details>

