# Branch containing the testing for the information retrieval

## Overview

This is the code for the Bachelor thesis "Literature Preprocessing and Information Extraction of Pathogen-Disease Relations for the (Post Acute) Infection Syndrome Database (PA)ISDB" of Tom Wölker.
---

## Directory Layout

- `workflow/`
  - `rules/`: Snakemake rules for each pipeline stage.
  - `scripts/`: Python scripts for the evaluation of the approaches.
  - `envs/`: Conda environment files for reproducibility.
- `src/`: Source data files (test data set and other manual curated lists).
- `config/`: Configuration files (e.g., `config.yml`).
- `results/`: Output data and intermediate results.

---

## Running the Pipeline

1. **Configure the Pipeline**
   - Edit [`config/config.yml`](config/config.yml) set up the large language modules to test.

2. **Set Up Docker for LLMs**
   - follow the instructions in [`Dockerfile`](Dockerfile) and set up the docker.

3. **Large Language Module evaluation**
   - run the [`Snakefile`](workflow/Snakefile) for each model change the config.
   - use snakemake --cores 1 

4. **Non LLM evaluation**
   - change the mode in the snakefile to non-llm
   - use snakemake --cores <n> to run the evaluation for all other rules and for the plots

---

## Customization

- **Changing Models:** Update the configuration in [`config/config.yml`](config/config.yml) and if not downloaded to ['.cache'] the flag local_files_only in [`workflow/scripts/utils/LLM_based_extraction_zero_shot.py`](workflow/scripts/utils/LLM_based_extraction_zero_shot.py) to False.
- **Changing Prompts:** Modify the functions in [`workflow/scripts/utils/create_query_batches.py`](workflow/scripts/utils/create_query_batches.py).
- **Add algorithms:** define functions in the corresponding module script in 'workflow/scripts/utils/' the inputs to the function are text and config, with config containing the global variables in a dictionary format, which are loaded via the snakemake params and add them to the class ['workflow/scripts/test_manager.py'](workflow/scripts/test_manager.py). They then can be referenced by the key in the snakemake params function

---

## References

- For technical details on each rule, see the corresponding `.smk` files in [`workflow/rules/`](workflow/rules/).
- For technical details on each algorithm, see the corresponding function in the scripts in [`workflow/scripts/utils/LLM_based_extraction_zero_shot.py`](workflow/scripts/utils/LLM_based_extraction_zero_shot.py)

---

## Contact

For questions or contributions, please contact the maintainers listed in the repository.

## Licences
This work uses data from the UMLS® Metathesaurus® (version 2025AB). 
Use of the UMLS is governed by the UMLS Metathesaurus License.
Users must comply with the terms available at the National Library of Medicine:
https://uts.nlm.nih.gov/uts/assets/LicenseAgreement.pdf

UMLS® is a registered trademark of the U.S. National Library of Medicine.