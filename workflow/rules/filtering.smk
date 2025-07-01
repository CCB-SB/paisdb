rule filter_abstracts:
    input: 
        csv = lambda wildcards: expand(rules.abstracts_extraction.output.csv, batch = wildcards.batch),
        disease = rules.pathogens_new.output.diseases_csv,
        pathogen = rules.pathogens_new.output.pathogens_csv
    output: csv = join(OUTDIR, "filtering", f"abstracts_{config['timestamp']}", "batch_{batch}.csv")
    params:
        date_collection_range = config['date_collection_range'],
    conda: "../envs/ete4.yml" # "../envs/nlp.yml"
    log: log = join(OUTDIR, "filtering", f"abstracts_{config['timestamp']}", "batch_{batch}.log")
    script:
        "../scripts/filtering/filter_abstracts.py"

rule filter_synonym_test:
    conda: "../envs/ete4.yml"
    script: "../tests/filtering/synonyms/with_duplicates.py"

rule filter_synonym_dubs:
    conda: "../envs/ete4.yml"
    script: "../tests/filtering/synonyms/no_duplicates.py"