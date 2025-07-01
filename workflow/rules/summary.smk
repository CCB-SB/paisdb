rule summary_abstracts:
    input: csv = rules.pubmed_search.output.csv
    output:
        pdf = join(OUTDIR, "data_collection", "summary", "abstracts.pdf"),
    conda: "../envs/r_env.yml"
    log:
        log = join(OUTDIR, "data_collection", "summary", "abstracts.log")
    script:
        "../scripts/data_collection/summary/abstracts.R" 

rule summary_diseases:
    input: 
        csv = rules.pathogens_new.output.diseases_csv,
        do_terms = rules.disease_new.output.do_terms,
    output:
        png = join(OUTDIR, "data_collection", "summary", "diseases.png"),
    conda: "../envs/r_env.yml"
    script:
        "../scripts/data_collection/summary/diseases.R" 

rule summary_pathogens:
    input: 
        csv = rules.pathogens_new.output.pathogens_csv,
    output:
        png = join(OUTDIR, "data_collection", "summary", "pathogens.png"),
    conda: "../envs/r_env.yml"
    script:
        "../scripts/data_collection/summary/pathogens.R" 