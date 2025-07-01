rule data_tables:
    input:
        diseases = rules.pathogens_new.output.diseases_csv,
        pathogens = rules.pathogens_new.output.pathogens_csv,
        articles = expand(join(OUTDIR, "filtering",f"abstracts_{config['timestamp']}",
             "batch_{batch}.csv"), batch=batches),
        relations = expand(join(OUTDIR, "relation_extraction", "mixtral",
            f"batches_{config['timestamp']}", "batch_{batch}.csv"), batch=batches),
    output:
        diseases = join(OUTDIR, "data_tables", "diseases_full.csv"),
        pathogens = join(OUTDIR, "data_tables", "pathogens_full.csv"),
        articles = join(OUTDIR, "data_tables", "articles.csv"),
        relations = join(OUTDIR, "data_tables", "relations.csv"),
        merged = join(OUTDIR, "data_tables", "merged_table.csv"),
    conda: "../envs/nlp.yml"
    log: join(OUTDIR, "data_tables", "data_tables.log")
    script:
        "../scripts/data_tables/tables.py"