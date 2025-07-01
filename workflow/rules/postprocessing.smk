rule full_article_retrieving:
    input:
        csv = lambda wildcards: expand(rules.mistral.output.csv, batch = wildcards.batch)
    output:
        csv = join(OUTDIR, "postprocessing", "retrieve_articles",f"retrieve_articles_{config['timestamp']}", "batch_{batch}.csv"),
        xml = join(OUTDIR, "postprocessing", "retrieve_articles",f"retrieve_articles_{config['timestamp']}", "batch_{batch}.xml")
    conda: 
        "../envs/postprocessing.yml"
    log: 
        log = join(OUTDIR, "postprocessing", "retrieve_articles",f"retrieve_articles_{config['timestamp']}", "batch_{batch}.log")
    params:
        batch_size = 300
        
    script:
       "../scripts/postprocessing/pmc_article_retrieving.py"


rule article_mining:
    input:
        csv = lambda wildcards: expand(rules.full_article_retrieving.output.csv, batch = wildcards.batch),
        xml = lambda wildcards: expand(rules.full_article_retrieving.output.xml, batch = wildcards.batch)
    output:
        csv = join(OUTDIR, "postprocessing", "text_mining",f"LLM_mining_{config['timestamp']}", "batch_{batch}.csv")
    conda: 
        "../envs/postprocessing.yml"
    resources: gpu_par=1
    log: 
        log = join(OUTDIR, "postprocessing", "text_mining",f"LLM_mining_{config['timestamp']}", "batch_{batch}.log")
    script:
        "../scripts/postprocessing/full_article_mining.py"

rule score_generation:
    input:
        csv = lambda wildcards: expand(rules.article_mining.output.csv, batch = wildcards.batch)
    output:
        csv = join(OUTDIR, "postprocessing", "scored_articles",f"score_{config['timestamp']}", "batch_{batch}.csv")
    conda:
        "../envs/postprocessing.yml"
    log: 
        log = join(OUTDIR, "postprocessing", "scored_articles",f"score_{config['timestamp']}", "batch_{batch}.log")
    script:
        "../scripts/postprocessing/score_generation.py"
