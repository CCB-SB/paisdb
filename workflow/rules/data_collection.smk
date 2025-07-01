rule pathogens_gc:
    input: csv = config['pathogens']['gc']
    output: csv = join(OUTDIR, 'data_collection', 'pathogens', 'pathogens_gc.csv')
    conda: "../envs/nlp.yml"
    script:
        "../scripts/data_collection/pathogens/pathogens_gc.py" 

rule pathogens_wikipedia:
    input: csv = config['pathogens']['wikipedia']
    output:
        relations_csv = join(OUTDIR, "data_collection", "relations", "relations_wikipedia.csv"),
        pathogens_csv = join(OUTDIR, "data_collection", "pathogens", "pathos_wikipedia.csv") ,
        diseases_csv = join(OUTDIR, "data_collection", "diseases", "diseases_wikipedia.csv")
    conda: "../envs/nlp.yml"
    script:
        "../scripts/data_collection/pathogens/pathogens_wikipedia.py"

rule pathogens_pathodb:
    input: json = config['pathogens']['pathodb']
    output:
        pathogens_csv = join(OUTDIR, "data_collection", "pathogens", "pathos_pathodb.csv"),
        relations_csv = join(OUTDIR, "data_collection", "relations", "relations_pathodb.csv"),
    conda: "../envs/nlp.yml"
    script:
        "../scripts/data_collection/pathogens/pathogens_pathodb.py"

rule pathogens_disbiome:
    input: json = config['pathogens']['disbiome']
    output:
        pathogens_csv = join(OUTDIR, "data_collection", "pathogens", "pathos_disbiome.csv"),
        diseases_csv = join(OUTDIR, "data_collection", "diseases", "diseases_disbiome.csv"),
        relations_csv = join(OUTDIR, "data_collection", "relations", "relations_disbiome.csv"),
    conda: "../envs/nlp.yml"
    script:
        "../scripts/data_collection/pathogens/pathogens_disbiome.py"

# With the same format [disease,source]
rule diseases_merge:
    input:
        wikipedia = rules.pathogens_wikipedia.output.diseases_csv,
        disbiome = rules.pathogens_disbiome.output.diseases_csv,
        inhouse = config['diseases']['inhouse'],
    output:
        csv = join(OUTDIR, 'data_collection', 'diseases', 'diseases_merged.csv')
    shell:
        """
        cat {input.wikipedia} > {output.csv}
        sed '1 d' {input.disbiome} >> {output.csv}
        sed '1 d' {input.inhouse} >> {output.csv}
        """

rule diseases_new:
    input:
        new_entries = rules.diseases_merge.output.csv,
    output:
        relations_csv = join(OUTDIR, "data_collection", "relations", "relations_do.csv"),
        diseases_csv = join(OUTDIR, "data_collection", "diseases", "diseases_do.csv"),
        pathogens_csv = join(OUTDIR, "data_collection", "pathogens", "pathogens_do.csv"),
        unmatches = join(OUTDIR, "data_collection", "diseases", "unmatches_do.csv"),
        do_terms = join(OUTDIR, "data_collection", "diseases", "do_terms.csv")
    params:
        paisdb = config['paisdb']['diseases'],
        # Ontology
        disease_ont = config['disease_ontology']['DO'],
        # Thresholds
        threshold_wratio_do = config['disease_ontology']['threshold_wratio']
    log:
        join(OUTDIR, "data_collection", "diseases", "disease_new.log")
    conda: "../envs/disease_ont.yml"
    script:
        "../scripts/data_collection/diseases/diseases.py"

rule relations_merge:
    input:
        wiki = rules.pathogens_wikipedia.output.relations_csv,
        disbiome = rules.pathogens_disbiome.output.relations_csv,
        do = rules.diseases_new.output.relations_csv,
    output:
        csv = join(OUTDIR, 'data_collection', 'relations', 'relations_merged.csv')
    shell:
        """
        cat {input.wiki} > {output.csv}
        sed '1 d' {input.disbiome} >> {output.csv}
        sed '1 d' {input.do} >> {output.csv}
        """

rule pathogens_merge:
    input:
        gc = rules.pathogens_gc.output.csv,
        wiki = rules.pathogens_wikipedia.output.pathogens_csv,
        pathodb = rules.pathogens_pathodb.output.pathogens_csv,
        disbiome = rules.pathogens_disbiome.output.pathogens_csv,
        do = rules.diseases_new.output.pathogens_csv,
        inhouse = config['pathogens']['inhouse'],
    output:
        csv = join(OUTDIR, 'data_collection', 'pathogens', 'pathogens_merged.csv')
    shell:
        """
        cat {input.gc} > {output.csv}
        sed '1 d' {input.wiki} >> {output.csv}
        sed '1 d' {input.pathodb} >> {output.csv}
        sed '1 d' {input.disbiome} >> {output.csv}
        sed '1 d' {input.do} >> {output.csv}
        sed '1 d' {input.inhouse} >> {output.csv}
        """

rule pathogens_new:
    input:
        new_entries = rules.pathogens_merge.output.csv,
        relations = rules.relations_merge.output.csv,
        diseases = rules.diseases_new.output.diseases_csv,
        do_terms = rules.diseases_new.output.do_terms,
    output:
        pathogens_csv = join(OUTDIR, "data_collection", "pathogens", "pathogens_full.csv"),
        diseases_csv = join(OUTDIR, "data_collection", "diseases", "diseases_full.csv"),
        relations_csv = join(OUTDIR, 'data_collection', 'relations', 'relations_full.csv'),
    params:
        paisdb = config['paisdb']['pathogens'],
        # Thresholds
        threshold_wratio_ = config['ncbi_taxonomy']['threshold_wratio']
    log:
        join(OUTDIR, "data_collection", "pathogens", "pathogens_new.log")
    conda: "../envs/ete4.yml"
    script:
        "../scripts/data_collection/pathogens/pathogens_new.py"

rule test_relations:
    conda: 
        "../envs/ete4.yml"
    script:
        "../tests/data_collection/relations/test_relation_filtering.py"

#########################################################
# ABSTRACTS
#########################################################
rule abstracts_processing:
    input:
        pathogens = rules.pathogens_new.output.pathogens_csv,
        diseases = rules.pathogens_new.output.diseases_csv,
        relations = rules.pathogens_new.output.relations_csv,
    params:
        date_collection_range = config['date_collection_range'],
        batch_size = 10000
    output: 
        DIR = directory(join(OUTDIR, "data_collection", "abstracts", 
            f"queries_{config['timestamp']}"))
    conda: "../envs/ncbi.yml"
    log: 
        join(OUTDIR, "data_collection", "abstracts",
            f"queries_{config['timestamp']}/abstracts_processing.log")
    benchmark: 
        join(OUTDIR, "data_collection", "abstracts",
            f"queries_{config['timestamp']}/abstracts_processing.bench")
    script:
        "../scripts/data_collection/abstracts/abstracts_processing.py"

ruleorder: abstracts_processing > abstracts_pubmed 
rule abstracts_pubmed:
    input: pickle = join(rules.abstracts_processing.output.DIR, "batch_{batch}.pickle")
    params:
        api_file = config["api_key_file"],
        ncbi_api = config["eutils"]["api_key"],
        date_collection_range = config['date_collection_range'],
    output:
        pickle = join(OUTDIR, "data_collection","abstracts",
            f"pubmed_{config['timestamp']}", "batch_{batch}.pickle")
    threads: 10 # Maximum to prevent too ncbi error (many requests)
    resources: api_req=config["api_req"]
    conda: "../envs/ncbi.yml"
    log:
        join(OUTDIR, "data_collection", "abstracts", 
            f"pubmed_{config['timestamp']}", "batch_{batch}.log")
    benchmark: 
        join(OUTDIR, "data_collection", "abstracts", 
            f"pubmed_{config['timestamp']}", "batch_{batch}.bench")
    script:
        "../scripts/data_collection/abstracts/abstracts_pubmed.py"

rule abstracts_extraction:
    input: pickle = lambda wildcards: expand(rules.abstracts_pubmed.output.pickle, batch=wildcards.batch)
    output: 
        csv = join(OUTDIR, "data_collection", "abstracts",
            f"extraction_{config['timestamp']}", "batch_{batch}.csv")
    conda: "../envs/ncbi.yml"
    log:  
        join(OUTDIR, "data_collection", "abstracts",
            f"extraction_{config['timestamp']}", "batch_{batch}.log")
    benchmark: 
        join(OUTDIR, "data_collection", "abstracts", 
            f"extraction_{config['timestamp']}", "batch_{batch}.bench")
    script:
        "../scripts/data_collection/abstracts/abstracts_extraction.py"
