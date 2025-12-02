rule pure_umls_anatomy:
    input:
        csv = config['test_set']

    output:
        csv = "../results/anatomy/pure_umls.csv"

    params:
        function = "pure_concept_extraction",
        quick_umls = config["quick_umls"],
        config_semtypes= config['semtypes_anatomy'],
        reference_row="anatomy_manual",

    conda:
        "../envs/test_env.yml"

    script:
        "../scripts/performtest.py"

rule pure_nlp_anatomy:
    input:
        csv = config['test_set']

    output:
        csv = "../results/anatomy/pure_nlp.csv"

    params:
        function = "pure_concept_nlp_extraction",
        config_semtypes= config['semtypes_anatomy'],
        reference_row="anatomy_manual",
        nlp=True,

    conda:
        "../envs/test_env.yml"

    script:
        "../scripts/performtest.py"


rule sentence_co_occurrence_anatomy_both:
    input:
        csv=config['test_set']

    output:
        csv = "../results/anatomy/sentence_co_occurrence_both.csv"

    params:
        function = "sentence_co_occurrence_concept_extraction",
        mode='and',
        config_semtypes= config['semtypes_anatomy'],
        reference_row="anatomy_manual",
        nlp=True,

    conda:
        "../envs/test_env.yml"

    script:
        "../scripts/performtest.py"

rule sentence_co_occurrence_anatomy_either:
    input:
        csv=config['test_set']

    output:
        csv = "../results/anatomy/sentence_co_occurrence_either.csv"

    params:
        function = "sentence_co_occurrence_concept_extraction",
        mode='or',
        config_semtypes= config['semtypes_anatomy'],
        reference_row="anatomy_manual",
        nlp=True,

    conda:
        "../envs/test_env.yml"

    script:
        "../scripts/performtest.py"

rule sentence_co_occurrence_anatomy_both_synonyms:
    input:
        csv=config['test_set']

    output:
        csv = "../results/anatomy/sentence_co_occurrence_both_synonyms.csv"

    params:
        function = "sentence_co_occurrence_concept_extraction",
        mode='and',
        config_semtypes= config['semtypes_anatomy'],
        reference_row="anatomy_manual",
        nlp=True,
        use_synonyms=True,

    conda:
        "../envs/test_env.yml"

    script:
        "../scripts/performtest.py"

rule sentence_co_occurrence_anatomy_either_synonyms:
    input:
        csv=config['test_set']

    output:
        csv = "../results/anatomy/sentence_co_occurrence_either_synonyms.csv"

    params:
        function = "sentence_co_occurrence_concept_extraction",
        mode='or',
        config_semtypes= config['semtypes_anatomy'],
        reference_row="anatomy_manual",
        nlp=True,
        use_synonyms=True,

    conda:
        "../envs/test_env.yml"

    script:
        "../scripts/performtest.py"

rule llm_anatomy:
    input:
        csv=config['test_set']

    output:
        csv = f"../results/anatomy/{config['llm_model']}.csv"

    params:
        function = "create_query_batch_extract_anatomy",
        llm = True,
        llm_model = config['llm_model'],
        reference_row="anatomy_manual",

    resources:
        gpu = 1
        
    conda:
        "../envs/llm.yml"

    script:
        "../scripts/performtest.py"