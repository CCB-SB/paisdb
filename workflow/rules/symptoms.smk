rule pure_umls_symptoms:
    input:
        csv=config['test_set']

    output:
        csv="../results/symptoms/pure_umls.csv"

    params:
        function="pure_concept_extraction",
        quick_umls = config["quick_umls"],
        config_semtypes= config['semtypes_symptoms'],
        reference_row="symptoms_manual",

    conda:
        "../envs/test_env.yml"

    script:
        "../scripts/performtest.py"

rule pure_nlp_symptoms:
    input:
        csv=config['test_set']

    output:
        csv="../results/symptoms/pure_nlp.csv"

    params:
        function="pure_concept_nlp_extraction",
        config_semtypes= config['semtypes_symptoms'],
        reference_row="symptoms_manual",
        nlp=True,

    conda:
        "../envs/test_env.yml"

    script:
        "../scripts/performtest.py"

rule sentence_co_occurrence_symptoms_both:
    input:
        csv=config['test_set']

    output:
        csv="../results/symptoms/sentence_co_occurrence_both.csv"

    params:
        function="sentence_co_occurrence_concept_extraction",
        nlp=True,
        mode='and',
        config_semtypes= config['semtypes_symptoms'],
        reference_row="symptoms_manual",

    conda:
        "../envs/test_env.yml"

    script:
        "../scripts/performtest.py"

rule sentence_co_occurrence_symptoms_either:
    input:
        csv=config['test_set']

    output:
        csv="../results/symptoms/sentence_co_occurrence_either.csv"

    params:
        function="sentence_co_occurrence_concept_extraction",
        nlp=True,
        mode='or',
        config_semtypes= config['semtypes_symptoms'],
        reference_row="symptoms_manual",

    conda:
        "../envs/test_env.yml"

    script:
        "../scripts/performtest.py"

rule sentence_co_occurrence_symptoms_both_synonyms:
    input:
        csv=config['test_set']

    output:
        csv="../results/symptoms/sentence_co_occurrence_both_synonyms.csv"

    params:
        function="sentence_co_occurrence_concept_extraction",
        nlp=True,
        mode='and',
        config_semtypes= config['semtypes_symptoms'],
        reference_row="symptoms_manual",
        use_synonyms=True,

    conda:
        "../envs/test_env.yml"

    script:
        "../scripts/performtest.py"

rule sentence_co_occurrence_symptoms_either_synonyms:
    input:
        csv=config['test_set']

    output:
        csv="../results/symptoms/sentence_co_occurrence_either_synonyms.csv"

    params:
        function="sentence_co_occurrence_concept_extraction",
        nlp=True,
        mode='or',
        config_semtypes= config['semtypes_symptoms'],
        reference_row="symptoms_manual",
        use_synonyms=True,

    conda:
        "../envs/test_env.yml"

    script:
        "../scripts/performtest.py"

rule llm_symptoms:
    input:
        csv=config['test_set']

    output:
        csv=f"../results/symptoms/{config['llm_model']}.csv"

    params:
        function="create_query_batch_extract_symptoms",
        llm = True,
        llm_model = config['llm_model'],
        reference_row="symptoms_manual",

    resources:
        gpu = 1

    conda:
        "../envs/llm.yml"
        

    script:
        "../scripts/performtest.py"