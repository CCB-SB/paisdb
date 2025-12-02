rule pure_regex_strain:
    input:
        csv=config['test_set']

    output:
        csv="../results/strain/pure_regex.csv"

    params:
        function="pure_regex_strain",
        reference_row="strains",

    conda:
        "../envs/test_env.yml"

    script:
        "../scripts/performtest.py"

rule sentence_co_occurrence_strain:
    input:
        csv=config['test_set']

    output:
        csv="../results/strain/sentence_co_occurrence.csv"

    params:
        function="sentence_co_occurrence_strain",
        nlp=True,
        reference_row="strains",

    conda:
        "../envs/test_env.yml"

    script:
        "../scripts/performtest.py"

rule sentence_co_occurrence_strain_synonyms:
    input:
        csv=config['test_set']

    output:
        csv="../results/strain/sentence_co_occurrence_synonyms.csv"

    params:
        function="sentence_co_occurrence_strain",
        nlp=True,
        reference_row="strains",
        use_synonyms=True

    conda:
        "../envs/test_env.yml"

    script:
        "../scripts/performtest.py"

rule llm_strain:
    input:
        csv=config['test_set']

    output:
        csv=f"../results/strain/{config['llm_model']}.csv"

    params:
        function="create_query_batch_strain",
        llm = True,
        llm_model = config['llm_model'],
        reference_row="strains",

    resources:
        gpu = 1
        
    conda:
        "../envs/llm.yml"

    script:
        "../scripts/performtest.py"