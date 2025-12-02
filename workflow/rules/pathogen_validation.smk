rule existence_based_pathogen_validation:
    input:
        csv=config['test_set']

    output:
        csv= "../results/pathogen_validation/existence.csv"
    
    params:
        function="existence_based_pathogen_validation",
        methods= config['methods'],
        reference_row="validation",

    conda:
        "../envs/test_env.yml"

    script:
        "../scripts/performtest.py"

rule proximity_based_pathogen_validation:
    input:
        csv=config['test_set']

    output:
        csvs = expand("../results/pathogen_validation/proximity_{distance}.csv", distance=config['distance'].split(','))

    params:
        function="proximity_based_pathogen_validation",
        methods= config['methods'],
        distance=config['distance'],
        reference_row="validation",

    conda:
        "../envs/test_env.yml"

    script:
        "../scripts/performtest.py"

rule sentence_co_occurrence_pathogen_validation:
    input:
        csv=config['test_set']

    output:
        csv="../results/pathogen_validation/sentence_co_occurrence.csv"

    params:
        function="sentence_co_occurrence_pathogen_validation",
        methods= config['methods'],
        reference_row="validation",
        nlp=True,

    conda:
        "../envs/test_env.yml"

    script:
        "../scripts/performtest.py"

rule existence_based_pathogen_validation_synonyms:
    input:
        csv=config['test_set']

    output:
        csv= "../results/pathogen_validation/existence_synonyms.csv"
    
    params:
        function="existence_based_pathogen_validation",
        methods= config['methods'],
        reference_row="validation",
        use_synonyms=True,

    conda:
        "../envs/test_env.yml"

    script:
        "../scripts/performtest.py"

rule proximity_based_pathogen_validation_synonyms:
    input:
        csv=config['test_set']

    output:
        csvs = expand("../results/pathogen_validation/proximity_{distance}_synonyms.csv", distance=config['distance'].split(','))

    params:
        function="proximity_based_pathogen_validation",
        methods= config['methods'],
        distance=config['distance'],
        reference_row="validation",
        use_synonyms=True,

    conda:
        "../envs/test_env.yml"

    script:
        "../scripts/performtest.py"

rule sentence_co_occurrence_pathogen_validation_synonyms:
    input:
        csv=config['test_set']

    output:
        csv="../results/pathogen_validation/sentence_co_occurrence_synonyms.csv"

    params:
        function="sentence_co_occurrence_pathogen_validation",
        methods= config['methods'],
        reference_row="validation",
        nlp=True,
        use_synonyms=True

    conda:
        "../envs/test_env.yml"

    script:
        "../scripts/performtest.py"

rule llm_pathogen_detection:
    input:
        csv=config['test_set']

    output:
        csv=f"../results/pathogen_validation/{config['llm_model']}.csv"

    params:
        function="create_query_batch_pathogen_validation",
        llm = True,
        llm_model = config['llm_model'],
        reference_row="validation",

    resources:
        gpu = 1

    conda:
        "../envs/llm.yml"

    script:
        "../scripts/performtest.py"