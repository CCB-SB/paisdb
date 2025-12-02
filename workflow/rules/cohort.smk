
rule list_based_cohort_size:
    input:
        csv=config['test_set']

    output:
        csv="../results/cohort/manual_list.csv"

    params:
        function="list_based_cohort_size",
        cohort_names= config['cohort_names'],
        nlp=True,
        reference_row="cohort",
        result_type='int'

    conda:
        "../envs/test_env.yml"

    script:
        "../scripts/performtest.py"

rule semtypes_based_cohort_size:
    input:
        csv=config['test_set']

    output:
        csv="../results/cohort/semtypes.csv"

    params:
        function="semtypes_based_cohort_size",
        config_semtypes= config['semtypes_cohort'],
        nlp=True,
        reference_row="cohort",
        result_type='int'

    conda:
        "../envs/test_env.yml"

    script:
        "../scripts/performtest.py"

rule list_and_semtype_based_cohort_size:
    input:
        csv=config['test_set']

    output:
        csv="../results/cohort/manual_list_and_semtype.csv"

    params:
        function="list_and_semtype_based_cohort_size",
        config_semtypes= config['semtypes_cohort'],
        cohort_names= config['cohort_names'],
        nlp=True,
        reference_row="cohort",
        result_type='int'

    conda:
        "../envs/test_env.yml"

    script:
        "../scripts/performtest.py"

rule llm_cohort:
    input:
        csv=config['test_set']

    output:
        csv=f"../results/cohort/{config['llm_model']}.csv"

    params:
        function="create_query_batch_cohort",
        llm = True,
        llm_model = config['llm_model'],
        reference_row="cohort_manual",

    resources:
        gpu = 1

    conda:
        "../envs/llm.yml"

    script:
        "../scripts/performtest.py"