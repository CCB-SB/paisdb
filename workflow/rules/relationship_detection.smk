rule proximity_based_relationship:
    input:
        csv = config['relationship_set']

    output:
        csvs = expand("../results/relationship_detection/proximity_{distance}.csv", distance=config['distance'].split(','))

    params:
        function = "proximity_based_relationship",
        distance=config['distance'],
        reference_row="Relationship",
        Relationship = True,

    conda:
        "../envs/test_env.yml"

    script:
        "../scripts/performtest.py"

rule existence_based_relationship:
    input:
        csv = config['relationship_set']

    output:
        csv = "../results/relationship_detection/existence.csv"

    params:
        function = "existence_based_relationship",
        reference_row="Relationship",
        Relationship = True,


    conda:
        "../envs/test_env.yml"

    script:
        "../scripts/performtest.py"

rule sentence_based_relationship:
    input:
        csv = config['relationship_set']

    output:
        csv = "../results/relationship_detection/sentence_co_occurrence.csv"

    params:
        function = "sentence_based_relationship",
        reference_row="Relationship",
        Relationship = True,
        nlp = True
        
    conda:
        "../envs/test_env.yml"

    script:
        "../scripts/performtest.py"

rule proximity_based_relationship_synonyms:
    input:
        csv = config['relationship_set']

    output:
        csvs = expand("../results/relationship_detection/proximity_{distance}_synonyms.csv", distance=config['distance'].split(','))

    params:
        function = "proximity_based_relationship",
        distance=config['distance'],
        reference_row="Relationship",
        Relationship = True,
        use_synonyms=True

    conda:
        "../envs/test_env.yml"

    script:
        "../scripts/performtest.py"

rule existence_based_relationship_synonyms:
    input:
        csv = config['relationship_set']

    output:
        csv = "../results/relationship_detection/existence_synonyms.csv"

    params:
        function = "existence_based_relationship",
        reference_row="Relationship",
        Relationship = True,
        use_synonyms=True,


    conda:
        "../envs/test_env.yml"

    script:
        "../scripts/performtest.py"

rule sentence_based_relationship_synonyms:
    input:
        csv = config['relationship_set']

    output:
        csv = "../results/relationship_detection/sentence_co_occurrence_synonyms.csv"

    params:
        function = "sentence_based_relationship",
        reference_row="Relationship",
        Relationship = True,
        nlp = True,
        use_synonyms=True,
        
    conda:
        "../envs/test_env.yml"

    script:
        "../scripts/performtest.py"
