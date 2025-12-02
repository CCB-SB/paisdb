
rule anatomy_tabledata:
    input:
        files = expand("../results/anatomy/{sample}.csv", sample=["pure_umls","pure_nlp", "sentence_co_occurrence_both", "sentence_co_occurrence_either", "sentence_co_occurrence_both_synonyms", "sentence_co_occurrence_either_synonyms"] + config['llm_models'])

    output:
        txt = "../results/tabledata/anatomy.txt"

    conda:
        "../envs/test_env.yml"

    script:
        "../scripts/extraction_summary.py"

rule cohort_tabledata:
    input:
        files = expand("../results/cohort/{sample}.csv", sample=["manual_list", "semtypes", "manual_list_and_semtype"] + config['llm_models'])

    output:
        txt = "../results/tabledata/cohort.txt"

    conda:
        "../envs/test_env.yml"

    script:
        "../scripts/extraction_summary.py"

distances = config['distance'].split(',')
proximity_based_pathogen_validation = [f'proximity_{distance}' for distance in distances]
proximity_based_pathogen_validation_synonyms = [f'proximity_{distance}_synonyms' for distance in distances]
sample = ["sentence_co_occurrence", "existence", "sentence_co_occurrence_synonyms", "existence_synonyms"] + proximity_based_pathogen_validation + proximity_based_pathogen_validation_synonyms + config['llm_models']

rule pathogen_validation_tabledata:
    input:
        files = expand("../results/pathogen_validation/{sample}.csv", sample=sample)

    output:
        txt = "../results/tabledata/pathogen_validation.txt"

    conda:
        "../envs/test_env.yml"

    script:
        "../scripts/extraction_summary.py"

rule post_acute_tabledata:
    input:
        files = expand("../results/post_acute/{sample}.csv", sample=["list_based_post_acute", "list_based_co_occurrence"] + config['llm_models'])

    output:
        txt = "../results/tabledata/post_acute.txt"

    conda:
        "../envs/test_env.yml"

    script:
        "../scripts/extraction_summary.py"

distances = config['distance'].split(',')
proximity_based_detections = [f'proximity_{distance}' for distance in distances]
sample = ["existence", "sentence_co_occurrence", ] + proximity_based_detections

rule relationship_detection_tabledata:
    input:
        files = expand("../results/relationship_detection/{sample}.csv", sample=sample)

    output:
        txt = "../results/tabledata/relationship_detection.txt"

    conda:
        "../envs/test_env.yml"

    script:
        "../scripts/relationship_summary.py"

distances = config['distance'].split(',')
proximity_based_detections_synonyms = [f'proximity_{distance}_synonyms' for distance in distances]
sample = ["existence_synonyms", "sentence_co_occurrence_synonyms"] + proximity_based_detections_synonyms

rule relationship_detection_tabledata_synonyms:
    input:
        files = expand("../results/relationship_detection/{sample}.csv", sample=sample)

    output:
        txt = "../results/tabledata/relationship_detection_synonyms.txt"

    conda:
        "../envs/test_env.yml"

    script:
        "../scripts/relationship_summary.py"

rule strain_tabledata:
    input:
        files = expand("../results/strain/{sample}.csv", sample=["pure_regex", "sentence_co_occurrence", "sentence_co_occurrence_synonyms"] + config['llm_models'])

    output:
        txt = "../results/tabledata/strain.txt"

    conda:
        "../envs/test_env.yml"

    script:
        "../scripts/extraction_summary.py"

rule symptoms_tabledata:
    input:
       files = expand("../results/symptoms/{sample}.csv", sample=["pure_umls", "pure_nlp", "sentence_co_occurrence_both", "sentence_co_occurrence_either", "sentence_co_occurrence_both_synonyms", "sentence_co_occurrence_either_synonyms"] + config['llm_models'])

    output:
        txt = "../results/tabledata/symptoms.txt"

    conda:
        "../envs/test_env.yml"

    script:
        "../scripts/extraction_summary.py"

        