
rule anatomy_plotting:
    input:
        files = expand("../results/anatomy/{sample}.csv", sample=["pure_umls","pure_nlp", "sentence_co_occurrence_both", "sentence_co_occurrence_either", "sentence_co_occurrence_both_synonyms", "sentence_co_occurrence_either_synonyms"] + config['llm_models']),
    output:
        png="../results/plots/anatomy_by_set.png"

    params:
        name = "target organ"

    conda:
        "../envs/test_env.yml"

    script:
        "../scripts/plotting_by_set.py"


rule cohort_plotting:
    input:
        files = expand("../results/cohort/{sample}.csv", sample=["manual_list", "semtypes", "manual_list_and_semtype"] + config['llm_models']),

    output:
        png="../results/plots/cohort_by_set.png"

    params:
        name = "cohort size"

    conda:
        "../envs/test_env.yml"

    script:
        "../scripts/plotting_by_set.py"


distances = config['distance'].split(',')
proximity_based_pathogen_validation = [f'proximity_{distance}' for distance in distances]
proximity_based_pathogen_validation_synonyms = [f'proximity_{distance}_synonyms' for distance in distances]
sample = ["sentence_co_occurrence", "existence", "sentence_co_occurrence_synonyms", "existence_synonyms"] + proximity_based_pathogen_validation + proximity_based_pathogen_validation_synonyms + config['llm_models']

rule pathogen_validation_plotting:
    input:
        files = expand("../results/pathogen_validation/{sample}.csv", sample=sample),

    output:
        png="../results/plots/pathogen_validation_by_set.png"

    params:
        name = "pathogen validation methode"

    conda:
        "../envs/test_env.yml"

    script:
        "../scripts/plotting_by_set.py"
    

rule post_acute_plotting:
    input:
        files = expand("../results/post_acute/{sample}.csv", sample=["list_based_post_acute", "list_based_co_occurrence"] + config['llm_models'] ),
    output:
        png="../results/plots/post_acute_by_set.png"

    params:
        name = "post acute evidence"

    conda:
        "../envs/test_env.yml"

    script:
        "../scripts/plotting_by_set.py"


distances = config['distance'].split(',')
proximity_based_detections = [f'proximity_{distance}' for distance in distances]
proximity_based_detections_synonyms = [f'proximity_{distance}' for distance in distances]
sample = ["existence", "sentence_co_occurrence"] + proximity_based_detections


rule relationship_detection_plotting:
    input:
        files = expand("../results/relationship_detection/{sample}.csv", sample=sample),

    output:
        png="../results/plots/relationship_detection_by_set.png"

    params:
        name = "relationship detection"

    conda:
        "../envs/test_env.yml"

    script:
        "../scripts/plotting_by_set_relationship.py"


distances = config['distance'].split(',')
proximity_based_detections_synonyms = [f'proximity_{distance}' for distance in distances]
sample = ["existence_synonyms", "sentence_co_occurrence_synonyms"] + proximity_based_detections_synonyms

rule relationship_detection_plotting_synonyms:
    input:
        files = expand("../results/relationship_detection/{sample}.csv", sample=sample),

    output:
        png="../results/plots/relationship_detection_synonyms_by_set.png"

    params:
        name = "relationship detection with synonyms"

    conda:
        "../envs/test_env.yml"

    script:
        "../scripts/plotting_by_set_relationship.py"



rule strain_plotting:
    input:
        files = expand("../results/strain/{sample}.csv", sample=["pure_regex", "sentence_co_occurrence", "sentence_co_occurrence_synonyms"] + config['llm_models']),

    output:
        png="../results/plots/strain_by_set.png"

    params:
        name = "strain identifier"

    conda:
        "../envs/test_env.yml"

    script:
        "../scripts/plotting_by_set.py"

rule symptoms_plotting:
    input:
       files = expand("../results/symptoms/{sample}.csv", sample=["pure_umls", "pure_nlp", "sentence_co_occurrence_both", "sentence_co_occurrence_either", "sentence_co_occurrence_both_synonyms", "sentence_co_occurrence_either_synonyms"] + config['llm_models']),

    output:
        png="../results/plots/symptoms_by_set.png"

    params:
       name = "symptom"

    conda:
        "../envs/test_env.yml"

    script:
        "../scripts/plotting_by_set.py"


        