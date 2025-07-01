rule llama3_1:
    input: csv = "../src/final_test_dataset_12_03_2025.csv"
    output: csv = join(OUTDIR, "/local/paisdb/results/relation_extraction/mini/", "llama3_1.csv")
    params:
        n_queries=1000,
        mode="zero-shot-prompt-3",
    threads: 10
    benchmark: join(OUTDIR, "/local/paisdb/results/relation_extraction/mini/", "llama3_1.bench")
    conda: "../envs/llm.yml"
    script:
        "../scripts/relation_extraction/mini/llama_3_1.py"

rule llama3:
    input: csv = "../src/final_test_dataset_12_03_2025.csv"
    output: csv = join(OUTDIR, "/local/paisdb/results/relation_extraction/mini/", "llama3.csv")
    params:
        n_queries=1000,
        mode="zero-shot-prompt-3",
    threads: 10
    benchmark: join(OUTDIR, "/local/paisdb/results/relation_extraction/mini/", "llama3.bench")
    conda: "../envs/llm.yml"
    script:
        "../scripts/relation_extraction/mini/llama3.py"

rule deepseek_llama:
    input: csv = "../src/final_test_dataset_12_03_2025.csv"
    output: csv = join(OUTDIR, "/local/paisdb/results/relation_extraction/mini/", "deepseek_llama.csv")
    params:
        n_queries=1000,
        mode="zero-shot-prompt-3",
    threads: 10
    benchmark: join(OUTDIR, "/local/paisdb/results/relation_extraction/mini/", "deepseek_llama.bench")
    conda: "../envs/llm.yml"
    script:
        "../scripts/relation_extraction/mini/deepseek_llama.py"

rule deepseek_qwen:
    input: csv = "../src/final_test_dataset_12_03_2025.csv"
    output: csv = join(OUTDIR, "/local/paisdb/results/relation_extraction/mini/", "deepseek_qwen.csv")
    params:
        n_queries=1000,
        mode="zero-shot-prompt-3",
    threads: 10
    benchmark: join(OUTDIR, "/local/paisdb/results/relation_extraction/mini/", "deepseek_qwen.bench")
    conda: "../envs/llm.yml"
    script:
        "../scripts/relation_extraction/mini/deepseek_qwen.py"

rule phi_4:
    input: csv = "../src/final_test_dataset_12_03_2025.csv"
    output: csv = join(OUTDIR, "/local/paisdb/results/relation_extraction/mini/", "phi_4.csv")
    params:
        n_queries=1000,
        mode="zero-shot-prompt-3",
    threads: 10
    benchmark: join(OUTDIR, "/local/paisdb/results/relation_extraction/mini/", "phi_4.bench")
    conda: "../envs/llm.yml"
    script:
        "../scripts/relation_extraction/mini/phi_4.py"

rule qwen:
    input: csv = "../src/final_test_dataset_12_03_2025.csv"
    output: csv = join(OUTDIR, "/local/paisdb/results/relation_extraction/mini/", "qwen.csv")
    params:
        n_queries=1000,
        mode="zero-shot-prompt-3",
    threads: 10
    benchmark: join(OUTDIR, "/local/paisdb/results/relation_extraction/mini/", "qwen.bench")
    conda: "../envs/llm.yml"
    script:
        "../scripts/relation_extraction/mini/qwen.py"

rule ministral:
    input: csv = "../src/final_test_dataset_12_03_2025.csv"
    output: csv = join(OUTDIR, "/local/paisdb/results/relation_extraction/mini/", "ministral.csv")
    params:
        n_queries=1000,
        mode="zero-shot-prompt-3",
    threads: 10
    benchmark: join(OUTDIR, "/local/paisdb/results/relation_extraction/mini/", "ministral.bench")
    conda: "../envs/llm.yml"
    script:
        "../scripts/relation_extraction/mini/ministral.py"

rule biomistral:
    input: csv = "../src/final_test_dataset_12_03_2025.csv"
    output: csv = join(OUTDIR, "/local/paisdb/results/relation_extraction/mini/", "biomistral.csv")
    params:
        n_queries=1000,
        mode="zero-shot-prompt-3",
    threads: 10
    benchmark: join(OUTDIR, "/local/paisdb/results/relation_extraction/mini/", "biomistral.bench")
    conda: "../envs/llm.yml"
    script:
        "../scripts/relation_extraction/mini/biomistral.py"

rule evaluation:
    input: csv="../src/final_test_dataset_12_03_2025_results.csv"
    output: csv=join(OUTDIR, "/local/paisdb/results/relation_extraction/mini/", "evaluation.csv")
    threads: 10
    benchmark: join(OUTDIR, "/local/paisdb/results/relation_extraction/mini/", "evaluation.bench")
    conda: "../envs/llm.yml"
    script:
        "../scripts/utils/utils_evaluation.py"

        
rule run_models:
    input:
        llama3_1_output = rules.llama3_1.output.csv,
        llama3_output = rules.llama3.output.csv,
        deepseek_llama_output = rules.deepseek_llama.output.csv,
        deepseek_qwen_output = rules.deepseek_qwen.output.csv,
        phi_4 = rules.phi_4.output.csv,
        qwen_output = rules.qwen.output.csv,
        ministral_new_output = rules.ministral.output.csv