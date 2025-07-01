rule llama3_1:
    input: csv = "../src/final_test_dataset_12_03_2025.csv"
    output: csv = join(OUTDIR, "/local/paisdb/results/relation_extraction/final/zero/", "llama3_1.csv")
    params:
        n_queries=1000,
        mode="zero-shot-prompt-3",
    threads: 10
    benchmark: join(OUTDIR, "/local/paisdb/results/relation_extraction/final/zero/", "llama3_1.bench")
    conda: "../envs/llm.yml"
    script:
        "../scripts/relation_extraction/llama3_1.py"
rule qwen:
    input: csv = "../src/final_test_dataset_12_03_2025.csv"
    output: csv = join(OUTDIR, "/local/paisdb/results/relation_extraction/", "qwen.csv")
    params:
        n_queries=1000,
        mode="zero-shot-prompt-3",
    threads: 10
    benchmark: join(OUTDIR, "/local/paisdb/results/relation_extraction/", "qwen.bench")
    conda: "../envs/llm.yml"
    script:
        "../scripts/relation_extraction/qwen.py"

rule mistral:
    input: csv = lambda wildcards: expand(rules.filter_abstracts.output.csv, batch = wildcards.batch)
    output: csv = join(OUTDIR, "relation_extraction", "mixtral", f"mixtral_{config['timestamp']}", "mixtral_small_batch_{batch}.csv")
    params:
        mode="zero-shot-prompt-3",
        api_file = config["api_key_file"],
        hf_api = config["huggingface"]["api_key"],
    threads: 10
    resources: gpu_par=1
    benchmark: join(OUTDIR, "relation_extraction", "mixtral", f"mixtral_{config['timestamp']}", "mixtral_small_batch_{batch}.bench")
    conda: "../envs/llm.yml"
    script:
        "../scripts/relation_extraction/mixtral_small.py"

rule mixtral_large:
    input: csv = "../src/final_test_dataset_12_03_2025.csv"
    output: csv = join(OUTDIR, "/local/paisdb/results/relation_extraction/", "mixtral_large.csv")
    params:
        n_queries=1000,
        mode="zero-shot-prompt-3",
    threads: 10
    benchmark: join(OUTDIR, "/local/paisdb/results/relation_extraction/", "mixtral_large.bench")
    conda: "../envs/llm.yml"
    script:
        "../scripts/relation_extraction/mixtral_large.py"

rule DeepSeek_Llama:
    input: csv = "../src/final_test_dataset_12_03_2025.csv"
    output: csv = join(OUTDIR, "/local/paisdb/results/relation_extraction/", "DeepSeek_Llama.csv")
    params:
        n_queries=1000,
        mode="zero-shot-prompt-3",
    threads: 10
    benchmark: join(OUTDIR, "/local/paisdb/results/relation_extraction/", "DeepSeek_Llama.bench")
    conda: "../envs/llm.yml"
    script:
        "../scripts/relation_extraction/DeepSeek_Llama.py"

rule llama3_3:
    input: csv = "../src/final_test_dataset_12_03_2025.csv"
    output: csv = join(OUTDIR, "/local/paisdb/results/relation_extraction/", "llama3_3.csv")
    params:
        n_queries=1000,
        mode="zero-shot-prompt-3",
    threads: 10
    benchmark: join(OUTDIR, "/local/paisdb/results/relation_extraction/", "llama3_3.bench")
    conda: "../envs/llm.yml"
    script:
        "../scripts/relation_extraction/llama3_3.py"


rule Nemotron:
    input: csv = "../src/final_test_dataset_12_03_2025.csv"
    output: csv = join(OUTDIR, "/local/paisdb/results/relation_extraction/", "Nemotron.csv")
    params:
        n_queries=1000,
        mode="zero-shot-prompt-3",
    threads: 10
    benchmark: join(OUTDIR, "/local/paisdb/results/relation_extraction/", "Nemotron.bench")
    conda: "../envs/llm.yml"
    script:
        "../scripts/relation_extraction/Nemotron.py"

rule QwQ:
    input: csv = "../src/final_test_dataset_12_03_2025.csv"
    output: csv = join(OUTDIR, "/local/paisdb/results/relation_extraction/", "QwQ.csv")
    params:
        n_queries=1000,
        mode="zero-shot-prompt-3",
    threads: 10
    benchmark: join(OUTDIR, "/local/paisdb/results/relation_extraction/", "QwQ.bench")
    conda: "../envs/llm.yml"
    script:
        "../scripts/relation_extraction/QwQ.py"

rule openai:
    input: csv = "../src/final_test_dataset_12_03_2025.csv"
    output: csv = join(OUTDIR, "/local/paisdb/results/relation_extraction/", "gpt.csv")
    params:
        n_queries=1000,
        mode="zero-shot-prompt-3",
    threads: 10
    benchmark: join(OUTDIR, "/local/paisdb/results/relation_extraction/", "gpt.bench")
    conda: "../envs/llm.yml"
    script:
        "../scripts/relation_extraction/gpt.py"

rule DeepSeek_Qwen:
    input: csv = "../src/final_test_dataset_12_03_2025.csv"
    output: csv = join(OUTDIR, "/local/paisdb/results/relation_extraction/", "DeepSeek_Qwen.csv")
    params:
        n_queries=1000,
        mode="zero-shot-prompt-3",
    threads: 10
    benchmark: join(OUTDIR, "/local/paisdb/results/relation_extraction/", "DeepSeek_Qwen.bench")
    conda: "../envs/llm.yml"
    script:
        "../scripts/relation_extraction/DeepSeek_Qwen.py"

rule Phi:
    input: csv = "../src/final_test_dataset_12_03_2025.csv"
    output: csv = join(OUTDIR, "/local/paisdb/results/relation_extraction/", "Phi.csv")
    params:
        n_queries=1000,
        mode="zero-shot-prompt-3",
    threads: 10
    benchmark: join(OUTDIR, "/local/paisdb/results/relation_extraction/", "Phi.bench")
    conda: "../envs/llm.yml"
    script:
        "../scripts/relation_extraction/Phi.py"

rule Meditron:
    input: csv = "../src/final_test_dataset_12_03_2025.csv"
    output: csv = join(OUTDIR, "/local/paisdb/results/relation_extraction/", "MEDITRON.csv")
    params:
        n_queries=1000,
        mode="zero-shot-prompt-3",
    threads: 10
    benchmark: join(OUTDIR, "/local/paisdb/results/relation_extraction/", "MEDITRON.bench")
    conda: "../envs/llm.yml"
    script:
        "../scripts/relation_extraction/MEDITRON.py"

rule Palmyra:
    input: csv = "../src/final_test_dataset_12_03_2025.csv"
    output: csv = join(OUTDIR, "/local/paisdb/results/relation_extraction/", "Palmyra.csv")
    params:
        n_queries=1000,
        mode="zero-shot-prompt-3",
    threads: 10
    benchmark: join(OUTDIR, "/local/paisdb/results/relation_extraction/", "Palmyra.bench")
    conda: "../envs/llm.yml"
    script:
        "../scripts/relation_extraction/Palmyra.py"

rule PMC_LLaMA:
    input: csv = "../src/final_test_dataset_12_03_2025.csv"
    output: csv = join(OUTDIR, "/local/paisdb/results/relation_extraction/final/few/", "PMC_LLaMA_13B.csv")
    params:
        n_queries=1000,
        mode="zero-shot-prompt-3",
    threads: 10
    benchmark: join(OUTDIR, "/local/paisdb/results/relation_extraction/final/few/", "PMC_LLaMA_13B.bench")
    conda: "../envs/llm.yml"
    script:
        "../scripts/relation_extraction/PMC_LLaMA_13B.py"

rule evaluation:
    input: csv="../src/final_test_dataset_12_03_2025_results.csv"
    output: csv=join(OUTDIR, "/local/paisdb/results/relation_extraction/", "evaluation.csv")
    threads: 10
    benchmark: join(OUTDIR, "/local/paisdb/results/relation_extraction/", "evaluation.bench")
    conda: "../envs/llm.yml"
    script:
        "../scripts/utils/utils_evaluation.py"


rule run_models:
    input:
        llama3_1_output = rules.llama3_1.output.csv,
        qwen_output = rules.qwen.output.csv,
        mistral_output = rules.mistral.output.csv,
        llama3_3_output = rules.llama3_3.output.csv,
        Nemotron_output = rules.Nemotron.output.csv,
        DeepSeek_Llama_output = rules.DeepSeek_Llama.output.csv,
        #mixtral_large_output = rules.mixtral_large.output.csv,
        DeepSeek_Qwen_output = rules.DeepSeek_Qwen.output.csv,
        QwQ_output = rules.QwQ.output.csv,
        Phi_output = rules.Phi.output.csv
