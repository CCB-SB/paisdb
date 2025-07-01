import os
import time
import pandas as pd
from tqdm import tqdm
import openai
import re

# Ensure your API key is set
openai.api_key = 'insert_token_here'


def openai_omni_query(pubmed_abstracts, batch_size=10, model_id="o1"):
    batch_size = 1
    def create_query_batch(df):
        queries = []
        for pathogen, disease, title, abstract in zip(
            df["pathogen_term"].to_list(),
            df["disease_term"].to_list(),
            df["title_process"].to_list(),
            df["abstract_process"].to_list()
        ):
            query = f"""I seek assistance with a systematic review focused on the direct relationship between pathogens and diseases, specifically {disease}. I’ll provide the title and abstract of a particular journal article and would appreciate an assessment for its inclusion based on the following criteria:

1. The title or abstract provides sufficient evidence of a direct relationship between the disease ({disease}) and the pathogen ({pathogen}).
2. The title or abstract investigates the Pathogen ({pathogen}) and reports evidence for the Disease ({disease}).
3. The title or abstract investigates the Disease ({disease}) and reports evidence for the Pathogen ({pathogen}).
4. The title or abstract states the association between the Pathogen ({pathogen}) and the Disease ({disease}), but does not focus on it.
5. The title and abstract present data or findings supporting this association.

Exclusion criteria:
1. The title and abstract do not provide sufficient evidence of a direct relationship between the disease ({disease}) and the pathogen ({pathogen}).

Please provide the assessment in the following dictionary format:
{{"relationship": 1, "unrelated": 0}} if there is a relationship, or {{"relationship": 0, "unrelated": 1}} if the study should be excluded.

Note: only one value can be 1 at a time.

Title: {title}

Abstract: {abstract}

You are required to classify a journal article based solely on the given title and abstract. Do not use any external knowledge or assumptions beyond the text provided. Your decision must be strictly based on the information within the title and abstract.

Respond only in the dictionary format with no explanation.

Answer:"""
            queries.append(query)
        return queries

    result_dict = {
        f"query_{model_id}": [],
        f"answer_{model_id}": [],
        "time_taken": []
    }

    for i in tqdm(range(0, len(pubmed_abstracts), batch_size)):
        batch = pubmed_abstracts.iloc[i:i+batch_size]
        queries = create_query_batch(batch)
        
        start_time = time.time()
        for query in queries:
            try:
                response = openai.chat.completions.create(
                    model=model_id,
                    messages=[
                        {"role": "user", "content": query}
                    ],
                    temperature=0.0,
                    max_tokens=30
                )
                answer_text = response.choices[0].message.content.strip()
                match = re.findall(r'\{.*?\}', answer_text)
                if match:
                    try:
                        parsed = eval(match[-1])
                    except:
                        parsed = answer_text
                else:
                    parsed = answer_text

                result_dict[f"query_{model_id}"].append(query)
                result_dict[f"answer_{model_id}"].append(parsed)
                result_dict["time_taken"].append(time.time() - start_time)
                break
            except Exception as e:
                print("Error:", e)
                result_dict[f"query_{model_id}"].append(query)
                result_dict[f"answer_{model_id}"].append("ERROR")
                result_dict["time_taken"].append(0)
                break

    result_df = pd.DataFrame(result_dict)

    try:
        output_dir = "/local/paisdb/results/relation_extraction/tmp"
        os.makedirs(output_dir, exist_ok=True)
        output_path = os.path.join(output_dir, f"results_{model_id.replace('-', '_')}.csv")
        result_df.to_csv(output_path, index=False)
    except Exception as e:
        print("Saving failed:", e)

    return result_df
