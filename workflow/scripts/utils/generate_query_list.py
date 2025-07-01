import pandas as pd

# Read the CSV file into a DataFrame
df = pd.read_csv("/local/paisdb/src/final_test_dataset_12_03_2025_results.csv")

# Create an empty list to store the query strings
queries = []

# Loop through the relevant columns to generate the queries
for pathogen, disease, title, abstract in zip(
    df["pathogen_term"].to_list(),
    df["disease_term"].to_list(),
    df["title_process"].to_list(),
    df["abstract_process"].to_list()
):
    query_relationship = f"""I seek assistance with a systematic review focused on the direct relationship between pathogens and diseases, specifically {disease}. I’ll provide the title and abstract of a particular journal article and would appreciate an assessment for its inclusion based on the following criteria:

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
    queries.append(query_relationship)

# Add the queries to a new column in the DataFrame
df['query'] = queries

# Create a new DataFrame with only the 'pmid' and 'query' columns
result_df = df[['pmid', 'query']]

# Write the resulting DataFrame to a new CSV file
result_df.to_csv("/local/paisdb/src/queries.csv", index=False)
