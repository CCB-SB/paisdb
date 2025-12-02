import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import string

# ensure result path is available
if not os.path.exists("../results/images/datasets/testset_rel"):
    os.makedirs("../results/images/datasets/testset_rel") 

def generate_counts_strings(dataset):
    # generate counts string length for text field
    string_lengths = dataset['text'].apply(lambda x : len(x.translate(str.maketrans('', '', string.punctuation))))
    string_lengths.plot(kind='hist', bins=20)
    average = np.mean(string_lengths)
    plt.xlabel('string length of text')
    plt.ylabel('Occurrences')
    plt.savefig(f'../results/images/datasets/testset_rel/string_length_occurrences.png', dpi=300, bbox_inches='tight')
    plt.close()  
    print(f"Average string count: {average}")

# generate count for word length
def count_words(text, nlp):
    doc = nlp(text)
    return len([token for token in doc if not token.is_punct])

def generate_word_lengths(dataset, nlp):
    word_lengths = dataset['text'].apply(lambda x : count_words(x, nlp))
    word_lengths.plot(kind='hist', bins=20)
    average = np.mean(word_lengths)
    plt.savefig(f'../results/images/datasets/testset_rel/word_length_occurrences.png', dpi=300, bbox_inches='tight')
    plt.xlabel('number of words in text')
    plt.ylabel('Occurrences')
    plt.close()
    print(f"Average word count: {average}")

# average sentence length
def average_sent_length(text, nlp):
    doc = nlp(text)
    return np.mean([len(sent.text) for sent in doc.sents])

def generate_average_string_length(dataset, nlp):
    average_sent_length = dataset['text'].apply(lambda x : average_sent_length(x, nlp))
    average_sent_length.plot(kind='hist', bins=20)
    plt.savefig(f'../results/images/datasets/testset_rel/average_sent_length.png', dpi=300, bbox_inches='tight')
    plt.xlabel('average string sentence length of the text')
    plt.ylabel('Occurrences')
    plt.close()
    print(f"Average string sentence length: {np.mean(average_sent_length)}")

def generate_count_pathogens(dataset):
    # generate count for pathogen occurences
    pathogen_counts = dataset.apply(lambda x: x['text'].lower().count(x['pathogen_term'].lower()), axis=1)
    average = np.mean(pathogen_counts)
    pathogen_counts = pathogen_counts.value_counts().sort_index()
    pathogen_counts.plot(kind='bar')

    plt.savefig(f'../results/images/datasets/testset_rel/pathogen_occurrences.png', dpi=300, bbox_inches='tight')
    plt.xlabel('Mentions of the pathogen')
    plt.ylabel('Occurrences')
    plt.close()
    print(f"Average Pathogen count: {average}")

def generate_count_diseases(dataset):
    # generate count for disease occurences
    disease_counts = dataset.apply(lambda x: x['text'].lower().count(x['disease_term'].lower()), axis=1)
    average = np.mean(disease_counts)
    disease_counts = disease_counts.value_counts().sort_index()
    disease_counts.plot(kind='bar')

    plt.savefig(f'../results/images/datasets/testset_rel/disease_occurrences.png', dpi=300, bbox_inches='tight')
    plt.xlabel('Mentions of the disease')
    plt.ylabel('Occurrences')
    plt.close()
    print(f"Average Disease count: {average}")

def generate_count_pathogens_and_synonyms(dataset):
    # generate count for pathogen occurences with synonyms
    pathogen_counts = dataset.apply(lambda x: sum(x['text'].lower().count(pathogen.lower().strip()) for pathogen in x['pathogen_term'].split('|')), axis=1)
    average = np.mean(pathogen_counts)
    pathogen_counts = pathogen_counts.value_counts().sort_index()
    pathogen_counts.plot(kind='bar')

    plt.savefig(f'../results/images/datasets/testset_rel/pathogen_occurrences.png', dpi=300, bbox_inches='tight')
    plt.xlabel('Mentions of the pathogen or synonym')
    plt.ylabel('Occurrences')
    plt.close()
    print(f"Average Pathogen count: {average}")

def generate_count_diseases_and_synonyms(dataset):
    # generate count for disease occurences with synonyms
    disease_counts = dataset.apply(lambda x: sum(x['text'].lower().count(disease.lower().strip()) for disease in x['disease_term'].split('|')), axis=1)
    average = np.mean(disease_counts)
    disease_counts = disease_counts.value_counts().sort_index()
    disease_counts.plot(kind='bar')

    plt.savefig(f'..//results/images/datasets/testset_rel/disease_occurrences.png', dpi=300, bbox_inches='tight')
    plt.xlabel('Mentions of the disease or synonym')
    plt.ylabel('Occurrences')
    plt.close()
    print(f"Average Disease count: {average}")

import spacy
nlp = spacy.load("en_core_sci_md")

dataset = pd.read_csv("../source/relationship_detection_source.csv")
dataset.replace({np.nan:None}, inplace=True)

# enable which functions to produce
# generate_counts_strings(dataset)
# generate_word_lengths(dataset, nlp)
# generate_average_string_length(dataset, nlp)
# generate_count_pathogens(dataset)
# generate_count_diseases(dataset)
# generate_count_pathogens_and_synonyms(dataset)
#generate_count_diseases_and_synonyms(dataset)