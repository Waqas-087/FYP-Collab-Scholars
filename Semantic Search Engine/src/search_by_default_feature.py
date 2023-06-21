from rank_bm25 import BM25Okapi

import nltk
import pickle
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk import FreqDist

from gensim.parsing.preprocessing import STOPWORDS

from spellchecker import SpellChecker

def clean_query(query):
    '''
    Function to perform lemmatization and cleaning on query
    '''
    lemmed = [WordNetLemmatizer().lemmatize(word) for word in word_tokenize(query) if word not in STOPWORDS]
    lemmed = [WordNetLemmatizer().lemmatize(word, pos='v') for word in lemmed]
    lemmed = list(set(lemmed))

    # applying spell checker on tags
    spell = SpellChecker()
    misspelled = spell.unknown(lemmed)
    if len(misspelled) == 0:
        return lemmed
    else:
        correct_words = list(set(lemmed) - misspelled)
        correction = []

        for word in misspelled:
            # Get the one `most likely` answer
            correction.append(spell.correction(word))
        new_query = query
        for i in range(len(correction)):
            new_query = new_query.replace(list(misspelled)[i], correction[i])

        # cleaned auto_tags
        lemmed = correct_words + correction
        print(f"Searching for {new_query} instead of {query}")
        return lemmed

if __name__ == '__main__':
    data = pickle.load(open(r"E:\Work\Bahria University\FYP\Intelligent_Document_Finder-master\DataBase/data_file.pkl", "rb"))
    titles = pickle.load(open(r"E:\Work\Bahria University\FYP\Intelligent_Document_Finder-master\DataBase/title_file.pkl", "rb"))

    option = eval(input("Enter option of search"))

    if option == 'default search':
        corpus = pickle.load(open(r"E:\Work\Bahria University\FYP\Intelligent_Document_Finder-master\DataBase/corpus_file.pkl", "rb"))
    elif option == 'tag search':
        corpus = pickle.load(open(r"E:\Work\Bahria University\FYP\Intelligent_Document_Finder-master\DataBase/tags_pickle.pkl", "rb"))
    elif option == 'title search':
        corpus = pickle.load(open(r"E:\Work\Bahria University\FYP\Intelligent_Document_Finder-master\DataBase/title_corpus.pkl", "rb"))
    else:
        print('Not valid option')

    bm25 = BM25Okapi(corpus)
    query = eval(input("Enter query"))
    tokenized_query = clean_query(query.lower())

    indexes, results = bm25.get_top_n(tokenized_query, data, n=20)
    results_titles = []
    for i in indexes:
        results_titles.append(titles[i])

    for i in range(len(results)):
        print(f"Title_{i}: {results_titles[i]}")
        print(f"\nText_{i}: {results[i]}")
        relevance_score = bm25.get_scores(tokenized_query)
        print(f"\nRelevance Score_{i}: {relevance_score}")
        print('\n\n')