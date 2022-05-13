import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import json


def word_counts(train):
    '''
    This function takes our train dataframe and returns a word_counts dataframe that tells us the frequency of the words 
    by each category of language. 
    '''
    # Gathering a list of the words by each language category
    java_words = (' '.join(train.lemmatized[train.language == 'Java'])).split()
    python_words = (' '.join(train.lemmatized[train.language == 'Python'])).split()
    c_words = (' '.join(train.lemmatized[train.language == 'C++'])).split()
    all_words = (' '.join(train.lemmatized)).split()
    # Creating a series of the frequency of the words inside the list for each category
    java_freq = pd.Series(java_words).value_counts()
    python_freq = pd.Series(python_words).value_counts()
    c_freq = pd.Series(c_words).value_counts()
    all_freq = pd.Series(all_words).value_counts()

    # Combining the frequencies into a word_counts df
    word_counts = (pd.concat([all_freq, java_freq, python_freq, c_freq], axis=1, sort=True)
              .set_axis(['all','java', 'python', 'c++'], axis=1, inplace=False)
              .fillna(0)
              .apply(lambda s: s.astype(int)))

    return word_counts
    