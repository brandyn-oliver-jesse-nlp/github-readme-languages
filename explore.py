import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import json
import matplotlib as mpl


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


def common_words(word_counts):
    for column in word_counts.drop(columns='all'):
    # axis=1 in .apply means row by row
        (word_counts.sort_values(by='all', ascending=False)
        .head(20).apply(lambda row: row / row['all'], axis=1).drop(columns='all')
        .sort_values(by= column).plot.barh(stacked=True, width=1, ec='black'))
        plt.title(f'% of  the most common 20 {column} README words')
        plt.legend(bbox_to_anchor= (1.03,1))

        plt.gca().xaxis.set_major_formatter(mpl.ticker.FuncFormatter('{:.0%}'.format))
        plt.show()