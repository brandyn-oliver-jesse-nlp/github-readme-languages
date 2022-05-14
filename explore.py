import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import json
import matplotlib as mpl
import nltk
from wordcloud import WordCloud


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
    '''
    This function take in a variable from word_count function and create a stack bar plot 
    to see which word is common in the README for each category of langue
    '''
    # axis=1 in .apply means row by row
    word_counts.sort_values(by='all', ascending=False).head(30)\
    .apply(lambda row: row / row['all'], axis=1).sort_values(by='all')\
    .drop(columns='all')\
    .plot.barh(stacked=True, width=1, ec='black')

    plt.title(f'% of the most common 30 words')
    plt.legend(bbox_to_anchor= (1.03,1))

    plt.gca().xaxis.set_major_formatter(mpl.ticker.FuncFormatter('{:.0%}'.format))
    plt.show()


def word_cloud (text):
    '''
    takes in a text and create a wordcloud
    '''
    img = WordCloud(background_color='white', width=800, height=600).generate(text)
    plt.imshow(img)
    plt.axis('off')
    plt.show()


def wordcloud_top(df,column, n_top=50):
    '''
    takes in a df , column and a number of top words to show
    '''
    top_all =df.sort_values(column, ascending=False)[[column]].head(n_top)
    word_cloud(' '.join(top_all.index))


def ngrams_wordcloud (text, title,  n=2, top = 20):
    '''
    takes in a text, title, number of ngrams, and number of the top words
    returns a plot barh and a word_cloud
    '''
    #plot barh
    
    plt.subplot(2,2,1)
    pd.Series(nltk.ngrams(text.split(), n=n)).value_counts().head(top).sort_values(ascending = True).plot.barh()
    plt.title(f'Top {top} most common {title} ngrams where n={n}')
    
    #word_cloud
    ng =(pd.Series(nltk.ngrams(text.split(), n=n)).value_counts().head(top)).to_dict()
    ng_words = {k[0] + ' ' + k[1]: v for k, v in ng.items()}
    plt.subplot(2,2,2)
    img = WordCloud(background_color='white', width=800, height=600).generate_from_frequencies(ng_words)
    plt.imshow(img)
    plt.axis('off')
    plt.title(f'Top {top} most common {title} ngrams where n={n}')
    #plt.tight_layout()
    plt.show()