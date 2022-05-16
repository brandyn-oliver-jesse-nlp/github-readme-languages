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


def unique_counts(word_counts):
    '''
    This function takes our word_counts dataframe and finds the number of uniques to each language and returns them as their own dataframes. 
    '''
    # Adding a column for each language with 1 or 0 (str) to represent if it is unique to that language
    word_counts['unique_p'] = np.where(word_counts['all'] == word_counts['python'], '1', '0')
    word_counts['unique_j'] = np.where(word_counts['all'] == word_counts['java'], '1', '0')
    word_counts['unique_c'] = np.where(word_counts['all'] == word_counts['c++'], '1', '0')
    # Getting separate df's for these unique words
    unique_p = word_counts[['python']][word_counts.unique_p == '1']
    unique_j = word_counts[['java']][word_counts.unique_j == '1']
    unique_c = word_counts[['c++']][word_counts.unique_c == '1']
    # returning the three dataframes
    return unique_p, unique_j, unique_c


def word_cloud(train, language):
    '''
    This function takes in our dataframe and a string of which language you want to specify for. Outputs a WordCloud of the lemmatized text for that language.
    '''
    # Getting our specified language words
    words = (' '.join(train.lemmatized[train.language == language]))
    # Making our WordCloud image object
    img = WordCloud(background_color='white').generate(words)
    # Displaying the WordCloud image
    plt.figure(figsize=(10,5))
    plt.title(f'Word Cloud for {language}')
    plt.imshow(img)
    plt.axis('off')
    
def plot_bigrams(df, category, plot_bar = True, plot_wordcloud = False):
    """ Accepts word count dataframe and outputs plots of top 20 bigrams and wordcloud based on category.
    Returns top 20 category bigrams"""
    # Generate bigrams
    bigrams = list(nltk.ngrams(all_words_df.T[category]['all_words'].split(),2))
    # Take top 20
    top_20_cat_bigrams = pd.Series(bigrams).value_counts().head(20)
    
    if plot_bar:
        # Plot bar chart
        top_20_cat_bigrams.sort_values().plot.barh(color='orange', width=.9, figsize=(10, 6))

        # Ensure only integer values for x axis
        plt.xticks(range(top_20_cat_bigrams.sort_values().max()+1))

        plt.title(f'20 Most frequently occuring {category} bigrams')
        plt.ylabel('Bigram')
        plt.xlabel('# Occurrences')

        # make the labels pretty
        ticks, _ = plt.yticks()
        labels = top_20_cat_bigrams.reset_index()['index'].apply(lambda t: t[0] + ' ' + t[1])
        _ = plt.yticks(ticks, labels)
        plt.show()
    
    if plot_wordcloud:
        # Plot wordcloud
        data = {k[0] + ' ' + k[1]: v for k, v in top_20_cat_bigrams.to_dict().items()}
        img = WordCloud(background_color='white', width=800, height=400).generate_from_frequencies(data)
        plt.figure(figsize=(8, 4))
        plt.imshow(img)
        plt.axis('off')
        plt.title(category)
        plt.show()
        
    return top_20_cat_bigrams

    