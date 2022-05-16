import pandas as pd
import numpy as np
import nltk
import os

from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.preprocessing import StandardScaler, MinMaxScaler

import warnings
warnings.filterwarnings('ignore')


def model_words(vectorizer, class_model, ngrams_range_value, train, validate, target, print_results = True):
    """Performs classification modeling of lemmatized data. Outputs (and returns) classification reports for train and validate/test.
    
    vectorizer: the type of feature extraction method, such as Count Vectorizer or tf-idf
    class_model: the classification model to use
    ngrams_range_value: whether to use unigram, bigrams, etc. for the feature extraction
    train and test sets as well as the target variable"""
    
    # Instantiate the feature extraction method
    feature_extraction_method = vectorizer(ngram_range=ngrams_range_value)
    
    # Instantiate scaler
    scaler = StandardScaler()
    
    # Perform feature extraction on lemmatized text from train
    X_train = feature_extraction_method.fit_transform(train.lemmatized)
    
    # Generate dataframe of results of feature extraction
    train_vectorizer_df = pd.DataFrame(X_train.todense(), columns=feature_extraction_method.get_feature_names_out())
    
    # Set index to train index
    train_vectorizer_df.index = train.index
    
    # Add in other features (lengths of readme)
    X_train = pd.concat([train_vectorizer_df,train[['original_length','stem_length', 'lem_length', 'original_word_count',
           'stemmed_word_count', 'lemmatized_word_count']]], axis = 1)
    
    # Scale features in train (necessary for logistic regression)
    X_train_scaled = scaler.fit_transform(X_train)
    
    # Perform feature extraction and add in additional features on validate
    X_validate = feature_extraction_method.transform(validate.lemmatized)
    validate_vectorizer_df = pd.DataFrame(X_validate.todense(), columns=feature_extraction_method.get_feature_names_out())
    validate_vectorizer_df.index = validate.index

    X_validate = pd.concat([validate_vectorizer_df,validate[['original_length','stem_length', 'lem_length', 'original_word_count',
           'stemmed_word_count', 'lemmatized_word_count']]], axis = 1)
    
    # Scale validate
    X_validate_scaled = scaler.transform(X_validate)
    
    # Define target 
    y_train = train[target]
    y_validate = validate[target]
    
    # Create dataframe of results from train and validate
    train_results=pd.DataFrame(dict(actual = y_train))
    validate_results = pd.DataFrame(dict(actual = y_validate))
    
    # Check if Logistic regression model being used, which requires scaled data
    if ('Logistic' in class_model.__repr__()):
        
        # Fit model to scaled train data 
        model_to_use = class_model.fit(X_train_scaled, y_train)
        
        # Predict on train and validate
        train_results['predicted'] = model_to_use.predict(X_train_scaled)
        validate_results['predicted'] = model_to_use.predict(X_validate_scaled)
        
        # Feature names and importances not available for Logistic Regression
        feature_names = [np.nan]
        feature_importances = [np.nan]      
    else:
        # Tree based models do not need scaled data
        model_to_use = class_model.fit(X_train, y_train)

        train_results['predicted'] = model_to_use.predict(X_train)
        validate_results['predicted'] = model_to_use.predict(X_validate)
        
        # Multinormal Naive Bayes do not have feature names or importances available. Other algorithms do and will be saved.
        if ('MultinomialNB' not in class_model.__repr__()):
            feature_names = feature_extraction_method.get_feature_names_out()
            feature_importances = model_to_use.feature_importances_
        else:
            # Feature names and importances not available for Multinomial Naive Bayes
            feature_names = [np.nan]
            feature_importances = [np.nan]   
            
    # Generate classification reports for train and validate
    train_class_report = classification_report(train_results.actual, train_results.predicted, output_dict = True)
    validate_class_report = classification_report(validate_results.actual, validate_results.predicted,output_dict=True)
    
    # Print results from predictions
    if print_results:
        print('Accuracy: {:.2%}'.format(accuracy_score(train_results.actual, train_results.predicted)))
        print('---')
        # print('Train Confusion Matrix')
        # print(pd.crosstab(train_results_tfidf.predicted, train_results_tfidf.actual))
        print('---')
        print(pd.DataFrame(train_class_report))


        print('Accuracy: {:.2%}'.format(accuracy_score(validate_results.actual, validate_results.predicted)))
        print('---')
        # print('Validate Confusion Matrix')
        # print(pd.crosstab(validate_results_tfidf.predicted, validate_results_tfidf.actual))
        print('---')
        print(pd.DataFrame(validate_class_report))
    
    return train_class_report, validate_class_report, feature_names, feature_importances

def model_multiple(vectorizers, class_models, ngram_range_values, train, validate, target, print_results_param=False):
    """Performs classification modeling of inputed text data and returns Pandas DataFrame with the performance results. As this can be a time consuming process will read in previously generated results from csv if available.
    vectorizers: list of vectorizers such as tf-idf to use
    class_models: list classification models to use
    ngram_range_values: list of tuples indicating ngrams to use in vectorizer
    train, validate: datasets
    target: the target - in this case 'language'
    print_results_param: whether to print the classification report"""
    
    if os.path.exists('model_results.csv'):
        return pd.read_csv('model_results.csv', index_col='model')
    
    train_accuracies = []
    validate_accuracies = []
    dropoffs = []
    indices=[]
    model_parameters = []
    feature_names_list = []
    feature_importances_list = []
    total_models = len(vectorizers)*len(class_models)*len(ngram_range_values)
    i=0
    # Iterate through vectorizers, models, and ngram ranges 
    for v in vectorizers:
        for m in class_models:
            for ngram_range in ngram_range_values:
                # print("----")
                print(f"{(v.__name__, type(m).__name__, ngram_range)} - {i}/{total_models}", end="\r")
                indices.append((v.__name__, type(m).__name__, ngram_range))
                # Perform modeling
                train_class_report, validate_class_report, feature_names, feature_importances = model_words(vectorizer = v,
                                                                        class_model = m, 
                                                                        ngrams_range_value=ngram_range, 
                                                                        train = train, 
                                                                        validate = validate, 
                                                                        target = target, 
                                                                        print_results=print_results_param)
                # Append results of modeling to lists
                train_accuracies.append(train_class_report['accuracy'])
                validate_accuracies.append(validate_class_report['accuracy'])
                dropoffs.append(train_class_report['accuracy']-validate_class_report['accuracy'])
                model_parameters.append(m.get_params())
                feature_names_list.append(feature_names)
                feature_importances_list.append(feature_importances)
                i += 1
        # Generate dataframe of results from modeling
    results = pd.DataFrame(data = {'model_parameters': model_parameters,
                               'train_accuracy':train_accuracies,
                               'validate_accuracy':validate_accuracies,
                               'dropoff': dropoffs,
                              'feature_names':feature_names_list,
                              'feature_importances':feature_importances_list}, 
                       index=indices).sort_values(['dropoff', 'validate_accuracy'])
    
    # Function for getting the top features (if available)
    def get_top_features(row, n=6):
        """Outputs a list of the top n features used for the modeling. n = 6 by default"""
        if len(row.feature_names)<10:
            return np.NaN
        else:
            return pd.Series(dict(zip(row.feature_names, row.feature_importances))).sort_values().tail(n).index.tolist()
        
    # Generate new column in results dataframe with list of top n features    
    results['top_features'] = results.apply(lambda row: get_top_features(row, 6), axis=1)
    
    return results.sort_values(by=['validate_accuracy','dropoff'], ascending = [False, True])