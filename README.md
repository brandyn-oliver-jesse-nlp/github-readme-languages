# GITHUB README LANGUAGE CLASSIFCATION

### This repository contains code for the natural language processing project completed as part of the Codeup Data Science curriculum.
 
### Jesse Marder, Oliver Ton, Brandyn Waterman    05/17/2022

Table of Contents
---
 
* I. [Project Description](#i-project-description)<br>
* II. [Project Goals](#ii-project-goals)<br>
* III. [Business Goals](#iii-gusiness-goals)<br>
* IV. [Data Dictionary](#iv-data-dictionary)<br>
* V. [Data Science Pipeline](#v-project-planning)<br>
[1. Project Plan](#1-plan)<br>
[2. Data Acquisition](#2-acquire)<br>
[3. Data Preparation](#3-prepare)<br>
[4. Data Exploration](#4-explore)<br>
[5. Modeling & Evaluation](#5-model)<br>
[6. Product Delivery](#6-deliver)<br>
* V. [Project Reproduction](#vi-to-recreate)<br>
* VI. [Key Takeaway](#vii-takeaways)<br>
* VI. [Next Steps](#viii-next-steps)<br>

## I. Project Description

-------------------

<details><summary><i>Click to expand</i></summary>

We collected README files from 150 repositories on GitHub and predicted the primary programming language used in each project. We used the search term "machine learning" and sorted by highest starts to acquire quality repositories. 50 repositories from three languages - Java, C++, and Python - were acquired in order to analyze a balanced dataset.

 
This project involves textual data cleaning, wrangling and exploration, as well as modeling and validation/verification of modeling results. We leveraged natural language processing techniques to garner insights and to generate features for classification models.

Slides based on the findings documented in our final_notebook.ipynb are located [here.](https://docs.google.com/presentation/d/1b4gRp4zUtTnriJPJs-4TC4Izplj_DkErRSwq5HreIPY/edit?usp=sharing)

</details>
 

## II. Project Goals

-------------

<details><summary><i>Click to expand</i></summary>

1. Create scripts to perform the following:

                a. acquisition of data from GitHub's website

                b. preparation of data

                c. exploration of data

                d. modeling

2. Build and evaluate classification models to predict the programming language used in a given Readme.
</details>
 

## III. Business Goals

--------------

<details><summary><i>Click to expand</i></summary>

- Make use of NLP and classification models to predict programming language of a repository based on Readme content.

- Perform a number of parsing operations to isolate and process key text features - including lemmatization, stemming and removal of stopwords.

</details>
 


 

## IV. Data Dictionary

---------------

<details><summary><i>Click to expand</i></summary>

| Name |   Datatype   |      Definition    |    Possible Values  |
| :----- | :----- | :----- | :----- |
| repo  |     object  | Unique name for the repo |  slash delimited string|
| language |  object | The programming language used in this project | string (eg python/javascript/etc) |
| readme contents     |   object |  The entirety of the project's readme file | plain text string |

 

Additionally, a set of features were added to the data set:

 

| Name                  |Datatype      | Definition                                             | Possible Values    |
|:-----                 | :-----       |:------------------------------                         |:-----              |
| clean                 | object       | Parsing Text of the readme_content column              | plain text string  |
| stemmed               | object       | Stemmed text of the clean column                       | plain text string  |
| lemmatized            | object       | Lemmatized text of clean  column                       | plain text string  |
| original_length       | int64        | Lenght of the README content                           | numeric            |
| stem_length           | int64        | Lenght of the README content after stemmed applied     | numeric            |
| lem_length            | int64        | Lenght of the README content after lemmatized applied  | numeric            |
| original_word_count   | int64        | Total of words of the README                           | numeric            |
| stemmed_word_count    | int64        | Total of words of the README after stemmed applied     | numeric            |
| lemmatized_word_count | int64        | Total of words of the README after lemmatized applied  | numeric            |

</details>

 

## V. Project Planning

----------------

The overall process followed in this project is as follows:

 

###  Plan  -->  Acquire   --> Prepare  --> Explore  --> Model  --> Deliver


--------------

<details><summary><i>Click to expand</i></summary>

### 1. Plan


Perform preliminary examination of a number of GitHub projects.

Acquire tokens and permissions to scrape data from the GitHub website.

Prepare the data for exploration and modeling.

Develop questions to explore and generate features for modeling.

Perform machine learning modeling to predict the repository language.

Deliver results in the form of a final notebook, this README, and Google slide deck. 


### 2. Acquire

This is accomplished via the python script named “acquire.py”. The script will use credentials (stored in env.py) to collect data from GitHub.com in various ways

- First, collect a number of "URLs", or Repository names, so that the subsequent acquisition function will be able to seek out those repositories. Store the names of the repositories in a Python list. 

- Once the list of repositories is collected, use functions from the acquire script to collect the following information from those repositories, including:

                - repository name

                - actual language of the project

                - contents of the readme for that repository
                
- store these in data.json - that way, we would not hit GituHub's page and scrape the same data repeatedly. Moreover, this ensures that subsequent processing executions will consistently use the same repo list, leading to a more reliable and consistent result.

 

### 3. Prepare

This functionality is stored in the python script "prepare.py". It will perform the following actions:

- lowercase the readme contents to avoid case sensitivity

- remove non-standard (non ascii) characters, any accented characters

- tokenize the data

- applying stemming

- apply lemmatization

- remove unnecessary stopwords

- remove any records where the readme contents were null or empty

- generate additional features for exploration and modeling such as README length and word counts

- split the data into 3 datasets - train/test/validate - used in modeling

  - Train: 56% of the data

  - Validate: 24% of the data

  - Test: 20% of the data

 

### 4. Explore

Answer the following questions using data visualization and statistical testing:

1. What are the most common words in the README files by language?
2. Does the length of the README file vary by language?
3. Are bigrams from the README useful for determining which language the repository belongs to?


### 5. Model

Generate a baseline, against which all models will be evaluated. In this case we have an equal amount of each programming language in the dataset - 50 each of Python, Java, and C++ - so the baseline prediction is 33%.

Each model uses a different combination of vectorizer type (with different values for ngram range), algorithm, and hyperparameter set.

Compare the models against the baseline and each other based on the accuracy score from the validate sample. We sorted by ascending dropoff in accuracy from train to validate to guard against choosing an overfit model. 

Test the best performing model on witheld test data.


### 6. Deliver

Present findings via Google slides.
The slide deck can be found [here.](https://docs.google.com/presentation/d/1b4gRp4zUtTnriJPJs-4TC4Izplj_DkErRSwq5HreIPY/edit?usp=sharing)

</details>

### VI. To recreate

----------------

<details><summary><i>Click to expand</i></summary>

Simply clone the project locally and create an env.py file in the same folder as the cloned code. The env.py file's format should be as follows:

 
github_token = 'GITHUB-TOKEN'

github_username = 'GITHUB-USERNAME'


Next, run the acquire script in your command line, using the following command:

Download the acquire.py, explore.py, model.py and data.json in this repository.
```bash
curl -LO https://github.com/brandyn-oliver-jesse-nlp/github-readme-languages/blob/main/prepare.py
curl -LO https://github.com/brandyn-oliver-jesse-nlp/github-readme-languages/blob/main/acquire.py
curl -LO https://github.com/brandyn-oliver-jesse-nlp/github-readme-languages/blob/main/explore.py
```
Finally, open the Jupyter notebook titled final_notebook.ipynb and execute the code within.

Note: We used the PyGitHub library to acquire a list of repositories from Github to be input into the acquire module. In order to leverage this functionality ensure you have this library installed. Other libraries needed are pandas, numpy, nltk, scipy, sci-kit learn, and matplotlib.

</details>
 

### VII. Takeaways

-------------

<details><summary><i>Click to expand</i></summary>

- A repository's language can be predicted with accuracies much greater than the baseline's using natural language processing techniques and a classification model. 
- Of the 80 models tested the best performing one was the Decision Tree Classifier with a max depth setting of 2 and utilizing the count vectorizer (bag of words) with unigrams only. This model was able to predict the repository language with 60% accuracy, an improvement of 27% over the baseline.
- Overall only 22 of 80 models exhibited a dropoff in train to validate accuracy of less than 10%, indicating most models overfit to the train data. The only algorithm with decent performance without overfitting was the Decision Tree Classifier. Multinomial Naive Bayes didn't overfit but accuracy was around 33%. Performance with the Decision Tree Classifier was best with the Count Vectorizer and was identical whether solely unigrams were used vs with bigrams and/or trigrams. This result seems to indicate that unigrams alone are good enough as inputs to the classification model to distinguish between languages.
- There is a difference in the general length of the READMEs, and there is a statistical significance for two (Java and C++) of the languages compared to the overall mean. At a quick glance we can see that Python does have the largest word count on average, but this stems from having a number of READMEs that exceed the average word count generally. Python was still determined to not be statistically significant compared to the average readme length, likely due to its large standard deviation of word lengths. 
- The overall most common words tend to relate with installation of libraries or packages for the coding languages, and is heavily skewed by Python in terms of frequency. 
- There are a lot of unique lemmatized 'words' for each language, with Python having the highest amount in this capacity by quite a margin. (12,000 + compared to ~2500 for Java or C++)
 
 </details>

 

### VIII. Next Steps

-------------

<details><summary><i>Click to expand</i></summary>

- Test additional models on the data with different hyperparameters and algorithm types. Deep learning has been applied in this domain successfully and can provide a more flexible model.
- Leverage additional Natural Language Processing techniques for analyzing the text, such as topic modeling. We could look at words on a sentence level as well rather than the overall document.
- Including more languages, a larger number of READMEs and different categories of topics we could test our bag of words, unique words, and outcomes to see if the NLP model works on the niche set of parameters we utilized or has broader implications for the coding languages themselves. 
- Varying the ‘star’ ranking and comparing the predicted outcomes by language could give indication to what degree of unique language makes for a better overall README quality, and if there is a threshold where it becomes too cumbersome.  

</details>



