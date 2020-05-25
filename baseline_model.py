# Satisfaction scores range from 0 to 10. I have the data of the patients' demographic information and their comment left on healthcare record.
# The classification problem would then be based on features extracted from numerical and textual data.

# Import necessary packages

import numpy as np
import pandas as pd

from sklearn.preprocessing import FunctionTransformer

import nltk
nltk.download('vader_lexicon')
from nltk.sentiment.vader import SentimentIntensityAnalyzer

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer

from sklearn.model_selection import train_test_split
from sklearn.feature_selection import chi2, SelectKBest


from sklearn.pipeline import Pipeline
from sklearn.pipeline import FeatureUnion

from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import SGDClassifier
from sklearn.ensemble import RandomForestClassifier

from sklearn import metrics

# Import training dataset

train = pd.read_csv('training_data.csv', delimiter='\t')
train.head(3)

# Generate sentiment polarity as a feature of the text data:

sid = SentimentIntensityAnalyzer()
train['sentiment'] = train['comment'].apply(lambda text: sid.polarity_scores(text))
train['sentiment'].apply(pd.Series) #Because polarity scores are numbers (negative, positive, and compound scores), I use them as numerical features.

# Generate dummy variables from patients' demographic information:

numeric = ['sex', 'age_cat', 'stay_cat', 'lang', 'category']
training = pd.get_dummies(train[numeric])
training1 = pd.concat([train, training, df], axis=1)

# Separate numerical and textual features before classification:

numerical_data = ['sex_1.0', 'sex_2.0', 'age_cat_1.0', 'age_cat_2.0',
                  'age_cat_3.0', 'age_cat_4.0', 'age_cat_5.0', 'age_cat_6.0',
                  'stay_cat_1.0', 'stay_cat_2.0', 'stay_cat_3.0',
                  'stay_cat_4.0', 'stay_cat_5.0', 'lang_1.0', 'lang_2.0',
                  'lang_3.0', 'lang_4.0', 'category_admn', 'category_disch',
                  'category_issues', 'category_meals', 'category_nurses',
                  'category_overall', 'category_physn', 'category_room', 'category_tests',
                  'category_visit','pos', 'neg']
                  
get_text_data = FunctionTransformer(lambda x: x['comment'], validate=False)
get_numeric_data = FunctionTransformer(lambda x: x[numerical_data], validate=False)
just_text_data = get_text_data.fit_transform(training1)
just_numeric_data = get_numeric_data.fit_transform(training1)

# Assign labels:

y = training1.score

# Split training data into train/test sets:

X_train, X_test, y_train, y_test = train_test_split( training1[['sex_1.0', 'sex_2.0', 'age_cat_1.0', 'age_cat_2.0',
                                                               'age_cat_3.0', 'age_cat_4.0', 'age_cat_5.0', 'age_cat_6.0',
                                                               'stay_cat_1.0', 'stay_cat_2.0', 'stay_cat_3.0',
                                                               'stay_cat_4.0', 'stay_cat_5.0', 'lang_1.0', 'lang_2.0',
                                                               'lang_3.0', 'lang_4.0', 'category_admn', 'category_disch',
                                                               'category_issues', 'category_meals', 'category_nurses',
                                                               'category_overall', 'category_physn', 'category_room', 'category_tests',
                                                               'category_visit', 'pos', 'neg', 'comment']], 
                                                               
                                                               y, 
                                                               
                                                               test_size=0.177501, random_state=22)
                                                               
# Create pipeline for feature extraction from numeric and text data:

process_and_join_features = FeatureUnion(transformer_list = [('numeric_features', Pipeline([('selector', get_numeric_data)])), 
                                                             ('text_features', Pipeline([('selector', get_text_data),
                                                                                         ('vectorizer', TfidfVectorizer( *maybe token pattern also* stop_words='english', ngram_range=(1,2)))]))
                                                            ])
                                                            
# Fitting the training data to classification model:

pl = Pipeline([('union', process_and_join_features),('clf', MultinomialNB(alpha=0))])
pl.fit(X_train, y_train)

# Checking model's performance:

pred = pl.predict(X_test)
score = sum((pred-y_test)**2)
print(score)

# Searching the best alpha for MultinomailNB classification model

alphas = np.arange(0, 1, .1)

def train_and_predict(alpha):
    pl = Pipeline([('union', process_and_join_features),('clf', MultinomialNB(alpha=alpha))])
    pl.fit(X_train, y_train)
    pred = pl.predict(X_test)
    score = sum((pred-y_test)**2)
    return score

for alpha in alphas:
    print('Alpha: ', alpha)
    print('Score: ', train_and_predict(alpha))
    print()
