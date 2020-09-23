import pandas as pd
from sklearn.metrics import f1_score, accuracy_score
import numpy as np
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV




class Base:
    """Base class that houses common utilities for reading in test data
    and calculating model accuracy and F1 scores.
    """
    def __init__(self) -> None:
        pass

    def read_data(self, fname: str, lower_case: bool=False,
                  colnames=['truth', 'text']) -> pd.DataFrame:
        "Read in test data into a Pandas DataFrame"
        df = pd.read_csv(fname, sep='\t', header=None, names=colnames)
        df['truth'] = df['truth'].str.replace('__label__', '')
        # Categorical data type for truth labels
        df['truth'] = df['truth'].astype(int).astype('category')
        # Optional lowercase for test data (if model was trained on lowercased text)
        if lower_case:
            df['text'] = df['text'].str.lower()
        return df

    def accuracy(self, df: pd.DataFrame) -> None:
        "Prediction accuracy (percentage) and F1 score"
        acc = accuracy_score(df['truth'], df['pred'])*100
        f1 = f1_score(df['truth'], df['pred'], average='macro')*100
        print("Accuracy: {:.3f}\nMacro F1-score: {:.3f}".format(acc, f1))





class VaderSentiment(Base):
    """Predict sentiment scores using Vader.
    Tested using nltk.sentiment.vader and Python 3.6+
    https://www.nltk.org/_modules/nltk/sentiment/vader.html
    """
    def __init__(self, model_file: str=None) -> None:
        super().__init__()
        import nltk
        #pip install nltk
        #nltk.download('vader_lexicon')
        from nltk.sentiment.vader import SentimentIntensityAnalyzer
        self.vader = SentimentIntensityAnalyzer()

    def score(self, text: str) -> float:
        return self.vader.polarity_scores(text)['compound']

    def predict(self, train_file: None, test_file: str, lower_case: bool) -> pd.DataFrame:
        df = self.read_data(test_file, lower_case)
        df['score'] = df['text'].apply(self.score)
        # Convert float score to category based on binning
        df['pred'] = pd.cut(df['score'],
                            bins=5,
                            labels=[1, 2, 3, 4, 5])
        df = df.drop('score', axis=1)
        #df.to_csv('vader_outcome.csv')
        return df




class SVMSentiment(Base):
    """Predict sentiment scores using a linear Support Vector Machine (SVM).
    Uses a sklearn pipeline.
    """
    def __init__(self, model_file: str=None) -> None:
        super().__init__()
        # pip install sklearn
        from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer, TfidfVectorizer
        from sklearn.linear_model import SGDClassifier
        from sklearn.svm import SVC, LinearSVC
        from sklearn.pipeline import Pipeline


        self.pipeline = Pipeline(
            [
                #('vect', CountVectorizer()),
               # ('tfidf', TfidfTransformer()),
                ('tfidf', TfidfVectorizer()),
                ('clf', LinearSVC( loss='hinge'


                 #SVC(C=1, kernel = 'linear', tol= 0.01

                 #SGDClassifier(
                  #  loss='hinge',
                  #  penalty='l2',
                 # alpha=1e-3,
                 #   random_state=42,
                 #max_iter=10,
                  #  learning_rate = 'optimal',
                  #  tol=None



                )),
            ]
        )



    def predict(self, train_file: str, test_file: str, lower_case: bool) -> pd.DataFrame:
        "Train model using sklearn pipeline"
        from sklearn.model_selection import GridSearchCV
        from sklearn.svm import SVC
        from sklearn.linear_model import SGDClassifier
        from sklearn import svm
        from sklearn import preprocessing
        from sklearn.preprocessing import LabelEncoder, OneHotEncoder
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.svm import LinearSVC
        import eli5

        train_df = self.read_data(train_file, lower_case)
        parameters = {
            #'clf__alpha': [0.0001, 0.001, 0.01, 1, 10, 100],
            #'clf__max_iter': [10, 100, 1000, 10000],
            #'clf__tol': [0, 0.0001, 0.001, 0.01],
            #'clf__loss':['hinge'],
            #'clf__penalty': ['l2'],
            #'clf__random_state': [0, 15, 42],
            #'clf__learning_rate': ['optimal']
            #'clf__C': [0.001, 0.01, 0.1, 1, 10, 100]
            #'clf__kernel': ['linear']

        }

        #lr = LinearSVC()
        #print(lr.get_params().keys())
        gs_clf = GridSearchCV(self.pipeline, parameters, cv=5, n_jobs=-1)
        gs_clf = gs_clf.fit(train_df['text'], train_df['truth'])
        print(gs_clf.best_score_)

        for param_name in sorted(parameters.keys()):
            print("%s: %r" % (param_name, gs_clf.best_params_[param_name]))



        learner = self.pipeline.fit(train_df['text'], train_df['truth'])


        # Fit the learner to the test data
        test_df = self.read_data(test_file, lower_case)

        test_df['pred'] = learner.predict(test_df['text'])
        #test_df.to_csv('svm_outcome.csv')

        return test_df










