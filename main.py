from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
import pandas as pd

df=pd.read_csv('./email_data.csv')
# print(df.describe())
# print(df.info())
# print(df["Category"][0])

vectorizer=CountVectorizer()


