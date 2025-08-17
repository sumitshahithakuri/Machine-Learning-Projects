# importing required libraries for the model
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.naive_bayes import MultinomialNB
import pandas as pd

# loading dataSet and viewing it
df=pd.read_csv('./email_data.csv')
# print(df.describe())
# print(df.info())
# print(df["Category"][0])

# preparing for model training
vectorizer=CountVectorizer()
x=vectorizer.fit_transform(df["Message"])
# print(x.shape)
X_train,X_test,Y_train,y_test=train_test_split(x,df['Category'],test_size=0.2,random_state=23)
# print(X_train.shape)

# selecting model
model=MultinomialNB()

# training model
model.fit(X_train,Y_train)

# testing on test data
pred=model.predict(X_test)

# checking accuracy score of the trained model
accuracy=accuracy_score(y_test,pred)

print("Accuracy: ",accuracy)


# checking what model does with our own custom email

my_email="""
Hello Mr. John you are invited to the interview tomorrow on 12 June. Thank You!
"""

#vectorizing
my_vect=vectorizer.transform([my_email])

res=model.predict(my_vect)
print("Result: ",res)