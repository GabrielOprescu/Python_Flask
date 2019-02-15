import pandas as pd
import tensorflow as tf
import tensorflow_estimator as ts
import os
from sklearn.model_selection import train_test_split

os.chdir('C:\\Users\\gabriel_oprescu\\Desktop\\Projects\\Production_Test\\Python_Flask')

fraud_Tbl = pd.read_csv('fraud.csv')

feature_Tbl = fraud_Tbl.drop('Fraud', axis = 1)
label = fraud_Tbl['Fraud']

x_train, x_test, y_train, y_test = train_test_split(feature_Tbl, label, test_size = 0.9, random_state = 123) 

train_input_fn = tf.estimator.inputs.pandas_input_fn(x = x_train, y = y_train, batch_size = 32, shuffle = True, num_epochs = None)

test_input_fn = tf.estimator.inputs.pandas_input_fn(x = x_test, y = y_test, batch_size = 32, shuffle = True, num_epochs = 1)

A2 = tf.feature_column.numeric_column('A2')
A3 = tf.feature_column.numeric_column('A3')
A8 = tf.feature_column.numeric_column('A8')

estimator = tf.estimator.LinearClassifier(feature_columns = [A2, A3, A8])

estimator.train(input_fn = train_input_fn, steps = 100)

