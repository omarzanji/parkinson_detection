'''
Parkinson's detector using a public dataset containing speech data from 
patients and healthy participants. The model will be trained using 
Gradient Boosting via the open source XGBoost package.

authors:
Omar Barazanji
-sign here-

date: 10/19/2021

dataset and references:
1) https://www.kaggle.com/vikasukani/parkinsons-disease-data-set
2) https://www.kaggle.com/vikasukani/detecting-parkinson-s-disease-machine-learning
3) https://stackoverflow.com/questions/35572000/how-can-i-plot-a-confusion-matrix
'''

# Import helper modules 
import os
import numpy as np
import pandas as pd
import seaborn as sn
import matplotlib.pyplot as plt

# Import sklearn dependencies
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix

# Import Extreme Gradient Boosting Model
from xgboost import XGBClassifier


class Model:
    '''
    Handles the entire setup and execution of the model. This includes grabbing training data, 
    pre-processing it, and training the model. 
    '''

    def __init__(self):
        '''
        Constructor grabs the data if it exists and exits if not found.
            note: link to data in header in references. 
        '''
        try:
            self.data = pd.read_csv('data/parkinsons.data')
        except:
            print('parkinsons.data not found')
            exit(1)

    def pre_process(self):
        '''
        Pre-processes the grabbed data and splits into training / testing arrays. 
        '''
        # feature vector
        self.x = self.data.loc[:, self.data.columns != 'status'].values[:, 1:]

        # label vector (status 1 or 0 for parkinsons or not)
        self.y = self.data.loc[:, 'status'].values

        scaler = MinMaxScaler((-1, 1))
        self.x = scaler.fit_transform(self.x)

        self.x_train, \
        self.x_test, \
        self.y_train,\
        self.y_test = train_test_split(self.x, self.y, test_size=0.15)

    def train(self):
        '''
        Train the model using XGBClassifier with training data.
        Print the accuracy of the model by testing predicted labels to actual.
        '''
        self.model = XGBClassifier(use_label_encoder=False, eval_metric='logloss')
        self.model.fit(self.x_train, self.y_train)
        self.y_pred = self.model.predict(self.x_test)
        accuracy = accuracy_score(self.y_test,self.y_pred)*100
        print("Accuracy: %d" % accuracy)

        c_matrix = confusion_matrix(self.y_test, self.y_pred)
        df = pd.DataFrame(c_matrix, range(len(c_matrix)), range(len(c_matrix)))
        sn.set(font_scale=1.2)
        sn.heatmap(df, annot=True, annot_kws={"size": 14})
        plt.show()

if __name__ == "__main__":

    # Construct the model class
    parkinsons = Model()

    # Pre-process data
    parkinsons.pre_process()

    # Train the model
    parkinsons.train()