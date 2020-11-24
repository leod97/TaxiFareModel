# imports
from TaxiFareModel.encoders import TimeFeaturesEncoder, DistanceTransformer
from TaxiFareModel.data import get_data,clean_data
from sklearn.model_selection import train_test_split
from  sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import make_pipeline, Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
import pandas as pd
import numpy as np

class Trainer():
    def __init__(self, X, y):
        """
            X: pandas DataFrame
            y: pandas Series
        """
        self.pipeline = None
        self.X = X
        self.y = y

    def set_pipeline(self):
        """defines the pipeline as a class attribute"""
        pipe_distance = make_pipeline(DistanceTransformer(),StandardScaler())
        pipe_time = make_pipeline(TimeFeaturesEncoder(time_column='pickup_datetime'), OneHotEncoder(handle_unknown='ignore'))
        
        dist_cols = ['pickup_latitude', 'pickup_longitude', 'dropoff_latitude', 'dropoff_longitude']
        time_cols = ['pickup_datetime']
        
        feat_eng_bloc = ColumnTransformer([('time', pipe_time, time_cols),
                                        ('distance', pipe_distance, dist_cols)])
        
        self.pipeline = Pipeline(steps=[('feat_eng_bloc', feat_eng_bloc),
                            ('regressor',RandomForestRegressor())])
                                        
        return self.pipeline

    def run(self):
        """set and train the pipeline"""
        self.pipeline=self.set_pipeline()
        self.pipeline.fit(self.X, self.y)
        return self.pipeline

    def evaluate(self, X_test, y_test):
        """evaluates the pipeline on df_test and return the RMSE"""
        y_pred=self.pipeline.predict(X_test)
        return np.sqrt(((y_pred - y_test)**2).mean())


if __name__ == "__main__":
    
    
    # get data and clean
    df=clean_data(get_data())
    
    # set X and y
    X=df.drop('fare_amount',axis=1)
    y=df.fare_amount
    # hold out
    X_train,X_test,y_train,y_test=train_test_split(X,y)
    trainer=Trainer(X_train,y_train)
    # train
    trainer.run()
    # evaluate
    trainer.evaluate(X_test,y_test)
    
