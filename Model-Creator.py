import pandas as pd
import numpy as mp
import h2o
from h2o.automl import H2OAutoML




if __name__ == "__main__":

    # Initialing h2o instance
    h2o.init()
    
    # Creating a data frame
    dfh = h2o.import_file("path")

    # Spliting our data 80% training, 20% testing which are different each time
    splits = dfh.split_frame(ratios=[0.8],seed=1)

    train = splits[0]
    test = splits[1]
    
    # Selecting the training columns
    y_train = 'C24'
    x_train = dfh.columns

    # Removing the column that we want to predict
    x_train.remove(y_train)
    
    # Creating the models 
    model = H2OAutoML(max_runtime_secs=480, verbosity="info")
    # Training the models
    model.train(x=x_train,y=y_train, training_frame=train)

    # Getting the predictions for the best model
    pred = model.leader.predict(test)