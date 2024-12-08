import pytest
import pandas as pd
import numpy as np
from prediction_demo import data_preparation,data_split,train_model,eval_model

@pytest.fixture
def housing_data_sample():
    return pd.DataFrame(
      data ={
      'price':[13300000,12250000],
      'area':[7420,8960],
    	'bedrooms':[4,4],	
      'bathrooms':[2,4],	
      'stories':[3,4],	
      'mainroad':["yes","yes"],	
      'guestroom':["no","no"],	
      'basement':["no","no"],	
      'hotwaterheating':["no","no"],	
      'airconditioning':["yes","yes"],	
      'parking':[2,3],
      'prefarea':["yes","no"],	
      'furnishingstatus':["furnished","unfurnished"]}
    )

def test_data_preparation(housing_data_sample):
    feature_df, target_series = data_preparation(housing_data_sample)
    # Target and datapoints has same length
    assert feature_df.shape[0]==len(target_series)

    #Feature only has numerical values
    assert feature_df.shape[1] == feature_df.select_dtypes(include=(np.number,np.bool_)).shape[1]

@pytest.fixture
def feature_target_sample(housing_data_sample):
    feature_df, target_series = data_preparation(housing_data_sample)
    return (feature_df, target_series)

def test_data_split(feature_target_sample):
    return_tuple = data_split(*feature_target_sample)
    
    # Test if the length of return_tuple is 4
    assert len(return_tuple) == 4, "The returned tuple must have four elements."
    
    # Test if the first two elements are DataFrames (X_train, X_test)
    assert isinstance(return_tuple[0], pd.DataFrame), "X_train should be a DataFrame."
    assert isinstance(return_tuple[1], pd.DataFrame), "X_test should be a DataFrame."
    
    # Test if the last two elements are Series (y_train, y_test)
    assert isinstance(return_tuple[2], pd.Series), "y_train should be a Series."
    assert isinstance(return_tuple[3], pd.Series), "y_test should be a Series."
    
    # Test if the data is split into train and test sets
    total_rows = feature_target_sample[0].shape[0]
    train_test_rows = return_tuple[0].shape[0] + return_tuple[1].shape[0]
    assert total_rows == train_test_rows, "The total number of rows in the dataset should match the sum of training and testing rows."
