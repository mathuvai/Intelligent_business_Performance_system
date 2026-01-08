import pandas as pd
import numpy as np
import warnings
import matplotlib.pyplot  as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler,OneHotEncoder
from sklearn.compose import ColumnTransformer

def student_feature_extraction(X_train,X_test,X_val,Y_val,Y_test,Y_train):
    X_train_numeric=(X_train.select_dtypes(include=["float64", "int64"]))
    corr_input=X_train_numeric.corr()
    #print(corr_input.head())
    #print(X_train['Study Hours per Week'].dtype,X_train['Attendance Rate'].dtype)
    #corr_outcome=X_train_numeric.corrwith(Y_train).to_frame(name="Correlation with Target")
    print(f"Please find the correlation between : {X_train.columns} and {Y_train.name}")
    print(corr_input.sort_values(by='Study Hours per Week',ascending=True))
    
    numerical_features=(X_train_numeric.columns).tolist()
    categorical_features=(X_train.select_dtypes(include=['object','object','category']).columns).tolist()

    print("Preprocessing of dataset started..........")
    preprocessor= ColumnTransformer(
        transformers=[
            ("num",StandardScaler(),numerical_features),
            ("cat",OneHotEncoder(),categorical_features)

        ]
    )

    X_train_processed = preprocessor.fit_transform(X_train)
    X_val_processed   = preprocessor.transform(X_val)
    X_test_processed  = preprocessor.transform(X_test)
    features_name=preprocessor.get_feature_names_out()
    
   

    return X_train_processed,X_val_processed,X_test_processed,features_name

   


