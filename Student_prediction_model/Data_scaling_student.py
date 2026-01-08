from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import pandas as pd




def student_scaling(df):
    print(df.head())
    print("Welcome to data scaling of student database.")
    instructions ="""Since the outliers were very less for numerical one thats why we will 
    apply z-score /standarization method """
    print(instructions)
    df_numerical_values=df.select_dtypes(include= 'float64')
    numerical_cols=df_numerical_values.columns

    scaler= StandardScaler()
    df_scaled=scaler.fit_transform(df_numerical_values)
    scaled_dataset= pd.DataFrame(
        df_scaled, columns=numerical_cols)
    
    return scaled_dataset
    
