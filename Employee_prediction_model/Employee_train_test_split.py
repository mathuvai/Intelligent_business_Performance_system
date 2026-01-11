from sklearn.model_selection import train_test_split
import pandas as pd

def Employee_train_test_split(Employee_df):

    
    Employee_df['Performance Category']=pd.cut(
        Employee_df['Performance Score'],
        bins=[0,2,3,4,5],
        labels=["Low", "Average", "Good", "Excellent"],
        include_lowest=True
    )
    Employee_df_train=Employee_df.drop(columns=['Performance Score','Salary','ID','Performance Category'])
    X=Employee_df_train
    Y=Employee_df['Performance Score']

    X_train, X_temp, Y_train, Y_temp = train_test_split(
        X, Y, test_size=0.30, random_state=42)
    
    X_val,X_test,Y_val,Y_test= train_test_split(
        X_temp,Y_temp,
        test_size=0.5,
        random_state=42
    )
    
    print(X_test.head(5))
    print(X_train.head(5))
    return X_train,X_test,X_val,Y_val,Y_test,Y_train

