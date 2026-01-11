import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import warnings

warnings.filterwarnings('ignore')
warnings.filterwarnings(action='ignore',category=DeprecationWarning)

def Cleansing_Employee()-> pd.DataFrame:
    Employee = pd.read_csv("data_raw/Employe_Performance_dataset.csv")
    Employee_df=pd.DataFrame(Employee)
    print(Employee_df.head(5))
    print(Employee_df.columns)
    pd.set_option('display.max_columns',None)
    pd.set_option('display.float_format',lambda x:'%.2f' %x)
    print(Employee_df.describe())
    print(Employee_df.info())
    print(Employee_df.isna().sum())
    print(Employee_df.value_counts())
    categorical_Employee_df=Employee_df.select_dtypes(include=['object','object','category'])
    numerical_Empliyee_df=Employee_df.select_dtypes(include=['int64','float64'])
    

    print(categorical_Employee_df.columns)
    print(numerical_Empliyee_df.columns)



    #For data cleansing only performance score will be considered as it has only one null value.
     # Handle missing values
    Employee_df.loc[:, 'Performance Score'] = (
        Employee_df['Performance Score']
        .fillna(Employee_df['Performance Score'].median())
    )
    
    print(Employee_df.isna().sum())
    numerical_Empliyee_df=numerical_Empliyee_df.drop(columns=['ID','Performance Score'])


      # Statistical representation for numerical and categorical datset.
    categorical_Employee_df_columns=categorical_Employee_df.columns
    numerical_Empliyee_df_columns=numerical_Empliyee_df.columns
    
    for col in numerical_Empliyee_df_columns:
        plt.figure(figsize=(6,8))
        plt.hist(numerical_Empliyee_df[col],bins=100,color='blue',edgecolor='black')
        plt.ylabel(f"Distribution of {col} data distribution")
        plt.xlabel(col)
        plt.title(f"Histogram of data distribution for {col} as below")
        plt.show()
    drop_cols=["Name","Joining Date"]
    categorical_feature_data=categorical_Employee_df.drop(columns=drop_cols)
    categorical_Employee_df_columns=categorical_feature_data.columns

    for col in categorical_Employee_df_columns:
        X=categorical_Employee_df[col].value_counts()
     
        plt.figure(figsize=(6,8))
        plt.pie(
                X.values,
                labels=X.index,
                autopct='%1.1f%%',  # Add percentage labels with one decimal place
                shadow=True,
                startangle=90,  # Start the first slice at 90 degrees (12 o'clock)
                 wedgeprops={'edgecolor': 'white', 'linewidth': 1} # Customize wedge borders
                )
        plt.axis('equal')
        plt.title(f"{col} distribution")
        plt.legend(X.index,loc="best")
        plt.show()
    
    for col in numerical_Empliyee_df_columns:
        plt.figure(figsize=(6,8))
        plt.boxplot(numerical_Empliyee_df[col],vert=False)
        plt.ylabel(f"Distribution of {col} outlier")
        plt.xlabel(col)
        plt.title('Boxplot Distribution')
        plt.show()
        # calculation of outliers
        Q_25=numerical_Empliyee_df[col].quantile(0.25)
        Q_75=numerical_Empliyee_df[col].quantile(0.75)
        IQR=Q_75-Q_25
        lower=Q_25-1.5*IQR
        upper=Q_75+1.5*IQR
        outliers=numerical_Empliyee_df[(numerical_Empliyee_df[col] <lower) | (numerical_Empliyee_df[col]>upper)]
        print(f"percentages of outliers for {col} is {outliers.shape[0]/numerical_Empliyee_df[col].shape[0]*100:.2f}")

    return Employee_df















    
