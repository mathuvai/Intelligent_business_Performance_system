import pandas as pd
import numpy as np
import warnings
import matplotlib.pyplot  as plt
from sklearn.preprocessing import OneHotEncoder


warnings.filterwarnings('ignore')
warnings.filterwarnings(action='ignore',category=DeprecationWarning)


def cleansing_student() -> pd.DataFrame:

    student = pd.read_csv("/Users/vaishalimathur/Documents/pyhton_learning_llms/first_project_capstone/data_raw/student_performance_prediction.csv")
    df_student = pd.DataFrame(student)
    pd.set_option('display.max_columns',None)
    pd.set_option('display.float_format',lambda x:'%.2f' %x)
    df_student=df_student.drop(columns=['Student ID'])
    student_dtypes_object = df_student.select_dtypes(include=['object','object','category'])
    student_dtypes_numerical = df_student.select_dtypes(include= 'float64')
  
    object_cols= student_dtypes_object.columns.to_list()
    numeric_cols = student_dtypes_numerical.columns.to_list()

    student_dtypes_numerical[numeric_cols]=student_dtypes_numerical[numeric_cols].apply(lambda cols: cols.fillna(cols.median()))
    student_dtypes_object[object_cols]=student_dtypes_object[object_cols].apply(lambda cols: cols.fillna(cols.mode()[0]))
    for cols in object_cols:
      print(student_dtypes_object[cols].unique())

    print(student_dtypes_object.isna().sum())
    #print(student_dtypes_object.columns)

    #print(student_dtypes_object.columns)
    #eprint(student_dtypes_numerical.columns)

    # checking numerical distribution
    for cols in student_dtypes_numerical.columns:
        plt.hist(student_dtypes_numerical[cols],bins=100,color='red',edgecolor='black')
        plt.title(f"Distribution System {student_dtypes_numerical[cols]}")
        plt.xlabel(cols)
        plt.ylabel("frequency")
        plt.show()
    
    for cols in student_dtypes_numerical.columns:
        plt.figure(figsize=(6,8))
        plt.boxplot(student_dtypes_numerical[cols],vert=False)
        plt.title(f"Boxplot for {cols} :")
        plt.xlabel(cols)
        plt.show()
    
    for cols in student_dtypes_numerical.columns:
        Q_25,Q_75=student_dtypes_numerical[cols].quantile(0.25),student_dtypes_numerical[cols].quantile(0.75)
        IQR=Q_75-Q_25
        lower=Q_25-1.5*IQR
        upper=Q_75+1.5*IQR
        outliers=student_dtypes_numerical[(student_dtypes_numerical[cols]< lower) | 
                                          (student_dtypes_numerical[cols] > upper)]
        #print("Number of outliers:", outliers.shape[0])
        #print(outliers["Study Hours per Week"].head())
        print(f"Percentage of outlier for {cols}: {(outliers.shape[0]/student_dtypes_numerical[cols].shape[0])*100:.2f}%")
        print("Since all the data having outliers is very low in percentage therefore not required to remove them")
    df_student=pd.concat([student_dtypes_numerical[numeric_cols],student_dtypes_object[object_cols]],axis=1)
    #print(df_student.isna().sum())
    return df_student

    







    

    

 







 
    
 