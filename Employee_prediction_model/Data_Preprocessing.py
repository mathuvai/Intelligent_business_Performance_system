from sklearn.preprocessing import RobustScaler
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler,OneHotEncoder
from sklearn.compose import ColumnTransformer
 

def Data_Preprocessing(Employee_df,X_val,X_train,X_test,Y_test,Y_train,Y_val):
    for col in X_train.select_dtypes(include="object").columns:
        print(f"{col}")
        print(X_train[col].nunique())
    
    numerical_feature_data=X_train.select_dtypes(include="number")
    numerical_feature_data_col=numerical_feature_data.columns

    categorical_feature_data=X_train.select_dtypes(include="object")
    categorical_feature_data_col=categorical_feature_data.columns

    target_feature=Y_train.name

    # seaborn plot for Numerical vs categorical target
    for col in numerical_feature_data_col:
        sns.regplot(
            data=Employee_df,
            x=col,
            y=target_feature,
            lowess=True
        )
        plt.title(f"{col} vs {target_feature}")
        plt.show()
    # categorical vs numeric relationship
    for col in categorical_feature_data_col:
        sns.barplot(
            data=Employee_df,
            x=col,
            y=target_feature,
            estimator="mean"
        )
        plt.title(f"Average {target_feature} by {col}")
        plt.xticks(rotation=45)
        plt.show()
    
    #correlation heatmap
    corr=numerical_feature_data[numerical_feature_data_col].corr()
    sns.heatmap(
        corr,
        annot=True,
        cmap="coolwarm"
    )
    plt.show()
    #class balance for target variable:
    sns.countplot(
        x=Employee_df[target_feature]
    )
    plt.show()

    drop_cols=["Name","Joining Date"]
    categorical_feature_data=categorical_feature_data.drop(columns=drop_cols)
    categorical_feature_data_col=categorical_feature_data.columns

    print("Data Preprocessing started.............")
    print("Preprocessing of dataset started..........")
    preprocessor= ColumnTransformer(
        transformers=[
            ("num",RobustScaler(),numerical_feature_data_col),
            ("cat",OneHotEncoder(),categorical_feature_data_col)

        ]
    )

    X_train_processed = preprocessor.fit_transform(X_train)
    X_val_processed   = preprocessor.transform(X_val)
    X_test_processed  = preprocessor.transform(X_test)
    features_name=preprocessor.get_feature_names_out()
    print(X_train_processed)
    
   

    return X_train_processed,X_val_processed,X_test_processed,features_name










    

