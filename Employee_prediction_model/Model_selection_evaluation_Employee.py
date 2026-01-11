
from sklearn.linear_model import LinearRegression,Ridge,Lasso
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error,mean_squared_error,r2_score
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt



def model_selection_evaluation_Employee(X_train_processed,X_val_processed,X_test_processed,
    Y_val,Y_test,Y_train):
    results_df=[]
    models={
        "LinearRegressor":LinearRegression(),
        "Ridge":Ridge(alpha=1.4),
        "Lasso":Ridge(alpha=0.02),
        "DecisionTreeRegressor":DecisionTreeRegressor(random_state=42),
        "RandomForestRegressor":RandomForestRegressor(
            n_estimators=200,
            random_state=42
        )

    }
    for model_name,model in models.items():
        model.fit(X_train_processed,Y_train)
        y_val_pred=model.predict(X_val_processed)
        results_df.append({
        "Model": model_name,
        "MAE": mean_absolute_error(Y_val, y_val_pred),
        "RMSE": np.sqrt(mean_squared_error(Y_val, y_val_pred)),
        "R2": r2_score(Y_val, y_val_pred)
    })

    results_df = pd.DataFrame(results_df).sort_values("RMSE")
    print(results_df)
    best_model_data=results_df[results_df['RMSE']==results_df['RMSE'].min()]
    print(f"The best model is {best_model_data['Model'].iloc[0]} and with RMSE is {best_model_data['RMSE'].iloc[0]}")
    best_model= best_model_data.iloc[0]
    y_test_pred =[]
    match best_model["Model"]:
        case "LinearRegressor":model=LinearRegression()
        case "Ridge":model=Ridge(alpha=1.4)
        case "Lasso":model=Ridge(alpha=0.02)
        case "DecisionTreeRegressor":model=DecisionTreeRegressor(random_state=42)
        case "RandomForestRegressor":model=RandomForestRegressor(
            n_estimators=200,
            random_state=42
        )
    model.fit(X_train_processed,Y_train)
    y_test_pred=model.predict(X_test_processed)
    sns.scatterplot(x=Y_test, y=y_test_pred)
    plt.plot([Y_test.min(), Y_test.max()],
         [Y_test.min(), Y_test.max()],
         color="red")
    plt.xlabel("Actual Performance")
    plt.ylabel("Predicted Performance")
    plt.show()

   
    print( f"Model : {best_model["Model"]}\n MAE:{mean_absolute_error(Y_test, y_test_pred):.2f},\n RMSE : {np.sqrt(mean_squared_error(Y_test, y_test_pred)):.2f},R2: {r2_score(Y_test, y_test_pred):.2f}")
    print("y_train range:", Y_train.min(), Y_train.max())
    print("y_test range :", Y_test.min(), Y_test.max())
    return best_model["Model"],model

    


    






        

    



    

        




    

    






