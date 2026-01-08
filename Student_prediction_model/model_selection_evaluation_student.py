
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score,accuracy_score,classification_report
import pandas as pd
from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt




def model_selection_evaluation_student(X_train_processed,X_val_processed,X_test_processed,
    Y_val,Y_test,Y_train):
    results_df=[]
    models={
        "DecisionTreeClassifier":DecisionTreeClassifier(
            max_depth=6,
            min_samples_leaf=50,
            random_state=42
        ),
        "RandomForestClassifier" :RandomForestClassifier(
            n_estimators=300,
            max_depth=10,
            random_state=42
        ),
        "LogisticRegression":LogisticRegression(
            solver='lbfgs',
            max_iter=3000
        )

    }
    for model_name,model in models.items():
        model.fit(X_train_processed,Y_train)
        y_val_pred=model.predict(X_val_processed)
        results_df.append({
                "Model Name": model_name,
                "f1_score_macro":f1_score(Y_val,y_val_pred,average="macro"),
                "f1_score_weighted":f1_score(Y_val,y_val_pred,average="weighted"),
                "accuracy_score":accuracy_score(Y_val,y_val_pred),
                "classification_report":classification_report(Y_val,y_val_pred)
        })
        print('Classification Report:')
        print(model_name)
        print(classification_report(Y_val,y_val_pred))
    
    results_df=pd.DataFrame(results_df)
    #print(results_df)
    
    best_model_f1_macro=results_df[results_df['f1_score_macro']==results_df['f1_score_macro'].max()]
    best_model_f1_weighted=best_model_f1_macro.sort_values(
    by="f1_score_weighted",
    ascending=False
    )
    print(f"The best model is :{(best_model_f1_weighted["Model Name"].iloc[0])} and f1 value is {best_model_f1_weighted['f1_score_macro'].iloc[0]} and {(best_model_f1_weighted['f1_score_weighted'].iloc[0])}")
    print(f"Accuracy is:{(best_model_f1_weighted['accuracy_score'].iloc[0]):.2f}")
    best_model= best_model_f1_weighted.iloc[0]
    y_test_pred =[]
    match best_model["Model Name"]:
        case "DecisionTreeClassifier":
            model=DecisionTreeClassifier(
            max_depth=6,
            min_samples_leaf=50,
            random_state=42)
        case "RandomForestClassifier":
            model=RandomForestClassifier(
            n_estimators=300,
            max_depth=10,
            random_state=42)
        case "LogisticRegression":
            model=LogisticRegression(
            solver='lbfgs',
            max_iter=3000
        )
    

    model.fit(X_train_processed,Y_train)
    y_test_pred=model.predict(X_test_processed)
    cms=confusion_matrix(Y_test,y_test_pred)
    print("Confusion Matrix:")
    print(cms)
    plt.figure(figsize=(3,4))
    sns.heatmap(
        cms,
        annot=True,
        fmt='d',
        cmap="Blues",
        xticklabels=model.classes_,
        yticklabels=model.classes_
    )
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title(f"Confusion Matrix for {best_model["Model Name"]}")
    plt.show()
    accuracy = accuracy_score(Y_test, y_test_pred)
    print(f"Model Accuracy is : {accuracy * 100:.2f}%")

    return model,best_model["Model Name"]






        

    



    

        




    

    






