import Data_cleansing_Employee
import Employee_train_test_split
import Data_Preprocessing
import Model_selection_evaluation_Employee
import predictions

def main():
    Employee_df=Data_cleansing_Employee.Cleansing_Employee()
    X_train,X_test,X_val,Y_val,Y_test,Y_train=Employee_train_test_split.Employee_train_test_split(Employee_df)
    X_train_processed,X_val_processed,X_test_processed,features_name=Data_Preprocessing.Data_Preprocessing(Employee_df,X_val,X_train,X_test,Y_test,Y_train,Y_val)
    model,model_name=Model_selection_evaluation_Employee.model_selection_evaluation_Employee(X_train_processed,X_val_processed,X_test_processed,
    Y_val,Y_test,Y_train)
    predictions.sal_perf_score(Employee_df)






if __name__ == "__main__":
    main()

