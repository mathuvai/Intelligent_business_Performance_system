import Data_cleansing_Employee
import Employee_train_test_split
import Data_Preprocessing

def main():
    Employee_df=Data_cleansing_Employee.Cleansing_Employee()
    X_train,X_test,X_val,Y_val,Y_test,Y_train=Employee_train_test_split.Employee_train_test_split(Employee_df)
    Data_Preprocessing.Data_Preprocessing(Employee_df,X_val,X_train,X_test,Y_test,Y_train,Y_val)





if __name__ == "__main__":
    main()

