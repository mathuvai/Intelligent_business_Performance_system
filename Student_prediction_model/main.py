import Data_cleansing_student
import Define_target
#import Data_scaling_student
import student_train_test_split
import Feature_Extraction
import model_selection_evaluation_student
import Feature_Importance
print("import done")
def main():
    print("running")
    df_student= Data_cleansing_student.cleansing_student()
    df_student=Define_target.define_target(df_student)
    X_train,X_test,X_val,Y_val,Y_test,Y_train = student_train_test_split.student_train_test_split(df_student)
    X_train_processed,X_val_processed,X_test_processed,features_name=Feature_Extraction.student_feature_extraction(X_train,X_test,X_val,Y_val,Y_test,Y_train)
    model,model_name=model_selection_evaluation_student.model_selection_evaluation_student(X_train_processed,X_val_processed,X_test_processed,
    Y_val,Y_test,Y_train)
    Feature_Importance.Student_feature_selection(X_test_processed,Y_test,model,model_name,features_name)




    
    



    




if __name__ == "__main__":
    main() 
