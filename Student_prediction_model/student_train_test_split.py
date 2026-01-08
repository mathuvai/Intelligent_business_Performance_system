from sklearn.model_selection import train_test_split
def student_train_test_split(df):
    X = df[[
    "Study Hours per Week",
    "Attendance Rate",
    "Participation in Extracurricular Activities",
    "Parent Education Level"
    ]]
    print(df.columns)
    y = df["performance_category"]
    X_train, X_temp, Y_train, Y_temp = train_test_split(
        X, y, test_size=0.30, random_state=42)
    
    X_val,X_test,Y_val,Y_test= train_test_split(
        X_temp,Y_temp,
        test_size=0.5,
        random_state=42
    )
    
    return X_train,X_test,X_val,Y_val,Y_test,Y_train
    