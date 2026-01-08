from sklearn.preprocessing import MinMaxScaler
import pandas as pd
def define_target(df)->pd.DataFrame:
    #print(df.isna().sum())
    scaler = MinMaxScaler()
    df[["grades_norm"]] = scaler.fit_transform(df[["Previous Grades"]])
    df[["attendance_norm"]] = scaler.fit_transform(df[["Attendance Rate"]])
    df[["study_hours_norm"]] = scaler.fit_transform(df[["Study Hours per Week"]])
    values=df['Parent Education Level'].unique()
    #print(values)
    #print(df["Participation in Extracurricular Activities"].unique())
    edu_map = {
    "High School": 1,
    "Associate": 2,
    "Bachelor": 3,
    "Master": 4,
    "Doctorate": 5
    }
    
    df['parent_edu_score']=df['Parent Education Level'].map(edu_map)
    df['parent_edu_norm']=scaler.fit_transform(df[['parent_edu_score']])

    df["extra_curricular_raw"] = df["Participation in Extracurricular Activities"].map(
    {"Yes": 1, "No": 0})
    df["passed_raw"] = df["Passed"].map({"Yes": 1, "No": 0})
    
    df["extra_curricular_norm"] = scaler.fit_transform(df[["extra_curricular_raw"]])
    df["Passed_norm"] = scaler.fit_transform(df[["passed_raw"]])


    df["overall_performance"] = (
    0.35 * df["grades_norm"] +
    0.05 * df["attendance_norm"] +
    0.15 * df["study_hours_norm"] +
    0.05 * df["extra_curricular_norm"] +
    0.05 * df["parent_edu_norm"]+
    0.35 * df["Passed_norm"])

    
    print(df["overall_performance"].head())
    print(df["overall_performance"].describe())
    print(df["overall_performance"].isna().sum())

    df['performance_category']=pd.qcut(
        df['overall_performance'],q=4,
        labels=["Low","Average","Good","Excellent"]
    )

    print(df["performance_category"].value_counts())
    print(df["performance_category"].value_counts(normalize=True))

    return df


    