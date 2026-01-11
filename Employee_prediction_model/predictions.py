import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler

def sal_perf_score(Employee_df):
    sns.barplot(
        data=Employee_df,
        x="Performance Score",
        y="Salary",
        alpha=0.5
    )
    plt.title("Salary vs Performance score")
    plt.show()

    sns.regplot(
        data=Employee_df,
        x="Performance Score",
        y="Salary",
        scatter_kws={"alpha":0.5},
        line_kws={"color":"red"}
    )
    plt.title("Salary vs Performance Score Trend (Trend)")
    plt.show()

    corr=Employee_df["Performance Score"].corr(Employee_df["Salary"])
    print("Correlation between Performance and Salary",round(corr,3))

    scaler=MinMaxScaler()
    Employee_df["salary_norm"]=scaler.fit_transform(Employee_df[["Salary"]])
    Employee_df["performance_score_norm"]=scaler.fit_transform(Employee_df[["Performance Score"]])

    Employee_df["salary_performance_gap"] = (
    Employee_df["salary_norm"] - Employee_df["performance_score_norm"]
    )

    #identfy top under paid employees
    underpaid_employees=Employee_df.sort_values("salary_performance_gap").head(10)
    print("Top 10 Underpaid Employees")
    print(
        underpaid_employees[
            ["Performance Score","Salary","salary_performance_gap","ID","Name"]
        ])
    #identify top overpaid employees
    overpaid_employees=Employee_df.sort_values("salary_performance_gap",ascending=False).head(10)
    print("Top 10 Overpaid Employees")
    print(
        overpaid_employees[
           ["Performance Score","Salary","salary_performance_gap","ID","Name"]
        ]
    )

    #salary performance gap means
    sns.histplot(
        Employee_df["salary_performance_gap"],kde=True,
    )
    plt.axvline(0,color="Red")
    plt.title('Salary performance gaps')
    plt.show()


