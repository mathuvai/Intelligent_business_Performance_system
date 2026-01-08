from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.inspection import permutation_importance
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def Student_feature_selection(X_test,Y_test,model,model_name,features_name):
    coeffiecients=model.coef_[0]
    odd_ratios=np.exp(coeffiecients)
    # # Display feature importance using coefficients and odds ratios 
    Feature_Importance=pd.DataFrame(
        {
            'Feature':features_name,
            'coeffiecients':coeffiecients,
            'Odd Ratios':odd_ratios

        }
    )
    comparison_df=Feature_Importance.copy()
    print(f"\n Feature Importance (coeffiecients & Odd Ratio)")
    print(Feature_Importance.sort_values(ascending=False, by='coeffiecients'))

    # Display feature importance using permutation importance
    perm_result=permutation_importance(
        model,
        X_test,
        Y_test,
        n_repeats=10,
        random_state=42,
        scoring='f1_macro'
    )
    Feature_Importance=pd.DataFrame(
        {
            'Feature':features_name,
            'Permutation Importance': perm_result.importances_mean

        }
    ).sort_values(ascending=False,by='Permutation Importance')
    comparison_df = (
    comparison_df.merge(Feature_Importance, on="Feature").sort_values(by="Permutation Importance", ascending=False)
)
    print(f"\n Feature Importance (Permutation Importance,Odd radios & coefficients")
    print(comparison_df)

    fig,axes=plt.subplots(1,3,figsize=(13,15),sharey=True)
    axes[0].barh(comparison_df['Feature'],comparison_df['coeffiecients'])
    axes[0].set_title("Coefficient Magnitude")

    axes[1].barh(comparison_df['Feature'],comparison_df['Odd Ratios'])
    axes[1].set_title("Odd ratios Values")

    axes[2].barh(comparison_df['Feature'],comparison_df['Permutation Importance'])
    axes[1].set_title("Permutation Importance   Values")

    for ax in axes:
        ax.invert_yaxis()
        ax.set_xlabel("Importance")
    
    plt.suptitle("Feature Importance Comparison – Logistic Regression")
    plt.tight_layout()
    plt.show()

    print(summary.model())


    




