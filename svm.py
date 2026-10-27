import re

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import GridSearchCV, GroupKFold, LeaveOneGroupOut
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import RobustScaler
from sklearn.svm import SVC


def main():
    data = pd.read_csv("Rois/Planilhas/texture_data.csv")
    feature_columns = data.columns[2:]
    data["Patient_ID"] = data["Arquivo"].map(
        lambda name: int(re.search(r"ROI_(\d+)_", name).group(1))
    )

    X = data[feature_columns]
    y = data["Classe"]
    groups = data["Patient_ID"]
    outer_cv = LeaveOneGroupOut()
    param_grid = {"svc__C": [1, 100, 1000], "svc__gamma": ["scale", "auto", 1], "svc__kernel": ["rbf"]}
    results = []

    for train_idx, test_idx in outer_cv.split(X, y, groups):
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
        train_groups = groups.iloc[train_idx]
        inner_cv = GroupKFold(n_splits=min(5, train_groups.nunique()))
        search = GridSearchCV(
            make_pipeline(RobustScaler(), SVC(class_weight="balanced")),
            param_grid,
            scoring="recall",
            cv=inner_cv,
            n_jobs=-1,
        )
        search.fit(X_train, y_train, groups=train_groups)
        predictions = search.predict(X_test)
        accuracy = accuracy_score(y_test, predictions)
        patient_id = groups.iloc[test_idx].iloc[0]
        print(f"Paciente de teste: {patient_id}")
        print(f"Melhores parâmetros: {search.best_params_}")
        print(f"Acurácia no teste: {accuracy:.2f}")
        results.append(
            {
                "Test_Patient_ID": patient_id,
                "Best_Params": search.best_params_,
                "Test_Accuracy": accuracy,
                "Classification_Report": classification_report(
                    y_test, predictions, output_dict=True, zero_division=0
                ),
            }
        )

    overall_accuracy = np.mean([result["Test_Accuracy"] for result in results])
    print(f"Acurácia média geral: {overall_accuracy:.2f}")


if __name__ == "__main__":
    main()
