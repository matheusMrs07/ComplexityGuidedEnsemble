"""
Demonstração Completa: Complexity-Guided Ensemble

Este script demonstra o uso completo do ensemble baseado em IHWR com
aprendizado ativo e otimização por fitness, incluindo:
- Comparação com métodos baseline
- Análise de diversidade
- Visualização de resultados
- Diferentes configurações
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestClassifier, BaggingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_auc_score,
    f1_score,
    precision_recall_curve,
    roc_curve,
)

# Import ensemble
from complexity_guided_ensemble import (
    ComplexityGuidedEnsemble,
    EnsembleConfig,
    compare_ensemble_strategies,
)


def main():
    diretorio = "../_PGCDS/extract_5f_data/"
    base_name = "ecoli2"

    data_train = pd.read_csv(
        f"{diretorio}{base_name}/{base_name}-5-1tra.dat",
        header=None,
        delimiter=",",
        comment="@",
    )
    data_test = pd.read_csv(
        f"{diretorio}{base_name}/{base_name}-5-1tst.dat",
        header=None,
        delimiter=",",
        comment="@",
    )

    data_train = data_train.replace({"positive": 1, "negative": 0}, regex=True)
    data_train = data_train.replace({"M": 0, "F": 1, "I": 2}, regex=True)

    data_test = data_test.replace({"positive": 1, "negative": 0}, regex=True)
    data_test = data_test.replace({"M": 0, "F": 1, "I": 2}, regex=True)

    np_data_train = data_train.to_numpy()
    np_data_test = data_test.to_numpy()

    X_train = np.asarray(np_data_train[:, :-1], dtype=np.float32)
    y_train = np.asarray(np_data_train[:, -1], dtype=np.int32)
    X_test = np_data_test[:, :-1]
    y_test = np_data_test[:, -1]

    ensemble_full = ComplexityGuidedEnsemble(
        n_estimators=10,
        hardness_function="local_set_cardinality",
        use_active_learning=False,
        use_fitness_optimization=False,
        store_subsets=False,
        max_fitness_iterations=2,
        verbose=0,
        random_state=42,
        n_jobs=1,
    )
    ensemble_full.fit(X_train, y_train)

    result = ensemble_full.predict(X_test)

    print("F1 Score:", f1_score(y_test, result))


if __name__ == "__main__":
    main()
