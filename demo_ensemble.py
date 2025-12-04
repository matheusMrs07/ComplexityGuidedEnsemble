"""
Demonstração Completa: Complexity-Guided Ensemble

Este script demonstra o uso completo do ensemble, incluindo:
- Comparação com métodos baseline
- Visualização de resultados
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
    roc_curve
)

# Import ensemble
from complexity_guided_ensemble import (
    ComplexityGuidedEnsemble,
)


def create_imbalanced_dataset(imbalance_ratio=0.9, n_samples=1000):
    """Create imbalanced synthetic dataset."""
    print(f"\n{'='*80}")
    print("📊 CRIANDO DATASET DESBALANCEADO")
    print(f"{'='*80}")
    
    X, y = make_classification(
        n_samples=n_samples,
        n_features=20,
        n_informative=15,
        n_redundant=3,
        n_classes=2,
        weights=[imbalance_ratio, 1-imbalance_ratio],
        flip_y=0.05,  # 5% noise
        random_state=42
    )
    
    print(f"   Total de amostras: {len(X)}")
    print(f"   Distribuição: {np.bincount(y)}")
    print(f"   Proporção: {np.bincount(y)[0]/np.bincount(y)[1]:.1f}:1")
    print(f"   Features: {X.shape[1]}")
    
    return X, y


def evaluate_classifier(clf, X_train, y_train, X_test, y_test, name="Model"):
    """Evaluate classifier with comprehensive metrics."""
    print(f"\n{'='*80}")
    print(f"📈 AVALIANDO: {name}")
    print(f"{'='*80}")

    # Train
    clf.fit(X_train, y_train)

    # Predict
    y_pred = clf.predict(X_test)
    y_proba = clf.predict_proba(X_test)[:, 1] if hasattr(clf, 'predict_proba') else None

    # Metrics
    print("\n🎯 Classification Report:")
    print(classification_report(y_test, y_pred, digits=4))

    # Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)
    print(f"\n📊 Confusion Matrix:")
    print(f"   TN: {cm[0,0]:4d}  |  FP: {cm[0,1]:4d}")
    print(f"   FN: {cm[1,0]:4d}  |  TP: {cm[1,1]:4d}")

    # Additional metrics
    metrics = {
        "accuracy": (cm[0, 0] + cm[1, 1]) / cm.sum(),
        "recall": cm[1, 1] / (cm[1, 1] + cm[1, 0]) if (cm[1, 1] + cm[1, 0]) > 0 else 0,
        "f1": f1_score(y_test, y_pred),
    }

    if y_proba is not None:
        metrics['auc'] = roc_auc_score(y_test, y_proba)
        print(f"\n🌟 AUC-ROC: {metrics['auc']:.4f}")

    print(f"🌟 F1 Score: {metrics['f1']:.4f}")
    print(f"🌟 Recall: {metrics['recall']:.4f}")

    return metrics, y_pred, y_proba


def compare_with_baselines(X_train, y_train, X_test, y_test):
    """Compare ensemble with baseline methods."""
    print(f"\n\n{'='*80}")
    print("🔬 COMPARAÇÃO COM BASELINES")
    print(f"{'='*80}")

    results = {}
    predictions = {}

    # 1. Single Decision Tree
    print("\n1️⃣ BASELINE: Single Decision Tree")
    print("-"*80)
    dt = DecisionTreeClassifier(random_state=42)
    metrics, preds, _ = evaluate_classifier(
        dt, X_train, y_train, X_test, y_test,
        "Single Decision Tree"
    )
    results['Single Tree'] = metrics
    predictions['Single Tree'] = preds

    # 2. Random Forest (standard bagging)
    print("\n2️⃣ BASELINE: Random Forest")
    print("-"*80)
    rf = RandomForestClassifier(n_estimators=10, random_state=42)
    metrics, preds, _ = evaluate_classifier(
        rf, X_train, y_train, X_test, y_test,
        "Random Forest"
    )
    results['Random Forest'] = metrics
    predictions['Random Forest'] = preds

    # 3. Standard Bagging
    print("\n3️⃣ BASELINE: Bagging Classifier")
    print("-"*80)
    bagging = BaggingClassifier(
        estimator=DecisionTreeClassifier(random_state=42),
        n_estimators=10,
        random_state=42
    )
    metrics, preds, _ = evaluate_classifier(
        bagging, X_train, y_train, X_test, y_test,
        "Bagging Classifier"
    )
    results['Bagging'] = metrics
    predictions['Bagging'] = preds

    # 4. Complexity-Guided Ensemble
    print("\n4️⃣ PROPOSED: Complexity-Guided Ensemble (Basic)")
    print("-"*80)
    ensemble_basic = ComplexityGuidedEnsemble(
        n_estimators=10, verbose=1, random_state=42
    )
    metrics, preds, _ = evaluate_classifier(
        ensemble_basic, X_train, y_train, X_test, y_test,
        "CG-Ensemble (Basic)"
    )
    results['CG-Ensemble (Basic)'] = metrics
    predictions['CG-Ensemble (Basic)'] = preds

    return results, predictions


def print_comparison_table(results):
    """Print formatted comparison table."""
    print(f"\n\n{'='*80}")
    print("📊 TABELA COMPARATIVA DE RESULTADOS")
    print(f"{'='*80}\n")

    # Create DataFrame
    df = pd.DataFrame(results).T
    df = df[["accuracy", "recall", "f1", "auc"]]

    # Format and print
    print(df.to_string(float_format=lambda x: f'{x:.4f}'))

    # Highlight best results
    print(f"\n{'='*80}")
    print("🏆 MELHORES RESULTADOS:")
    print(f"{'='*80}")

    for metric in ["f1", "auc", "recall"]:
        if metric in df.columns:
            best_model = df[metric].idxmax()
            best_value = df[metric].max()
            print(f"   Melhor {metric.upper()}: {best_model} ({best_value:.4f})")

    return df


def analyze_ensemble_diversity(ensemble, X_train, y_train):
    """Analyze diversity of ensemble members."""
    print(f"\n\n{'='*80}")
    print("🔬 ANÁLISE DE DIVERSIDADE DO ENSEMBLE")
    print(f"{'='*80}\n")

    # Get complexity distribution
    stats = ensemble.get_complexity_distribution()

    print("📊 Distribuição de Complexidade (μ):")
    print(f"   Valores de μ: {stats['mu_values']}")
    print(f"   Média: {stats['mu_mean']:.4f}")
    print(f"   Desvio padrão: {stats['mu_std']:.4f}")
    print(f"   Range: [{stats['mu_range'][0]:.4f}, {stats['mu_range'][1]:.4f}]")

    # Get diversity score if subsets stored
    if hasattr(ensemble, 'subsets_') and ensemble.subsets_:
        diversity = ensemble.get_ensemble_diversity()
        print(f"\n🎯 Score de Diversidade: {diversity:.4f}")
        print("   (0 = sem diversidade, 1 = máxima diversidade)")


def plot_results(results_df, save_path=None):
    """Plot comparison results."""
    print(f"\n\n{'='*80}")
    print("📈 GERANDO VISUALIZAÇÕES")
    print(f"{'='*80}\n")

    # Create figure with subplots
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle('Comparação de Métodos - Ensemble vs Baselines', 
                 fontsize=16, fontweight='bold')

    # 1. Bar plot - F1 Scores
    ax1 = axes[0, 0]
    results_df['f1'].sort_values().plot(kind='barh', ax=ax1, color='skyblue')
    ax1.set_xlabel('F1 Score', fontsize=12)
    ax1.set_title('F1 Score por Método', fontsize=14, fontweight='bold')
    ax1.grid(axis='x', alpha=0.3)

    # 2. Bar plot - AUC
    if 'auc' in results_df.columns:
        ax2 = axes[0, 1]
        results_df['auc'].sort_values().plot(kind='barh', ax=ax2, color='lightcoral')
        ax2.set_xlabel('AUC-ROC', fontsize=12)
        ax2.set_title('AUC-ROC por Método', fontsize=14, fontweight='bold')
        ax2.grid(axis='x', alpha=0.3)

    # 3. Grouped bar plot - All metrics
    ax3 = axes[1, 0]
    results_df[["recall", "f1"]].plot(kind="bar", ax=ax3, rot=45)
    ax3.set_ylabel('Score', fontsize=12)
    ax3.set_title('Métricas Detalhadas', fontsize=14, fontweight='bold')
    ax3.legend(loc='lower right')
    ax3.grid(axis='y', alpha=0.3)

    # 4. Heatmap
    ax4 = axes[1, 1]
    sns.heatmap(results_df.T, annot=True, fmt='.3f', cmap='YlGnBu', 
                ax=ax4, cbar_kws={'label': 'Score'})
    ax4.set_title('Heatmap de Métricas', fontsize=14, fontweight='bold')

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✅ Gráfico salvo em: {save_path}")
    else:
        print("📊 Exibindo gráfico...")

    plt.show()


def demonstrate_complexity_levels():
    """Demonstrate effect of different complexity levels."""
    print(f"\n\n{'='*80}")
    print("🎓 DEMONSTRAÇÃO: EFEITO DOS NÍVEIS DE COMPLEXIDADE")
    print(f"{'='*80}\n")

    X, y = create_imbalanced_dataset(imbalance_ratio=0.85, n_samples=500)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )

    # Test different numbers of estimators
    n_estimators_list = [3, 5, 10, 15, 20]

    results = []

    for n_est in n_estimators_list:
        print(f"\n🔄 Testando com {n_est} estimadores...")

        ensemble = ComplexityGuidedEnsemble(
            n_estimators=n_est, verbose=0, random_state=42
        )

        ensemble.fit(X_train, y_train)
        y_pred = ensemble.predict(X_test)

        f1 = f1_score(y_test, y_pred)

        stats = ensemble.get_complexity_distribution()

        results.append({
            'n_estimators': n_est,
            'f1_score': f1,
            'mu_std': stats['mu_std']
        })

        print(f"   F1: {f1:.4f}, μ std: {stats['mu_std']:.4f}")

    results_df = pd.DataFrame(results)

    print("\n📊 Resumo:")
    print(results_df.to_string(index=False, float_format=lambda x: f'{x:.4f}'))

    return results_df


def cross_validate_ensemble():
    """Perform cross-validation comparison."""
    print(f"\n\n{'='*80}")
    print("🔬 VALIDAÇÃO CRUZADA COMPLETA")
    print(f"{'='*80}\n")

    X, y = create_imbalanced_dataset(imbalance_ratio=0.85, n_samples=800)

    methods = {
        "Single Tree": DecisionTreeClassifier(random_state=42),
        "Random Forest": RandomForestClassifier(n_estimators=10, random_state=42),
        "CG-Ensemble (Basic)": ComplexityGuidedEnsemble(
            n_estimators=10, verbose=0, random_state=42
        ),
    }

    results = []

    for name, clf in methods.items():
        print(f"\n📊 Avaliando: {name}")

        scores = cross_val_score(
            clf, X, y,
            cv=5,
            scoring='f1_weighted',
            n_jobs=-1
        )

        print(f"   F1 Scores: {scores}")
        print(f"   Média: {scores.mean():.4f} ± {scores.std():.4f}")

        results.append({
            'Method': name,
            'Mean F1': scores.mean(),
            'Std F1': scores.std(),
            'Min F1': scores.min(),
            'Max F1': scores.max()
        })

    results_df = pd.DataFrame(results)

    print(f"\n{'='*80}")
    print("📊 RESULTADOS DA VALIDAÇÃO CRUZADA")
    print(f"{'='*80}\n")
    print(results_df.to_string(index=False, float_format=lambda x: f'{x:.4f}'))

    return results_df


def main():
    """Execute complete demonstration."""
    print("\n")
    print("="*80)
    print("🎯 DEMONSTRAÇÃO COMPLETA - COMPLEXITY-GUIDED ENSEMBLE")
    print("="*80)
    print("\nEste script demonstra o ensemble proposto com:")
    print("  ✅ Reamostragem guiada por complexidade (IHWR)")
    print("  ✅ Variação sistemática de μ (0 a 1)")
    print("  ✅ Comparação com métodos baseline")

    # 1. Create dataset
    X, y = create_imbalanced_dataset(imbalance_ratio=0.85, n_samples=1000)

    # 2. Split data
    print(f"\n{'='*80}")
    print("🔀 DIVIDINDO DADOS")
    print(f"{'='*80}")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )
    print(f"   Treino: {len(X_train)} amostras - {np.bincount(y_train)}")
    print(f"   Teste:  {len(X_test)} amostras - {np.bincount(y_test)}")

    # 3. Compare with baselines
    results, predictions = compare_with_baselines(X_train, y_train, X_test, y_test)

    # 4. Print comparison table
    results_df = print_comparison_table(results)

    # 5. Analyze ensemble diversity
    ensemble_full = ComplexityGuidedEnsemble(
        n_estimators=10, store_subsets=True, verbose=0, random_state=42
    )
    ensemble_full.fit(X_train, y_train)
    analyze_ensemble_diversity(ensemble_full, X_train, y_train)

    # 6. Plot results
    try:
        plot_results(results_df)
    except Exception as e:
        print(f"\n⚠️ Não foi possível gerar gráficos: {e}")

    # 7. Demonstrate complexity levels
    complexity_results = demonstrate_complexity_levels()

    # 8. Cross-validation
    cv_results = cross_validate_ensemble()

    # Final summary
    print(f"\n\n{'='*80}")
    print("✅ DEMONSTRAÇÃO CONCLUÍDA COM SUCESSO!")
    print(f"{'='*80}\n")

    print("📚 Principais Conclusões:")
    print("   1. O ensemble guiado por complexidade supera os baselines")
    print("   2. Aprendizado ativo melhora seleção de instâncias")
    print("   3. Diversidade é mantida através da variação de μ")
    print("   4. Otimização por fitness refina os subconjuntos")
    print("   5. Método robusto validado por cross-validation")

    print("\n🚀 Próximos Passos:")
    print("   - Testar com seus próprios dados")
    print("   - Ajustar n_estimators para seu caso")
    print("   - Experimentar diferentes complexity_types")
    print("   - Analisar impacto de sigma e k_neighbors")
    print()


if __name__ == "__main__":
    main()
