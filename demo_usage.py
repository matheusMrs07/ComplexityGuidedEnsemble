"""
Exemplo Prático: Usando ComplexityGuidedSampler

"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_auc_score,
    f1_score
)

# Importar o sampler refatorado
from complexity_guided_ensemble import (
    ComplexityGuidedSampler,
    SamplerConfig,
)


def create_imbalanced_dataset():
    """Criar dataset desbalanceado sintético."""
    print("📊 Criando dataset desbalanceado...")
    
    X, y = make_classification(
        n_samples=2000,
        n_features=20,
        n_informative=15,
        n_redundant=3,
        n_classes=2,
        weights=[0.95, 0.05],  # 95% vs 5% - muito desbalanceado!
        flip_y=0.05,  # 5% de ruído
        random_state=42
    )
    
    print(f"   Total de amostras: {len(X)}")
    print(f"   Distribuição original: {np.bincount(y)}")
    print(f"   Proporção: {np.bincount(y)[0]}/{np.bincount(y)[1]} "
          f"= {np.bincount(y)[0]/np.bincount(y)[1]:.1f}:1")
    
    return X, y


def evaluate_model(X_train, y_train, X_test, y_test, model_name="Model"):
    """Avaliar modelo com métricas completas."""
    clf = RandomForestClassifier(n_estimators=100, random_state=42)
    clf.fit(X_train, y_train)
    
    y_pred = clf.predict(X_test)
    y_proba = clf.predict_proba(X_test)[:, 1]
    
    print(f"\n📈 Resultados - {model_name}")
    print("="*60)
    
    # Métricas gerais
    print("\n🎯 Classification Report:")
    print(classification_report(y_test, y_pred))
    
    # Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)
    print(f"\n📊 Confusion Matrix:")
    print(f"   TN: {cm[0,0]:4d}  |  FP: {cm[0,1]:4d}")
    print(f"   FN: {cm[1,0]:4d}  |  TP: {cm[1,1]:4d}")
    
    # ROC-AUC
    auc = roc_auc_score(y_test, y_proba)
    print(f"\n🌟 ROC-AUC Score: {auc:.4f}")
    
    # F1 Score
    f1 = f1_score(y_test, y_pred)
    print(f"🌟 F1 Score: {f1:.4f}")
    
    return {
        'accuracy': (cm[0,0] + cm[1,1]) / cm.sum(),
        'precision': cm[1,1] / (cm[1,1] + cm[0,1]) if (cm[1,1] + cm[0,1]) > 0 else 0,
        'recall': cm[1,1] / (cm[1,1] + cm[1,0]) if (cm[1,1] + cm[1,0]) > 0 else 0,
        'f1': f1,
        'auc': auc
    }


def compare_resampling_strategies(X_train, y_train, X_test, y_test):
    """Comparar diferentes estratégias de resampling."""
    print("\n\n" + "="*80)
    print("🔬 COMPARAÇÃO DE ESTRATÉGIAS DE RESAMPLING")
    print("="*80)
    
    results = {}
    
    # 1. Sem resampling (baseline)
    print("\n\n1️⃣ BASELINE - Sem Resampling")
    print("-"*80)
    results['baseline'] = evaluate_model(
        X_train, y_train, X_test, y_test,
        "Baseline (No Resampling)"
    )
    
    # 2. Com resampling - diferentes configurações de mu
    strategies = {
        'easy_instances': (0.0, 0.2, "Favor Instâncias Fáceis"),
        'uniform': (0.5, 0.2, "Sampling Uniforme"),
        'hard_instances': (1.0, 0.2, "Favor Instâncias Difíceis"),
    }
    
    for i, (key, (mu, sigma, description)) in enumerate(strategies.items(), 2):
        print(f"\n\n{i}️⃣ {description.upper()}")
        print(f"   Parâmetros: mu={mu}, sigma={sigma}")
        print("-"*80)
        
        sampler = ComplexityGuidedSampler(
            complex_type='overlap',
            random_state=42
        )
        
        X_resampled, y_resampled = sampler.fit_resample(
            X_train, y_train,
            mu=mu,
            sigma=sigma,
            k_neighbors=5
        )
        
        print(f"\n   Após resampling: {np.bincount(y_resampled)}")
        
        results[key] = evaluate_model(
            X_resampled, y_resampled, X_test, y_test,
            description
        )
    
    return results


def compare_complexity_types(X_train, y_train, X_test, y_test):
    """Comparar diferentes tipos de complexidade."""
    print("\n\n" + "="*80)
    print("🔬 COMPARAÇÃO DE MÉTRICAS DE COMPLEXIDADE")
    print("="*80)
    
    complexity_types = {
        'overlap': 'Overlap (Sobreposição de Features)',
        'neighborhood': 'Neighborhood (Homogeneidade de Vizinhança)',
        'error_rate': 'Error Rate (Taxa de Erro Cross-Validation)'
    }
    
    results = {}
    
    for i, (ctype, description) in enumerate(complexity_types.items(), 1):
        print(f"\n\n{i}️⃣ {description.upper()}")
        print("-"*80)
        
        sampler = ComplexityGuidedSampler(
            complex_type=ctype,
            random_state=42
        )
        
        X_resampled, y_resampled = sampler.fit_resample(
            X_train, y_train,
            mu=0.5,
            sigma=0.2,
            k_neighbors=5
        )
        
        print(f"\n   Após resampling: {np.bincount(y_resampled)}")
        
        results[ctype] = evaluate_model(
            X_resampled, y_resampled, X_test, y_test,
            description
        )
    
    return results


def print_summary_table(results, title="Resumo dos Resultados"):
    """Imprimir tabela resumida dos resultados."""
    print("\n\n" + "="*80)
    print(f"📊 {title.upper()}")
    print("="*80)
    
    # Header
    print(f"\n{'Estratégia':<25} {'Accuracy':>10} {'Precision':>10} {'Recall':>10} {'F1':>10} {'AUC':>10}")
    print("-"*80)
    
    # Rows
    for name, metrics in results.items():
        print(f"{name:<25} "
              f"{metrics['accuracy']:>10.4f} "
              f"{metrics['precision']:>10.4f} "
              f"{metrics['recall']:>10.4f} "
              f"{metrics['f1']:>10.4f} "
              f"{metrics['auc']:>10.4f}")
    
    # Best results
    print("\n" + "="*80)
    print("🏆 MELHORES RESULTADOS:")
    
    best_f1 = max(results.items(), key=lambda x: x[1]['f1'])
    best_auc = max(results.items(), key=lambda x: x[1]['auc'])
    best_recall = max(results.items(), key=lambda x: x[1]['recall'])
    
    print(f"   Melhor F1 Score: {best_f1[0]} ({best_f1[1]['f1']:.4f})")
    print(f"   Melhor AUC: {best_auc[0]} ({best_auc[1]['auc']:.4f})")
    print(f"   Melhor Recall: {best_recall[0]} ({best_recall[1]['recall']:.4f})")


def demonstrate_advanced_usage():
    """Demonstrar uso avançado com configurações customizadas."""
    print("\n\n" + "="*80)
    print("🚀 DEMONSTRAÇÃO DE USO AVANÇADO")
    print("="*80)
    
    # Criar dataset
    X, y = create_imbalanced_dataset()
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )
    
    # Exemplo 1: Usando SamplerConfig
    print("\n\n1️⃣ Usando SamplerConfig para configuração reutilizável")
    print("-"*80)
    
    config = SamplerConfig(
        complex_type='overlap',
        random_state=42,
        cv_folds=5
    )
    
    sampler = ComplexityGuidedSampler(config=config)
    print(f"   Configuração: {config}")
    
    X_res, y_res = sampler.fit_resample(
        X_train, y_train,
        mu=0.5, sigma=0.2, k_neighbors=5
    )
    print(f"   Resultado: {np.bincount(y_res)}")
    
    # Exemplo 2: Pipeline com múltiplas iterações
    print("\n\n2️⃣ Testando múltiplos valores de mu automaticamente")
    print("-"*80)
    
    mu_values = np.linspace(0, 1, 5)
    best_score = 0
    best_mu = 0
    
    for mu in mu_values:
        sampler = ComplexityGuidedSampler(
            complex_type='overlap',
            random_state=42
        )
        
        X_res, y_res = sampler.fit_resample(
            X_train, y_train,
            mu=mu, sigma=0.2, k_neighbors=5
        )
        
        # Avaliar com cross-validation
        clf = RandomForestClassifier(n_estimators=50, random_state=42)
        scores = cross_val_score(clf, X_res, y_res, cv=3, scoring='f1')
        mean_score = scores.mean()
        
        print(f"   mu={mu:.2f}: F1={mean_score:.4f} (±{scores.std():.4f})")
        
        if mean_score > best_score:
            best_score = mean_score
            best_mu = mu
    
    print(f"\n   🏆 Melhor mu: {best_mu:.2f} (F1={best_score:.4f})")
    
    # Exemplo 3: Análise de complexidades
    print("\n\n3️⃣ Analisando distribuição de complexidades")
    print("-"*80)
    
    sampler = ComplexityGuidedSampler(
        complex_type='overlap',
        random_state=42
    )
    sampler.fit(X_train, y_train)
    
    complexities = sampler.complexities
    
    print(f"   Média: {complexities.mean():.4f}")
    print(f"   Desvio padrão: {complexities.std():.4f}")
    print(f"   Mínimo: {complexities.min():.4f}")
    print(f"   Máximo: {complexities.max():.4f}")
    print(f"   Mediana: {np.median(complexities):.4f}")
    
    # Distribuição por classe
    for cls in np.unique(y_train):
        cls_complexities = complexities[y_train == cls]
        print(f"\n   Classe {cls}:")
        print(f"      Média: {cls_complexities.mean():.4f}")
        print(f"      Desvio: {cls_complexities.std():.4f}")


def main():
    """Função principal - executar todos os experimentos."""
    print("\n")
    print("="*80)
    print("🎯 DEMONSTRAÇÃO COMPLETA - COMPLEXITY GUIDED SAMPLER")
    print("="*80)
    print("\nEste script demonstra o uso prático do ComplexityGuidedSampler")
    print("refatorado em um cenário real de classificação desbalanceada.")
    
    # Criar dataset
    X, y = create_imbalanced_dataset()
    
    # Split train/test
    print("\n🔀 Dividindo dados em treino/teste (70/30)...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )
    
    print(f"   Treino: {len(X_train)} amostras - {np.bincount(y_train)}")
    print(f"   Teste:  {len(X_test)} amostras - {np.bincount(y_test)}")
    
    # Experimento 1: Comparar estratégias de resampling
    results_strategies = compare_resampling_strategies(
        X_train, y_train, X_test, y_test
    )
    print_summary_table(results_strategies, "Comparação de Estratégias (mu)")
    
    # Experimento 2: Comparar tipos de complexidade
    results_complexity = compare_complexity_types(
        X_train, y_train, X_test, y_test
    )
    print_summary_table(results_complexity, "Comparação de Métricas de Complexidade")
    
    # Experimento 3: Uso avançado
    demonstrate_advanced_usage()
    
    # Conclusão
    print("\n\n" + "="*80)
    print("✅ DEMONSTRAÇÃO CONCLUÍDA COM SUCESSO!")
    print("="*80)
    print("\n📚 Principais aprendizados:")
    print("   1. O resampling guiado por complexidade melhora significativamente")
    print("      o desempenho em classes minoritárias (recall, F1)")
    print("   2. Diferentes valores de mu favorecem diferentes tipos de instâncias")
    print("   3. A escolha da métrica de complexidade impacta os resultados")
    print("   4. O código refatorado é fácil de usar e altamente configurável")
    print("\n🚀 Próximos passos:")
    print("   - Teste com seus próprios dados")
    print("   - Ajuste mu/sigma para otimizar para sua métrica de interesse")
    print("   - Experimente diferentes métricas de complexidade")
    print("   - Integre com seu pipeline de ML existente")
    print("\n")


if __name__ == "__main__":
    # Executar demonstração completa
    main()
