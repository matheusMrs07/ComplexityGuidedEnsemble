"""
Demonstração Completa: Stream Learning Ensemble

Este script demonstra o uso do ensemble para aprendizado em streams de dados,
incluindo:
- Processamento incremental de chunks
- Detecção de concept drift
- Adaptação dinâmica
- Visualização de performance ao longo do tempo
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import SGDClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, f1_score

from stream_learning_ensemble import (
    StreamLearningEnsemble,
    StreamEnsembleConfig,
    simulate_data_stream
)


def demo_basic_stream_learning():
    """Demonstração básica de stream learning."""
    print("\n" + "="*80)
    print("🌊 DEMO 1: Stream Learning Básico")
    print("="*80)
    
    # Criar ensemble
    ensemble = StreamLearningEnsemble(
        n_estimators=5,
        base_estimator=SGDClassifier,
        chunk_size=100,
        window_size=500,
        drift_detection='adwin',
        rebalance_frequency=3,
        verbose=1,
        random_state=42
    )
    
    # Processar stream
    print("\n📊 Processando stream de dados...")
    
    n_chunks = 20
    for i, (X_chunk, y_chunk) in enumerate(simulate_data_stream(
        n_chunks=n_chunks,
        chunk_size=100,
        drift_points=[7, 14]
    )):
        if i == 0:
            ensemble.partial_fit(X_chunk, y_chunk, classes=np.array([0, 1]))
        else:
            ensemble.partial_fit(X_chunk, y_chunk)
    
    # Resultados
    print("\n" + "="*80)
    print("📈 RESULTADOS FINAIS")
    print("="*80)
    
    summary = ensemble.get_performance_summary()
    print(f"\n✅ Total de chunks processados: {len(summary)}")
    print(f"✅ Total de amostras vistas: {ensemble.n_samples_seen_}")
    print(f"✅ Accuracy média: {summary['accuracy'].mean():.4f}")
    print(f"✅ F1 Score médio: {summary['f1_score'].mean():.4f}")
    
    drift_summary = ensemble.get_drift_summary()
    print(f"\n🚨 Drifts detectados: {drift_summary['n_drifts_detected']}")
    if drift_summary['drift_points']:
        print(f"   Pontos de drift: {drift_summary['drift_points']}")
    
    return ensemble, summary


def demo_drift_comparison():
    """Comparar diferentes métodos de detecção de drift."""
    print("\n" + "="*80)
    print("🔬 DEMO 2: Comparação de Detecção de Drift")
    print("="*80)
    
    drift_methods = ['none', 'adwin', 'ddm', 'page_hinkley']
    results = {}
    
    for method in drift_methods:
        print(f"\n📊 Testando: {method}")
        print("-"*80)
        
        ensemble = StreamLearningEnsemble(
            n_estimators=5,
            base_estimator=SGDClassifier,
            chunk_size=100,
            drift_detection=method,
            min_samples_before_update=200,
            verbose=0,
            random_state=42
        )
        
        # Processar stream com drifts conhecidos
        drift_points = [10, 20, 30]
        for i, (X, y) in enumerate(simulate_data_stream(
            n_chunks=40,
            chunk_size=100,
            drift_points=drift_points
        )):
            if i == 0:
                ensemble.partial_fit(X, y, classes=np.array([0, 1]))
            else:
                ensemble.partial_fit(X, y)
        
        summary = ensemble.get_performance_summary()
        drift_summary = ensemble.get_drift_summary()
        
        results[method] = {
            'accuracy': summary['accuracy'].mean(),
            'f1_score': summary['f1_score'].mean(),
            'drifts_detected': drift_summary['n_drifts_detected'],
            'true_drifts': len(drift_points)
        }
        
        print(f"   Accuracy: {results[method]['accuracy']:.4f}")
        print(f"   Drifts detectados: {results[method]['drifts_detected']}")
    
    # Tabela comparativa
    print("\n" + "="*80)
    print("📊 TABELA COMPARATIVA")
    print("="*80)
    
    df_results = pd.DataFrame(results).T
    print(df_results.to_string(float_format=lambda x: f'{x:.4f}'))
    
    return results


def demo_ensemble_evolution():
    """Demonstrar evolução do ensemble ao longo do tempo."""
    print("\n" + "="*80)
    print("📈 DEMO 3: Evolução do Ensemble")
    print("="*80)
    
    # Configurações para testar
    configs = {
        'Conservative (No Pruning)': {
            'update_strategy': 'weighted',
            'prune_threshold': 0.0  # Never prune
        },
        'Aggressive (Active Pruning)': {
            'update_strategy': 'replace_worst',
            'prune_threshold': 0.7
        },
        'Adaptive (Add New)': {
            'update_strategy': 'add_new',
            'prune_threshold': 0.6
        }
    }
    
    results = {}
    
    for name, params in configs.items():
        print(f"\n📊 Testando: {name}")
        print("-"*80)
        
        ensemble = StreamLearningEnsemble(
            n_estimators=5,
            base_estimator=SGDClassifier,
            chunk_size=100,
            min_samples_before_update=200,
            verbose=0,
            random_state=42,
            **params
        )
        
        # Processar stream
        for i, (X, y) in enumerate(simulate_data_stream(n_chunks=30)):
            if i == 0:
                ensemble.partial_fit(X, y, classes=np.array([0, 1]))
            else:
                ensemble.partial_fit(X, y)
        
        summary = ensemble.get_performance_summary()
        results[name] = summary
        
        print(f"   Final accuracy: {summary['accuracy'].iloc[-1]:.4f}")
        print(f"   Final ensemble size: {summary['n_estimators'].iloc[-1]}")
    
    return results


def demo_memory_efficiency():
    """Demonstrar eficiência de memória."""
    print("\n" + "="*80)
    print("💾 DEMO 4: Eficiência de Memória")
    print("="*80)
    
    window_sizes = [500, 1000, 2000]
    
    for window_size in window_sizes:
        print(f"\n📊 Window size: {window_size}")
        print("-"*80)
        
        ensemble = StreamLearningEnsemble(
            n_estimators=5,
            window_size=window_size,
            chunk_size=100,
            verbose=0,
            random_state=42
        )
        
        # Processar múltiplos chunks
        for i, (X, y) in enumerate(simulate_data_stream(n_chunks=50)):
            if i == 0:
                ensemble.partial_fit(X, y, classes=np.array([0, 1]))
            else:
                ensemble.partial_fit(X, y)
        
        # Estatísticas de memória
        window = ensemble.sliding_window_
        memory_mb = window.get_memory_usage_mb()
        
        print(f"   Samples in window: {window.get_size()}")
        print(f"   Memory usage: {memory_mb:.2f} MB")
        print(f"   Total samples seen: {ensemble.n_samples_seen_}")


def visualize_stream_performance(ensemble, summary):
    """Visualizar performance ao longo do stream."""
    print("\n" + "="*80)
    print("📊 GERANDO VISUALIZAÇÕES")
    print("="*80)
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle('Stream Learning Performance Over Time', 
                 fontsize=16, fontweight='bold')
    
    # 1. Accuracy over time
    ax1 = axes[0, 0]
    ax1.plot(summary['chunk'], summary['accuracy'], 'b-', linewidth=2, label='Accuracy')
    ax1.plot(summary['chunk'], summary['f1_score'], 'g--', linewidth=2, label='F1 Score')
    
    # Mark drift points
    drift_points = summary[summary['drift_detected']]['chunk']
    for dp in drift_points:
        ax1.axvline(x=dp, color='r', linestyle=':', alpha=0.5)
    
    ax1.set_xlabel('Chunk')
    ax1.set_ylabel('Score')
    ax1.set_title('Performance Metrics Over Time')
    ax1.legend()
    ax1.grid(alpha=0.3)
    
    # 2. Number of estimators
    ax2 = axes[0, 1]
    ax2.plot(summary['chunk'], summary['n_estimators'], 'purple', linewidth=2)
    ax2.set_xlabel('Chunk')
    ax2.set_ylabel('Number of Estimators')
    ax2.set_title('Ensemble Size Evolution')
    ax2.grid(alpha=0.3)
    
    # 3. Individual estimator performance (last chunk)
    ax3 = axes[1, 0]
    if len(summary) > 0 and 'estimator_scores' in summary.columns:
        last_scores = summary.iloc[-1]['estimator_scores']
        if isinstance(last_scores, list):
            ax3.bar(range(len(last_scores)), last_scores, color='skyblue')
            ax3.axhline(y=summary['accuracy'].mean(), color='r', 
                       linestyle='--', label='Ensemble Avg')
            ax3.set_xlabel('Estimator Index')
            ax3.set_ylabel('Accuracy')
            ax3.set_title('Individual Estimator Performance (Final Chunk)')
            ax3.legend()
            ax3.grid(alpha=0.3)
    
    # 4. Cumulative samples
    ax4 = axes[1, 1]
    ax4.plot(summary['chunk'], summary['n_samples'], 'orange', linewidth=2)
    ax4.set_xlabel('Chunk')
    ax4.set_ylabel('Cumulative Samples')
    ax4.set_title('Total Samples Processed')
    ax4.grid(alpha=0.3)
    
    plt.tight_layout()
    
    try:
        plt.savefig('stream_ensemble_performance.png', dpi=300, bbox_inches='tight')
        print("✅ Gráfico salvo: stream_ensemble_performance.png")
    except:
        print("⚠️ Não foi possível salvar o gráfico")
    
    try:
        plt.show()
    except:
        print("⚠️ Display não disponível")


def compare_with_batch_learning():
    """Comparar stream learning com batch learning."""
    print("\n" + "="*80)
    print("⚖️  DEMO 5: Stream vs Batch Learning")
    print("="*80)
    
    from sklearn.ensemble import RandomForestClassifier
    
    # Coletar todos os dados
    print("\n📊 Coletando dados do stream...")
    X_all, y_all = [], []
    for X, y in simulate_data_stream(n_chunks=20, drift_points=[10]):
        X_all.append(X)
        y_all.append(y)
    
    X_all = np.vstack(X_all)
    y_all = np.concatenate(y_all)
    
    # Split for testing
    split_point = int(0.8 * len(X_all))
    X_train, X_test = X_all[:split_point], X_all[split_point:]
    y_train, y_test = y_all[:split_point], y_all[split_point:]
    
    # 1. Batch Learning (Random Forest)
    print("\n1️⃣ Batch Learning (Random Forest)...")
    rf = RandomForestClassifier(n_estimators=10, random_state=42)
    rf.fit(X_train, y_train)
    y_pred_batch = rf.predict(X_test)
    acc_batch = accuracy_score(y_test, y_pred_batch)
    f1_batch = f1_score(y_test, y_pred_batch)
    
    print(f"   Accuracy: {acc_batch:.4f}")
    print(f"   F1 Score: {f1_batch:.4f}")
    
    # 2. Stream Learning
    print("\n2️⃣ Stream Learning (Incremental)...")
    stream_ensemble = StreamLearningEnsemble(
        n_estimators=10,
        base_estimator=SGDClassifier,
        chunk_size=100,
        drift_detection='adwin',
        verbose=0,
        random_state=42
    )
    
    # Train on chunks
    chunk_size = 100
    for i in range(0, split_point, chunk_size):
        X_chunk = X_all[i:i+chunk_size]
        y_chunk = y_all[i:i+chunk_size]
        
        if i == 0:
            stream_ensemble.partial_fit(X_chunk, y_chunk, classes=np.array([0, 1]))
        else:
            stream_ensemble.partial_fit(X_chunk, y_chunk)
    
    y_pred_stream = stream_ensemble.predict(X_test)
    acc_stream = accuracy_score(y_test, y_pred_stream)
    f1_stream = f1_score(y_test, y_pred_stream)
    
    print(f"   Accuracy: {acc_stream:.4f}")
    print(f"   F1 Score: {f1_stream:.4f}")
    
    # Comparação
    print("\n" + "="*80)
    print("📊 COMPARAÇÃO")
    print("="*80)
    
    comparison = pd.DataFrame({
        'Method': ['Batch (RF)', 'Stream (Ensemble)'],
        'Accuracy': [acc_batch, acc_stream],
        'F1 Score': [f1_batch, f1_stream]
    })
    
    print(comparison.to_string(index=False, float_format=lambda x: f'{x:.4f}'))
    
    print("\n💡 Observações:")
    print("   - Stream learning adapta-se a concept drift")
    print("   - Batch learning pode ser mais preciso em dados estáticos")
    print("   - Stream learning é mais eficiente em memória")


def main():
    """Executar todas as demonstrações."""
    print("\n" + "="*80)
    print("🌊 DEMONSTRAÇÃO COMPLETA - STREAM LEARNING ENSEMBLE")
    print("="*80)
    print("\nEste script demonstra:")
    print("  ✅ Aprendizado incremental em streams")
    print("  ✅ Detecção e adaptação a concept drift")
    print("  ✅ Múltiplas estratégias de atualização")
    print("  ✅ Eficiência de memória")
    print("  ✅ Comparação com batch learning")
    
    # Demo 1: Básico
    ensemble, summary = demo_basic_stream_learning()
    
    # Demo 2: Drift detection
    drift_results = demo_drift_comparison()
    
    # Demo 3: Ensemble evolution
    evolution_results = demo_ensemble_evolution()
    
    # Demo 4: Memory efficiency
    demo_memory_efficiency()
    
    # Demo 5: Stream vs Batch
    compare_with_batch_learning()
    
    # Visualização
    try:
        visualize_stream_performance(ensemble, summary)
    except Exception as e:
        print(f"\n⚠️ Não foi possível gerar visualizações: {e}")
    
    # Conclusão
    print("\n" + "="*80)
    print("✅ DEMONSTRAÇÃO CONCLUÍDA COM SUCESSO!")
    print("="*80)
    
    print("\n📚 Principais Aprendizados:")
    print("   1. Stream learning processa dados incrementalmente")
    print("   2. Detecta e adapta-se a mudanças nos dados (drift)")
    print("   3. Mantém performance com memória limitada")
    print("   4. Balanceia dados usando complexity-guided sampling")
    print("   5. Competitivo com batch learning em cenários dinâmicos")
    
    print("\n🚀 Próximos Passos:")
    print("   - Testar com seus streams de dados reais")
    print("   - Ajustar window_size e chunk_size")
    print("   - Experimentar diferentes drift detectors")
    print("   - Customizar update_strategy conforme necessidade")
    
    print()


if __name__ == "__main__":
    main()
