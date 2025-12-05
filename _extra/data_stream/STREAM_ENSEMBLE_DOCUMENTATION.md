# 🌊 Stream Learning Ensemble - Documentação Completa

## 📋 Visão Geral

O **StreamLearningEnsemble** é uma implementação avançada de aprendizado incremental para dados em stream que combina:

1. **Processamento Incremental** - Aprende com chunks de dados sem retraining completo
2. **Detecção de Drift** - Identifica mudanças no padrão dos dados
3. **Complexity-Guided Sampling** - Balanceamento inteligente de classes
4. **Sliding Window** - Gerenciamento eficiente de memória
5. **Adaptação Dinâmica** - Atualiza ensemble conforme necessário

---

## 🎯 Quando Usar

### ✅ Use Stream Ensemble Quando:
- Dados chegam continuamente (streams)
- Não pode armazenar todos os dados
- Conceitos mudam ao longo do tempo (drift)
- Precisa de predições em tempo real
- Classes desbalanceadas em streams

### ❌ Use Batch Ensemble Quando:
- Dataset completo disponível de uma vez
- Conceitos são estáveis
- Tem memória suficiente
- Pode retreinar periodicamente

---

## 🚀 Quick Start

### Instalação
```bash
pip install numpy pandas scikit-learn
```

### Uso Básico
```python
from stream_learning_ensemble import StreamLearningEnsemble
from sklearn.linear_model import SGDClassifier

# Criar ensemble
ensemble = StreamLearningEnsemble(
    n_estimators=10,
    base_estimator=SGDClassifier,
    chunk_size=100,
    window_size=1000,
    drift_detection='adwin',
    verbose=1,
    random_state=42
)

# Processar stream
for X_chunk, y_chunk in data_stream:
    if first_chunk:
        ensemble.partial_fit(X_chunk, y_chunk, classes=[0, 1])
    else:
        ensemble.partial_fit(X_chunk, y_chunk)
    
    # Predizer
    y_pred = ensemble.predict(X_chunk)
```

---

## 🏗️ Arquitetura

### Componentes Principais

```
┌─────────────────────────────────────────────────────────┐
│            StreamLearningEnsemble                       │
│                                                         │
│  ┌────────────────────────────────────────────┐       │
│  │         Sliding Window                      │       │
│  │  - Armazena dados recentes                 │       │
│  │  - Gerencia memória                        │       │
│  │  - max_size, FIFO                          │       │
│  └────────────────────────────────────────────┘       │
│                      │                                  │
│                      ▼                                  │
│  ┌────────────────────────────────────────────┐       │
│  │      Drift Detector (opcional)             │       │
│  │  - ADWIN / DDM / Page-Hinkley             │       │
│  │  - Detecta mudanças                        │       │
│  └────────────────────────────────────────────┘       │
│                      │                                  │
│                      ▼                                  │
│  ┌────────────────────────────────────────────┐       │
│  │   Complexity-Guided Sampler                │       │
│  │  - Balanceia classes                       │       │
│  │  - Usa métricas de complexidade            │       │
│  └────────────────────────────────────────────┘       │
│                      │                                  │
│         ┌────────────┴───────────┐                    │
│         ▼                        ▼                    │
│  ┌─────────────┐          ┌─────────────┐            │
│  │ Estimator 1 │   ...    │ Estimator N │            │
│  │  (μ=0.0)    │          │  (μ=1.0)    │            │
│  └─────────────┘          └─────────────┘            │
│         │                        │                    │
│         └────────────┬───────────┘                    │
│                      ▼                                  │
│           ┌─────────────────────┐                     │
│           │  Weighted Voting    │                     │
│           │  Final Prediction   │                     │
│           └─────────────────────┘                     │
└─────────────────────────────────────────────────────────┘
```

---

## 📊 Parâmetros de Configuração

### StreamEnsembleConfig

```python
@dataclass
class StreamEnsembleConfig:
    # Ensemble
    n_estimators: int = 10              # Máximo de classificadores
    base_estimator = SGDClassifier      # Deve ter partial_fit
    
    # Stream Processing
    chunk_size: int = 100               # Tamanho do chunk
    window_size: int = 1000             # Tamanho da janela
    
    # Complexity Sampling
    complexity_type: str = "overlap"    # Tipo de complexidade
    sigma: float = 0.2                  # Spread da gaussiana
    k_neighbors: int = 5                # Vizinhos para síntese
    
    # Drift Detection
    drift_detection: str = "adwin"      # none, adwin, ddm, page_hinkley
    drift_threshold: float = 0.1        # Sensibilidade
    
    # Update Strategy
    update_strategy: str = "replace_worst"  # add_new, weighted
    min_samples_before_update: int = 500    # Amostras mínimas
    rebalance_frequency: int = 5        # Chunks entre rebalanceamentos
    prune_threshold: float = 0.6        # Threshold para poda
    
    # Memory
    max_memory_mb: float = 100.0        # Limite de memória
    
    # Other
    verbose: int = 1                    # Verbosidade
    random_state: int = None            # Seed
```

---

## 🔬 Detecção de Drift

### Métodos Disponíveis

#### 1. ADWIN (Adaptive Windowing)
```python
ensemble = StreamLearningEnsemble(
    drift_detection='adwin',
    drift_threshold=0.002  # Menor = mais sensível
)
```

**Características:**
- Detecta mudanças na distribuição
- Sem parâmetros fixos de janela
- Bom para drifts graduais e abruptos

**Quando usar:** Uso geral, boa escolha padrão

#### 2. DDM (Drift Detection Method)
```python
ensemble = StreamLearningEnsemble(
    drift_detection='ddm',
    drift_threshold=0.1
)
```

**Características:**
- Monitora taxa de erro e desvio
- Detecta aumentos significativos no erro
- Rápido e eficiente

**Quando usar:** Drifts que aumentam erro

#### 3. Page-Hinkley
```python
ensemble = StreamLearningEnsemble(
    drift_detection='page_hinkley',
    drift_threshold=50.0
)
```

**Características:**
- Teste de soma acumulada
- Detecta mudanças na média
- Sensível a drifts abruptos

**Quando usar:** Mudanças súbitas

#### 4. Sem Detecção
```python
ensemble = StreamLearningEnsemble(
    drift_detection='none'
)
```

**Quando usar:** Quando sabe que não há drift ou quer economia computacional

---

## 🔄 Estratégias de Atualização

### 1. Replace Worst (Padrão)
```python
update_strategy='replace_worst'
```

**Como funciona:**
- Quando drift detectado, substitui pior classificador
- Novo classificador treina em dados recentes
- Mantém tamanho do ensemble fixo

**Vantagens:** Eficiente, tamanho constante  
**Desvantagens:** Pode perder conhecimento útil

### 2. Add New
```python
update_strategy='add_new'
```

**Como funciona:**
- Adiciona novo classificador ao detectar drift
- Ensemble cresce até n_estimators
- Não remove classificadores antigos

**Vantagens:** Preserva conhecimento histórico  
**Desvantagens:** Ensemble pode ficar grande

### 3. Weighted
```python
update_strategy='weighted'
```

**Como funciona:**
- Ajusta pesos baseado em performance
- Classificadores ruins têm menos influência
- Não remove ou adiciona

**Vantagens:** Suave, adaptativo  
**Desvantagens:** Classificadores ruins ainda consomem recursos

---

## 📈 Exemplos Avançados

### Exemplo 1: Stream com Concept Drift

```python
from stream_learning_ensemble import StreamLearningEnsemble, simulate_data_stream

# Criar ensemble
ensemble = StreamLearningEnsemble(
    n_estimators=10,
    drift_detection='adwin',
    update_strategy='replace_worst',
    verbose=1,
    random_state=42
)

# Simular stream com drifts em chunks 10 e 20
for i, (X, y) in enumerate(simulate_data_stream(
    n_chunks=30, 
    chunk_size=100,
    drift_points=[10, 20]
)):
    if i == 0:
        ensemble.partial_fit(X, y, classes=[0, 1])
    else:
        ensemble.partial_fit(X, y)

# Ver onde drifts foram detectados
drift_summary = ensemble.get_drift_summary()
print(f"Drifts detectados: {drift_summary['drift_points']}")
```

### Exemplo 2: Monitoramento de Performance

```python
# Processar stream
performances = []

for X, y in data_stream:
    ensemble.partial_fit(X, y)
    
    # Avaliar em chunk atual
    y_pred = ensemble.predict(X)
    acc = accuracy_score(y, y_pred)
    performances.append(acc)

# Plotar evolução
import matplotlib.pyplot as plt
plt.plot(performances)
plt.xlabel('Chunk')
plt.ylabel('Accuracy')
plt.title('Performance Over Time')
plt.show()
```

### Exemplo 3: Diferentes Base Estimators

```python
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import PassiveAggressiveClassifier

# Naive Bayes (rápido, probabilístico)
ensemble_nb = StreamLearningEnsemble(
    base_estimator=GaussianNB,
    n_estimators=10
)

# Passive Aggressive (bom para text/sparse)
ensemble_pa = StreamLearningEnsemble(
    base_estimator=PassiveAggressiveClassifier,
    n_estimators=10
)

# SGD (versátil, personalizável)
from sklearn.linear_model import SGDClassifier
ensemble_sgd = StreamLearningEnsemble(
    base_estimator=lambda: SGDClassifier(loss='log_loss', max_iter=10),
    n_estimators=10
)
```

### Exemplo 4: Ajuste Fino de Parâmetros

```python
# Para streams rápidos (alta frequência)
fast_stream = StreamLearningEnsemble(
    chunk_size=50,              # Chunks menores
    window_size=500,            # Janela menor
    rebalance_frequency=2,      # Rebalanceia mais
    drift_detection='adwin',    # Detecção sensível
    verbose=0                   # Menos output
)

# Para streams lentos (baixa frequência)
slow_stream = StreamLearningEnsemble(
    chunk_size=200,             # Chunks maiores
    window_size=2000,           # Janela maior
    rebalance_frequency=10,     # Rebalanceia menos
    drift_detection='ddm',      # Detecção menos sensível
    verbose=1
)

# Para dados muito desbalanceados
imbalanced_stream = StreamLearningEnsemble(
    complexity_type='overlap',   # Bom para imbalance
    sigma=0.3,                  # Maior spread
    rebalance_frequency=3,      # Rebalanceia frequente
    drift_detection='none'      # Foco no balanceamento
)
```

---

## 🎯 Casos de Uso Reais

### 1. Detecção de Fraude em Transações
```python
fraud_detector = StreamLearningEnsemble(
    n_estimators=15,
    chunk_size=1000,  # 1000 transações por chunk
    window_size=10000,
    complexity_type='error_rate',  # Prioriza casos difíceis
    drift_detection='adwin',  # Padrões de fraude mudam
    update_strategy='replace_worst',
    verbose=1
)

# Processar transações em tempo real
for transactions_batch in transaction_stream:
    fraud_detector.partial_fit(
        transactions_batch.drop('is_fraud', axis=1),
        transactions_batch['is_fraud']
    )
    
    # Detectar fraudes
    predictions = fraud_detector.predict(new_transactions)
```

### 2. Classificação de Sentimento em Redes Sociais
```python
sentiment_classifier = StreamLearningEnsemble(
    n_estimators=10,
    base_estimator=lambda: SGDClassifier(loss='hinge'),
    chunk_size=500,  # 500 posts por chunk
    drift_detection='page_hinkley',  # Tendências mudam rápido
    rebalance_frequency=5,
    verbose=0
)

# Processar posts contínuos
for posts_batch in social_media_stream:
    # Extrair features (TF-IDF, embeddings, etc)
    X_features = feature_extractor.transform(posts_batch['text'])
    y_sentiment = posts_batch['sentiment']
    
    sentiment_classifier.partial_fit(X_features, y_sentiment)
```

### 3. Manutenção Preditiva em IoT
```python
maintenance_predictor = StreamLearningEnsemble(
    n_estimators=12,
    chunk_size=100,  # 100 leituras de sensores
    window_size=1000,
    complexity_type='neighborhood',  # Padrões espaciais
    drift_detection='ddm',  # Degradação gradual
    update_strategy='weighted',  # Preserva histórico
    verbose=1
)

# Processar leituras de sensores
for sensor_readings in iot_stream:
    maintenance_predictor.partial_fit(
        sensor_readings[['temperature', 'vibration', 'pressure']],
        sensor_readings['needs_maintenance']
    )
    
    # Predizer necessidade de manutenção
    predictions = maintenance_predictor.predict(current_readings)
```

---

## ⚡ Performance e Otimização

### Recomendações por Tamanho de Stream

#### Pequeno Stream (<10k samples/hora)
```python
StreamLearningEnsemble(
    n_estimators=5,
    chunk_size=100,
    window_size=500,
    rebalance_frequency=5,
    drift_detection='ddm'
)
```

#### Médio Stream (10k-100k samples/hora)
```python
StreamLearningEnsemble(
    n_estimators=10,
    chunk_size=200,
    window_size=1000,
    rebalance_frequency=10,
    drift_detection='adwin'
)
```

#### Grande Stream (>100k samples/hora)
```python
StreamLearningEnsemble(
    n_estimators=15,
    chunk_size=500,
    window_size=2000,
    rebalance_frequency=20,
    drift_detection='none',  # Economia
    verbose=0
)
```

### Dicas de Otimização

1. **Reduzir rebalance_frequency** para streams rápidos
2. **Usar drift_detection='none'** se não espera mudanças
3. **Diminuir n_estimators** para predições mais rápidas
4. **Aumentar chunk_size** para processar mais de uma vez
5. **Usar base_estimator eficiente** (GaussianNB é rápido)

---

## 🐛 Troubleshooting

### Problema: Memory Error
**Solução:**
```python
# Reduzir window_size
ensemble = StreamLearningEnsemble(window_size=500)

# Ou aumentar chunk_size (processa menos frequente)
ensemble = StreamLearningEnsemble(chunk_size=500)
```

### Problema: Performance Degradando
**Solução:**
```python
# Ativar drift detection
ensemble = StreamLearningEnsemble(drift_detection='adwin')

# Aumentar rebalance_frequency
ensemble = StreamLearningEnsemble(rebalance_frequency=3)

# Ativar pruning agressivo
ensemble = StreamLearningEnsemble(prune_threshold=0.7)
```

### Problema: Muitos Falsos Alarmes de Drift
**Solução:**
```python
# Aumentar threshold
ensemble = StreamLearningEnsemble(
    drift_detection='adwin',
    drift_threshold=0.01  # Menos sensível
)

# Ou trocar método
ensemble = StreamLearningEnsemble(drift_detection='ddm')
```

### Problema: Processamento Lento
**Solução:**
```python
# Simplificar ensemble
ensemble = StreamLearningEnsemble(
    n_estimators=5,  # Menos estimadores
    complexity_type='overlap',  # Mais rápido que error_rate
    rebalance_frequency=10,  # Rebalanceia menos
    drift_detection='none'  # Desativar se não precisa
)
```

---

## 📊 Comparação com Outros Métodos

| Característica | Stream Ensemble | Batch Ensemble | Online SVM |
|----------------|----------------|----------------|------------|
| Memória | ⭐⭐⭐⭐⭐ | ⭐⭐ | ⭐⭐⭐⭐ |
| Velocidade | ⭐⭐⭐⭐ | ⭐⭐⭐ | ⭐⭐⭐⭐⭐ |
| Accuracy | ⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐⭐ |
| Drift Handling | ⭐⭐⭐⭐⭐ | ⭐ | ⭐⭐⭐ |
| Imbalance | ⭐⭐⭐⭐⭐ | ⭐⭐⭐ | ⭐⭐ |

---

## 🎓 Conclusão

O StreamLearningEnsemble é ideal para:
- ✅ Dados em fluxo contínuo
- ✅ Concept drift
- ✅ Classes desbalanceadas
- ✅ Restrições de memória
- ✅ Predições em tempo real

**Próximos Passos:**
1. Execute `demo_stream_ensemble.py` para ver exemplos
2. Execute `test_stream_ensemble.py` para validar
3. Adapte para seus dados
4. Ajuste parâmetros conforme necessidade

**Código pronto para produção!** 🚀
