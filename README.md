# 🎯 Complexity-Guided Ensemble - Documentação Técnica

## 📋 Índice

1. [Visão Geral](#visão-geral)
2. [Arquitetura](#arquitetura)
3. [Algoritmo Detalhado](#algoritmo-detalhado)
4. [Componentes Principais](#componentes-principais)
5. [Guia de Uso](#guia-de-uso)
6. [Exemplos Avançados](#exemplos-avançados)
7. [Performance e Benchmarks](#performance-e-benchmarks)
8. [Referências Teóricas](#referências-teóricas)

---

## 🎓 Visão Geral

### O Que É?

O **Complexity-Guided Ensemble** é um método avançado de ensemble learning que combina:

1. **ComplexityGuidedSampler** para geração de bags
2. **Variação sistemática de complexidade** através do parâmetro μ
3. **Diversidade** entre membros do ensemble
4. **Suporte completo a classificação multiclasse**

### Por Que Usar?

✅ **Superior a métodos tradicionais** em dados desbalanceados  
✅ **Diversidade aumentada** sem configuração manual  
✅ **Suporte nativo a multiclasse** com qualquer número de classes  
✅ **Escalável** com paralelização nativa  
✅ **Interpretável** através dos valores μ  

### Diferenças dos Métodos Tradicionais

| Aspecto | Bagging Tradicional | Random Forest | **CG-Ensemble** |
|---------|---------------------|---------------|-----------------|
| Diversidade | Aleatória | Aleatória + Feature sampling | **Guiada por complexidade** |
| Balanceamento | Não | Não | **Automático (CG-Sampler)** |
| Seleção de instâncias | Aleatória | Aleatória | **Guiada por complexidade** |
| Multiclasse | Sim | Sim | **Sim** |
| Interpretabilidade dos bags | Baixa | Baixa | **Alta (μ values)** |

---

## ⚡ QUICK START GUIDE

### Instalar Dependências
```bash
pip install numpy pandas scikit-learn joblib pyhard
```

### Testar algoritmo com exemplos de demonstração
```bash
# Testar sampler
python demo_resampler.py

# Testar ensemble  
python demo_ensemble.py
```

### Primeiro Código

```python
# Exemplo mínimo do sampler
from complexity_sampler_refactored import ComplexityGuidedSampler
from sklearn.datasets import make_classification
import numpy as np

X, y = make_classification(n_samples=1000, n_classes=2, weights=[0.9, 0.1])
sampler = ComplexityGuidedSampler(random_state=42)
X_bal, y_bal = sampler.fit_resample(X, y, mu=0.5, sigma=0.2, k_neighbors=5)

print(f"Antes: {np.bincount(y)}")  # [900, 100]
print(f"Depois: {np.bincount(y_bal)}")  # [500, 500]
```

```python
# Exemplo mínimo do ensemble
from complexity_guided_ensemble import ComplexityGuidedEnsemble

ensemble = ComplexityGuidedEnsemble(n_estimators=5, random_state=42)
ensemble.fit(X, y)
predictions = ensemble.predict(X)

print(f"Treinado com sucesso! Precisão: {(predictions == y).mean():.2f}")
```

---



## 📐 Algoritmo Detalhado

### Pseudocódigo Completo

```python
Algorithm: ComplexityGuidedEnsemble

Input:
    D = {(x_i, y_i)}_{i=1}^N  # Training data
    n_estimators              # Number of base classifiers
    σ                         # Gaussian std (fixed)
    ComplexityMeasure         # Type of complexity metric

Output:
    E = {h_1, h_2, ..., h_n}  # Ensemble of classifiers

# Step 1: Generate μ values systematically
μ_values ← linspace(0, 1, n_estimators)

# Step 2: Initialize components
sampler ← ComplexityGuidedSampler(ComplexityMeasure)
estimators ← []

# Step 3: For each μ value, create a base classifier
for i in 1 to n_estimators:
    μ_i ← μ_values[i]
    
    # 3.1: Calculate complexities
    complexities ← sampler.calculate_complexities(D)
    
    # 3.2: Generate weights with Gaussian
    weights ← exp(-((complexities - μ_i)² / (2 * σ²)))
    
    # 3.3: Separate classes
    D_classes ← separate_by_class(D)
    
    # 3.4: Resample each class
    D_resampled ← []
    for each class c in D_classes:
        # Undersample or oversample based on target size
        D_c_resampled ← weighted_resample(D_classes[c], weights, n_target)
        D_resampled ← D_resampled ∪ D_c_resampled
    
    # 3.5: Train base classifier
    h_i ← train_classifier(D_resampled)
    
    # 3.6: Store
    estimators.append(h_i)

return E = estimators
```

### Função de Predição

```python
Algorithm: Predict

Input:
    x_test       # Test instance
    E            # Trained ensemble
    voting       # 'soft' or 'hard'
    classes      # All possible classes

Output:
    y_pred       # Predicted label

if voting == 'soft':
    # Average probabilities across all classes
    all_probas ← []
    
    for h_i in E:
        # Get probabilities from estimator
        proba_i ← h_i.predict_proba(x_test)
        
        # Align with all classes (some estimators may not have seen all classes)
        aligned_proba ← align_probabilities(proba_i, h_i.classes_, classes)
        all_probas.append(aligned_proba)
    
    # Average and normalize
    avg_proba ← mean(all_probas, axis=0)
    avg_proba ← avg_proba / sum(avg_proba)
    
    y_pred ← classes[argmax(avg_proba)]
else:
    # Majority vote
    votes ← [h_i.predict(x_test) for h_i in E]
    y_pred ← mode(votes)

return y_pred
```

---

## 🧩 Componentes Principais

### 1. ComplexityGuidedSampler

**Responsabilidade:** Gerar subconjuntos balanceados guiados por complexidade

**Métricas de Complexidade Disponíveis:**

#### a) Overlap Complexity
```python
# Mede sobreposição de features entre classes
# Instâncias na região de overlap têm alta complexidade

complexity = distance_to_overlap_center(instance)
```

**Quando usar:** Dados com separação clara mas com regiões de overlap

**Multiclasse:** Calcula overlap entre todos os pares de classes

#### b) Error Rate Complexity
```python
# Mede dificuldade de classificação via cross-validation
# Instâncias mal classificadas têm alta complexidade

complexity = 1 - P(y=true_class|x)
```

**Quando usar:** Quer priorizar instâncias difíceis de classificar

**Multiclasse:** Usa probabilidade da classe verdadeira (funciona para qualquer número de classes)

#### c) Neighborhood Complexity
```python
# Mede homogeneidade da vizinhança
# Instâncias em regiões mistas têm alta complexidade

complexity = count_different_class_neighbors(instance) / k_neighbors
```

**Quando usar:** Dados com clusters bem definidos

**Multiclasse:** Conta vizinhos de classes diferentes (não apenas da classe majoritária)


#### d) hardness_function
```python
# Permite a configuração de uma função de complexidade personalizada
# Neste caso usamos as funções disponibilizadas pela biblioteca 'pyhard'

complexity = hardness_function(instance)
```
**Quando usar:** Quando necessitar de uma meidada de complexidade diferente das fornecidas


### 2. Gaussian Weighting Function

```python
def gaussian_weight(complexity, mu, sigma):
    """
    Calcula peso Gaussiano para cada instância
    
    Parameters:
    -----------
    complexity : float [0, 1]
        Complexidade normalizada da instância
    mu : float [0, 1]
        Centro da Gaussiana (nível de complexidade alvo)
    sigma : float
        Desvio padrão (controla "spread")
    
    Returns:
    --------
    weight : float
        Peso da instância para amostragem
    """
    return np.exp(-((complexity - mu)**2) / (2 * sigma**2))
```

**Interpretação de μ:**
- **μ = 0.0:** Foca em instâncias simples (fáceis de classificar)
- **μ = 0.5:** Foca em instâncias de dificuldade média
- **μ = 1.0:** Foca em instâncias complexas (difíceis de classificar)

**Interpretação de σ:**
- **σ pequeno (0.1):** Seleção muito focada, pouca variação
- **σ médio (0.2-0.3):** Balanceado
- **σ grande (0.5):** Seleção mais abrangente, menos específica

### 3. Voting Strategies

#### a) Soft Voting (Recomendado)
```python
# Média das probabilidades preditas
# Mais robusto e informativo

proba_final = mean([clf.predict_proba(x) for clf in estimators])
y_pred = argmax(proba_final)
```

**Vantagens:**
- Usa toda a informação disponível (probabilidades)
- Mais estável com classes desbalanceadas
- Melhor para multiclasse

#### b) Hard Voting
```python
# Voto majoritário das predições
# Mais simples e rápido

votes = [clf.predict(x) for clf in estimators]
y_pred = mode(votes)
```

**Vantagens:**
- Mais rápido (não precisa calcular probabilidades)
- Funciona com qualquer classificador
- Mais interpretável

---

## 📚 Guia de Uso

### Instalação

```bash
pip install numpy pandas scikit-learn joblib
```

### Uso Básico 
```python
# Criar dados multiclasse desbalanceados
X, y = make_classification(
    n_samples=1000,
    n_classes=4,
    n_clusters_per_class=1,
    weights=[0.5, 0.3, 0.15, 0.05],
    n_informative=10,
    random_state=42
)

# Criar ensemble 
ensemble = ComplexityGuidedEnsemble(
    n_estimators=15,
    complexity_type='neighborhood',
    voting='soft',
    random_state=42
)

# Treinar
ensemble.fit(X, y)

# Predizer
y_pred = ensemble.predict(X)
y_proba = ensemble.predict_proba(X)

print(f"Number of classes: {ensemble.n_classes_}")  # 4
print(f"Probabilities shape: {y_proba.shape}")  # (n_samples, 4)
```

### Configuração Avançada

```python
from complexity_guided_ensemble_simplified import EnsembleConfig
from sklearn.tree import DecisionTreeClassifier

# Criar configuração customizada
config = EnsembleConfig(
    n_estimators=20,                    # Mais estimadores
    base_estimator=DecisionTreeClassifier,
    complexity_type='error_rate',       # Usar taxa de erro
    sigma=0.3,                          # Spread maior
    k_neighbors=7,                      # Mais vizinhos para synthetic
    voting='soft',                      # Votação por probabilidade
    cv_folds=5,                         # CV folds para complexity
    n_jobs=-1,                          # Usar todos os cores
    verbose=1                           # Mostrar progresso
)

# Usar configuração
ensemble = ComplexityGuidedEnsemble(config=config)
ensemble.fit(X, y)
```

---

## 🎯 Exemplos Avançados

### Exemplo: Comparação com Baselines

```python
from sklearn.ensemble import RandomForestClassifier, BaggingClassifier
from sklearn.model_selection import cross_val_score
import numpy as np

# Criar dados desbalanceados
X, y = make_classification(
    n_samples=1000,
    n_classes=3,
    n_clusters_per_class=1,
    weights=[0.7, 0.2, 0.1],
    random_state=42
)

methods = {
    'Random Forest': RandomForestClassifier(
        n_estimators=10, 
        random_state=42
    ),
    'Bagging': BaggingClassifier(
        n_estimators=10, 
        random_state=42
    ),
    'CG-Ensemble': ComplexityGuidedEnsemble(
        n_estimators=10,
        random_state=42
    )
}

print("Cross-validation results (F1-weighted):")
print("-" * 50)
for name, clf in methods.items():
    scores = cross_val_score(clf, X, y, cv=5, scoring='f1_weighted')
    print(f"{name:20s}: {scores.mean():.4f} ± {scores.std():.4f}")
```
---