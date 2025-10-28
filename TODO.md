# ClusterTK - TODO

## Текущий статус: v0.5.0 🚀

**Latest Release:** v0.5.0
**PyPI:** https://pypi.org/project/clustertk/
**GitHub:** https://github.com/alexeiveselov92/clustertk

## ✅ Что завершено

### v0.1.0-v0.1.1 - Базовый функционал
- [x] Полный pipeline кластерного анализа
- [x] Preprocessing (4 модуля)
- [x] Feature selection (2 модуля)
- [x] Dimensionality reduction (PCA, t-SNE, UMAP)
- [x] Clustering (KMeans, GMM)
- [x] Evaluation (метрики + optimal k finder)
- [x] Interpretation (ClusterProfiler)
- [x] Публикация на PyPI

### v0.2.0 - Дополнительные алгоритмы
- [x] HierarchicalClustering (Ward, Complete, Average linkage)
- [x] DBSCANClustering с автоподбором параметров (eps, min_samples)

### v0.3.0 - Visualization & Naming
- [x] 11 visualization функций в 4 модулях
- [x] Интеграция с Pipeline (.plot_*() методы)
- [x] ClusterNamer с 3 стратегиями (top_features, categories, combined)
- [x] Условные импорты для опциональных зависимостей

### v0.3.1-v0.3.5 - Bugfixes & Improvements
- [x] v0.3.1: Hotfix README (plot_cluster_profiles → plot_cluster_heatmap)
- [x] v0.3.2: Исправлена нормализация для малого числа кластеров
- [x] v0.3.3-v0.3.5: Улучшения в visualization модуле

### v0.4.0-v0.4.2 - Visualization Fix & Documentation
- [x] v0.4.0: Попытка исправить дублирование (неудачная)
- [x] v0.4.1: Правильное решение через plt.close(fig)
- [x] v0.4.2: Обновлен Quick Start + навигация в README
- [x] Исследована причина дублирования (pyplot global state)
- [x] Изучен подход seaborn (возвращает Axes, не Figure)
- [x] Добавлены Table of Contents и Quick Links
- [x] Pipeline Components теперь в collapsed section

### v0.5.0 - Export & Reports ✅
- [x] Реализован `export_results()` для CSV формата
- [x] Реализован `export_results()` для JSON формата
- [x] Реализован `export_report()` - HTML отчёты с embedded plots
- [x] Реализованы `save_pipeline()` и `load_pipeline()` методы
- [x] Добавлена зависимость joblib
- [x] Обновлён README с примерами экспорта
- [x] Протестирован весь функционал экспорта

## 🎯 Приоритеты для v0.6.0

### HIGH PRIORITY (Must-have для v0.6.0)

1. **Documentation Structure** 🔥
   - [ ] Создать docs/ директорию с полной структурой
   - [ ] Сократить README до ~100-150 строк (Quick Start + ссылки на docs)
   - [ ] Создать отдельные markdown файлы:
     - docs/installation.md
     - docs/quickstart.md
     - docs/user_guide/ (preprocessing, clustering, evaluation, visualization, export)
     - docs/api_reference.md
     - docs/examples.md
     - docs/faq.md
   - [ ] Настроить GitHub Pages или MkDocs
   - **Обоснование:** README уже 426 строк и продолжит расти. Сейчас лучшее время для реструктуризации.

2. **Algorithm Comparison & Selection** 🔥
   - [ ] Метод `pipeline.compare_algorithms()` - автоматическое сравнение KMeans/GMM/Hierarchical/DBSCAN
   - [ ] Таблица с метриками для каждого алгоритма
   - [ ] Рекомендация лучшего алгоритма на основе данных
   - [ ] Визуализация сравнения алгоритмов
   - **Обоснование:** Пользователи часто не знают какой алгоритм выбрать - это решит проблему.

### MEDIUM PRIORITY (Желательно для v0.6.0)

3. **Enhanced Feature Analysis**
   - [ ] Расширенный feature importance (не только top features)
     - SHAP values для интерпретации
     - Permutation importance
     - Feature contribution to cluster separation
   - [ ] Anomaly detection внутри кластеров
     - Local Outlier Factor per cluster
     - Isolation Forest per cluster
     - Визуализация аномалий
   - **Обоснование:** Практически полезно для реального анализа данных.

4. **More Clustering Algorithms**
   - [ ] HDBSCAN (hierarchical DBSCAN) - очень популярен
   - [ ] Spectral Clustering
   - [ ] Mini-Batch K-Means (для больших данных)
   - **Обоснование:** HDBSCAN особенно востребован сообществом.

5. **Tests**
   - [ ] Unit tests для критичных модулей (preprocessing, clustering, evaluation)
   - [ ] Integration tests для Pipeline
   - [ ] Тесты для export функционала
   - [ ] Цель: покрытие >50% для v0.6.0

### LOW PRIORITY (Backlog)

6. **CI/CD**
   - [ ] GitHub Actions для автоматических тестов
   - [ ] Автоматическая проверка code style (black, flake8)
   - [ ] Coverage reports
   - [ ] Автоматическая публикация на PyPI при release tag

7. **Advanced Features**
   - [ ] Ensemble clustering (voting между алгоритмами)
   - [ ] Автоматический feature engineering (polynomial, interactions)
   - [ ] Stability analysis (bootstrap resampling)
   - [ ] Time series clustering support

## 📋 Backlog (для будущих версий)

- Полная Sphinx документация с API reference
- Оптимизация производительности для больших данных
- GPU support (cuML integration)
- Incremental/streaming clustering
- Time series clustering support
- Interactive web UI

## 🐛 Известные проблемы

Нет критических проблем в текущей версии.

## 💡 Идеи для улучшения

1. **User Experience:**
   - Прогресс-бар для долгих операций (tqdm)
   - Более информативные error messages
   - Warnings для потенциально проблемных настроек

2. **Performance:**
   - Кэширование промежуточных результатов
   - Параллелизация где возможно
   - Опциональная поддержка Dask для больших данных

3. **Features:**
   - Автоматический feature engineering
   - Anomaly detection внутри кластеров
   - Stability analysis (bootstrap resampling)
   - Cluster evolution tracking (для временных данных)

## 📝 Заметки по разработке

### Приоритизация
- **v0.4.0-v0.4.2** ✅ - исправлено дублирование графиков + улучшена навигация
- **v0.5.0** ✅ - export/reports (CSV, JSON, HTML, save/load)
- **v0.6.0** 🎯 - docs/ структура + algorithm comparison + enhanced analysis
- **v0.7.0** - HDBSCAN/Spectral + базовые тесты + CI/CD
- **v0.8.0+** - стабилизация API + advanced features
- **v1.0.0** - стабильный API, полная документация, production-ready

### Принципы
- Backward compatibility важна
- Опциональные зависимости остаются опциональными
- Код на английском, docstrings обязательны
- Type hints везде

### Для новых фич
Перед добавлением новой фичи проверь:
1. Соответствует ли общей архитектуре?
2. Нужны ли новые зависимости? (минимизировать!)
3. Как это повлияет на API?
4. Есть ли похожее в sklearn? (следовать их паттернам)
