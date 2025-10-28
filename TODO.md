# ClusterTK - TODO

## Текущий статус: v0.4.2 ✅

**Latest Release:** v0.4.2
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

## 🎯 Приоритеты для v0.5.0

### HIGH PRIORITY

1. **Export & Reports**
   - [ ] Метод `pipeline.export_results()` - экспорт в CSV/JSON
     - CSV: датафрейм с labels и cluster assignments
     - JSON: профили кластеров, метрики, настройки pipeline
   - [ ] Метод `pipeline.export_report()` - HTML отчёт
     - Встроенные графики (base64)
     - Таблицы с метриками и профилями
     - Описание используемых настроек
   - [ ] Сохранение/загрузка pipeline (pickle или joblib)

### MEDIUM PRIORITY

2. **Tests**
   - [ ] Unit tests для критичных модулей (preprocessing, clustering, evaluation)
   - [ ] Integration tests для Pipeline
   - [ ] Тесты для edge cases
   - [ ] Цель: покрытие >50% для v0.5.0

3. **Documentation** (отложено на v0.6.0)
   - [x] ~~Текущий README - приемлемо для v0.4.x~~
   - [x] Добавлена навигация (Table of Contents, Quick Links)
   - [ ] **v0.6.0**: Создать docs/ структуру
   - [ ] **v0.6.0**: Сократить README до 100-150 строк
   - [ ] **v0.6.0**: GitHub Pages или ReadTheDocs

**Решение:** Для v0.4.x оставить текущий README с улучшенной навигацией. Полная реструктуризация документации - для v0.6.0+ когда API стабилизируется.

### LOW PRIORITY

4. **CI/CD**
   - [ ] GitHub Actions для автоматических тестов
   - [ ] Автоматическая проверка code style
   - [ ] Coverage reports

5. **Дополнительные фичи**
   - [ ] Больше алгоритмов (Spectral, HDBSCAN)
   - [ ] Ensemble clustering
   - [ ] Feature importance analysis
   - [ ] Сравнение разных алгоритмов

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
- **v0.5.x** - export/reports + базовые тесты
- **v0.6.x** - docs/ структура + полное покрытие тестами + CI/CD
- **v0.7.x+** - стабилизация API
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
