# 🚢 Titanic Survival Prediction

> Решение задачи Kaggle «Titanic: Machine Learning from Disaster» с использованием ансамблевых методов и стекинга.

## 📊 Результат

| Метрика | Значение |
|---------|----------|
| **Cross-Validation Accuracy** | ~0.8428 (GradientBoosting) |
| **Stacking CV Accuracy** | 0.8384 (±0.0141) |

## 🧠 Подход

### Feature Engineering
- Извлечение **титулов** из имён (Mr, Miss, Mrs, Officer, Royalty)
- Обработка **кают**: извлечение палубы (`Deck`), заполнение по билетам
- **Семейные признаки**: размер семьи, одинок ли пассажир, большая семья
- **Группировка по билетам**: сколько человек едет по одному билету
- Бинарные признаки: `IsChild`, `IsElderly`, `HasCabin`, `HasBoatInfo`
- Логарифмирование тарифа (`LogFare`)
- Взаимодействия: `Sex_Pclass`, `Title_Pclass`

### Модели
| Модель | Accuracy (CV) |
|--------|---------------|
| GradientBoosting | **0.8428** |
| RandomForest | 0.8406 |
| XGBoost | 0.8384 |
| LightGBM | 0.8384 |
| SVM | 0.8271 |
| KNN | 0.8260 |
| LogisticRegression | 0.8215 |

### Ансамблирование
- **Стекинг (Stacking)**: 7 базовых моделей → мета-модель LogisticRegression
- Out-of-Fold предсказания для предотвращения переобучения

## 🛠️ Используемые технологии

- `pandas` / `numpy` — обработка данных
- `scikit-learn` — модели, кросс-валидация, масштабирование
- `XGBoost` / `LightGBM` — градиентный бустинг
- `GridSearchCV` — подбор гиперпараметров
