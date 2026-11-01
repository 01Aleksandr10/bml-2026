# Регуляризация

Команда: 11

## Содержание каталога

`src/main.ipynb` — код к презентации.

| Ячейка | Что показывает |
|---|---|
| `sin(2πx)` | полином 15 степени против Ridge на 12 шумных точках |
| Analytical solution | мультиколлинеарность, МНК против Ridge |
| L1 feature selection | Lasso path: коэффициенты обнуляются один за другим с ростом λ |
| Cost-complexity pruning | `R(T) + α|T|`  |
| Boosting | `w* = -G/(H+λ)` и порог по `γ`, посчитанные вручную |
| Итоги | замер по шести задачам |

## Запуск кода

```
pip install numpy pandas matplotlib scikit-learn
jupyter lab src/main.ipynb
```
