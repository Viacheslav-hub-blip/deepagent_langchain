---
name: average-transaction-by-rule
description: "Workflow расчета статистики суммы транзакций по правилу антифрода в hits."
---

# Статистика суммы по правилу

Используй, когда пользователь спрашивает среднюю, минимальную, максимальную сумму или количество сработок по правилу антифрода.

Основная таблица: `hits`.

## Алгоритм

1. Найди сработки в `hits`.
2. Фильтр по правилу: `main_rule contains <название или ключевая подстрока>`.
3. Запроси минимум поля:
   - `event_id`
   - `event_dt`
   - `main_rule`
   - `transaction_amount`
   - `transaction_amount_in_rub`
4. Если пользователь не задал период, не добавляй период.
5. Если строк много или результат ушёл в `.pkl`, считай статистику через `execute_python_code`.

## Пример load_data

```text
table_name: hits
select_columns: ["event_id", "event_dt", "main_rule", "transaction_amount", "transaction_amount_in_rub"]
filters:
  - {"column": "main_rule", "operator": "contains", "value": "<текст правила>"}
```

## Ограничения

- `main_rule` может быть JSON-строкой; используй `contains`, а не точное равенство, если пользователь дал только название.
- Для рублевой статистики предпочитай `transaction_amount_in_rub`.
- Если сработок нет, проверь уникальные `main_rule` по ключевым словам из запроса.

## Дополнительный контекст

- `/skills/hit-table/SKILL.md` - краткая карточка `hits`.
- `/skills/hit-table/fields.md` - редкие поля `hits`, если нужна расширенная выгрузка.

