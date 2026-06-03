---
name: hit-table
description: "Краткая карточка источника hits: сработки антифрода, правила, решения, save/fp, жалобы и связь с raw-таблицами."
keywords: "hits, сработки, алерты, антифрод, правила, main_rule, policy_action, resolution, save, fp, жалобы, event_description"
---

# Таблица сработок hits

Источник для `load_data`: `hits`.

Когда использовать:

- пользователь спрашивает про сработки, алерты, hit-события, правила антифрода;
- нужны `policy_action`, `main_rule`, резолюции, save/fp, жалобы;
- нужно найти событие по `event_id`;
- нужно начать маршрут к raw-истории клиента через `cards` или `uko`.

Зерно: одна строка = одна сработка антифрод-мониторинга. Это не полная транзакционная история клиента.

## Ключи

- `event_id` - id сработки; может совпадать с `event_id` в `cards` или `uko`.
- `epk_id` - клиентский ключ для fallback-связи.
- `event_dt` - дата события в формате `YYYYMMDD`; основной фильтр периода.
- `event_channel`, `sub_channel`, `event_type`, `sub_type`, `type_operation` - признаки для выбора raw-таблицы.

## Главные поля

- `event_id`
- `event_dt`
- `event_time`
- `epk_id`
- `user_id`
- `fio`
- `event_description`
- `transaction_amount`
- `transaction_amount_in_rub`
- `event_channel`
- `sub_channel`
- `event_type`
- `sub_type`
- `type_operation`
- `policy_action`
- `main_rule`
- `resolution_first`
- `resolution_last`
- `has_claim`
- `is_save`
- `marked_as_not_save_reason`
- `previous_events`
- `posterious_events`

## Ограничения

- Не ищи IP/гео в `hits`; для этого переходи в raw-таблицы `cards` или `uko`.
- Не копируй `hits.event_time` как фильтр для `uko.event_time`: в `uko` другой формат времени.
- Для периода используй `event_dt`, а не преобразованный ISO.

## Дополнительный контекст

Читай дополнительные файлы только если краткой карточки недостаточно:

- `/skills/hit-table/fields.md` - полный список полей `hits` и краткие описания.
- `/skills/hit-table/joins.md` - правила связи `hits` с `cards` и `uko`.

Триггеры для чтения `/skills/hit-table/fields.md`:

- нужно выбрать редкое поле;
- пользователь спрашивает смысл поля;
- `load_data` вернул schema error;
- нужен широкий список колонок для выгрузки.
