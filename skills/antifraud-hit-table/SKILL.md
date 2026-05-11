---
name: antifraud-hit-table
description: Разбор антифрод-сработки по event_id — правило, policy_action, резолюции, save, жалоба; выбор сырой таблицы событий (cards vs uko).
---
# Сработки: `cspfs_repo_features3.hits_extra_info_129372427_view`

Только операции с срабатыванием фрод-мониторинга (не полная история клиента).

**Идентификация и время:** `event_id`, `event_dt`, `event_time`

**Канал → сырая история:** `event_channel`, `surface`, `product` → см. skills `antifraud-cards-event-table` / `antifraud-uko-event-table`

**Деньги и операция:** `transaction_amount_in_rub`, `type_operation`, `sub_channel`, `event_type` / `sub_type`, `purpose`, `payment_transaction_flag`, `client_balance`

**Антифрод:** `policy_action` (allow/review/deny — не финальная резолюция), `main_rule`, `resolution_first` / `resolution_last`, `accept_time_sec`

**Save / жалобы:** `is_save`, `marked_as_not_save_reason`, `has_claim`

**Краткий контекст (не замена полной истории):** `previous_events`, `posterious_events`

**Правила:** `policy_action` ≠ итоговая резолюция; для поведения до/после — выгрузка из `cards_event` или `uko_event`, не только hits.
