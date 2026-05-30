---
name: hit-table
description: "Сработки антифрода: решения, правила, резолюции, save-флаги, жалобы и краткий контекст событий до/после."
---

# Таблица сработок

Источник: `cspfs_repo_features3.hits_extra_info_129372427_view`.

Смысл: сработки антифрод-мониторинга — решения, правила, резолюции, признаки предотвращённого мошенничества, жалобы, краткий контекст событий до/после. Основная точка входа для поиска alert/hit-событий. Не полная транзакционная история клиента.

Зерно: одна строка = одна сработка / hit-событие.

## Ключи связи

- `event_id` - id сработки; может совпадать с `event_id` в `cards_event` / `uko_event`.
- `epk_id` - клиентский ключ для связи, когда совпадения по `event_id` нет.
- `event_dt` (`YYYYMMDD`) - предпочтительное поле фильтрации и связи с raw-таблицами.
- `event_channel`, `sub_channel`, `event_type`, `sub_type`, `type_operation` - подсказывают, в какой raw-таблице лежит детальная запись.

## Связывание с raw-таблицами (cards_event / uko_event)

1. Сначала по `event_id`, если он есть в hits.
2. Для дневного сопоставления и fallback — по `event_dt`, а не по точному `event_time`.
3. `event_dt` передавай как `YYYYMMDD` (`20260124`), без преобразования в ISO.
4. Не добавляй фильтр `event_time = <значение из hits>` для `uko_event` — там другой формат хранения.
5. Точность по времени в `uko_event` — через `event_dttm_readable`, а не `event_time`.

## Ограничения

- Нет полей геолокации по IP — геоданные ищи в raw-таблицах событий.
- `event_time` здесь читаемый (`YYYY-MM-DD HH:MM:SS`); не копируй как фильтр в `uko_event`.

## Поля

```text
index
event_time
event_id
transaction_amount
transaction_amount_in_rub
client_balance
transaction_amount_currency
event_channel
sub_channel
event_type
sub_type
type_operation
event_description
tree_info
policy_action
main_rule
epk_id
user_id
fio
segment
age
age_category
phone
phone_operator
region_phone_operator
dul_number
dul_type
payer_inn
card_number
transaction_sender_account_number
p2p_sender_account_number
payer_account_number
payer_card_number
mobile_phone_number
payer_transfer_type
payee_transfer_type
transaction_beneficiar_account_number
recipient_bik
payee_bank_name
member_id
sbp_id
operation_id
recipient_info
card_info
trust_info
recipient_inn
atm_merchant_name
merchant_info
pos_info
link_cf
mobile_sdk_info
scoring_oss
type_accept
source_type_accept
resolution_first
resolution_first_dttm
resolution_last
resolution_last_dttm
accept_time_sec
purpose
surface
product
product_type
payment_transaction_flag
has_claim
is_save
marked_as_not_save_reason
posterious_events
previous_events
hits_extra_facts
posterious_events_additional_info
previous_events_additional_info
own_loading_id
own_dt
event_dt
```
