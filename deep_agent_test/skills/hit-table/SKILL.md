---
name: hit-table
description: "Сработки антифрода: решения, правила, резолюции, save-флаги, жалобы и краткий контекст событий до/после."
---

# Описание файла

Описание таблицы со сработками антифрода. Содержит бизнес-смысл источника и перечень полей без описания каждого поля.

# Таблица сработок

Источник: `cspfs_repo_features3.hits_extra_info_129372427_view`.

Бизнес-смысл: таблица фиксирует сработки антифрод-мониторинга, решения антифрод-системы, правила, резолюции, признаки предотвращенного мошенничества, жалобы и краткий контекст событий до/после сработки. Используется как основная точка входа для поиска и анализа alert/hit-событий. Не является полной транзакционной историей клиента.

Зерно данных: одна строка описывает одну антифрод-сработку или hit-событие.

Связи с другими источниками:

- `event_id` - идентификатор события/сработки. Может совпадать с `event_id` в raw-истории `cards_event` или `uko_event`, если сработка относится к конкретному транзакционному событию.
- `epk_id` - идентификатор клиента. Используется как клиентский ключ для сопоставления с raw-историей, когда прямого совпадения по `event_id` нет.
- `event_dt` и `event_time` - дата и время сработки. Используются для временного сопоставления с raw-событиями клиента.
- `event_channel`, `sub_channel`, `event_type`, `sub_type`, `type_operation` - признаки канала и типа события, которые помогают понять, какая raw-таблица содержит детальную транзакционную запись.

Ограничение источника: в hits нет полей геолокации пользователя по IP. Геоданные по IP находятся в raw-таблицах событий, если соответствующее событие там присутствует.

Поля:

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
