---
name: cards-event-table
description: "Raw-история карточного канала: POS, e-commerce, ATM, авторизации, токены и связь со сработкой из hits."
---

# Транзакционная таблица cards

Источник: `csp_afpc_sss_inc.cards_event`.

Смысл: raw-история карточных событий — POS, e-commerce, ATM, авторизации, операции по карте, merchant/MCC, терминальные признаки, карточные скоринги и ответы карточного контура. Для восстановления карточной последовательности до/после сработки. Строка ≠ антифрод-сработка.

Зерно: одна строка = одно raw-событие карточного канала.

## Ключи связи с hits

- `event_id` - может совпадать с `event_id` в hits.
- `epk_id` - клиентский ключ для сопоставления.
- `event_dt` (`YYYYMMDD`) - предпочтительное поле фильтрации и связи.
- `event_channel`, `event_type`, `sub_type`, `type_operation` - признаки карточного сценария.

## Ограничения

- `event_time` читаемый (`YYYY-MM-DD HH:MM:SS`), формат совместим с hits; для связи hits → cards_event достаточно `event_id` или `event_dt`.
- IP/гео полей мало: `token_device_ip`, `user_ip_location_city`, `user_ip_location_country`.

## Поля

```text
index
event_id
user_id
epk_id
card_number
event_type
sub_type
type_operation
client_transaction_id
card_owner
client_lastname
client_firstname
client_patronymicname
client_id_document_number
client_inn
client_phone
event_description
event_channel
transaction_amount
transaction_amount_in_rub
transaction_amount_currency
transaction_sender_account_number
transaction_beneficiar_account_number
atm_merchant_name
atm_mcc
atm_mcc_name
atm_city
atm_address
atm_country
atm_acquiring_country
user_ip_location_country
user_ip_location_city
token_device_ip
atm_terminal_id
atm_id
atm_merchant_id
atm_acquiring_iic
card_bin
card_type
card_ps
card_brand
time_transaction_local
data_transaction_local
response_code
cards_response_code_1
cards_dsl_model_risk_score
cards_dsl_model_receiver_score
cards_dsl_nspk_fraud_score
cards_client_markers
cards_fs_comprpid_marker
dbo_client_markers
phone_os
version_mp
channel_ext_system
own_dt
event_dt
event_time
```
