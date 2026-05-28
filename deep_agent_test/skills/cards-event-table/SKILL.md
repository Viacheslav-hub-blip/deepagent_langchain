---
name: cards-event-table
description: "Raw-история карточного канала: POS, e-commerce, ATM, авторизации, токены и связь со сработкой из hits."
---

# Описание файла

Описание транзакционной таблицы карточного контура. Содержит бизнес-смысл источника и перечень полей без описания каждого поля.

# Транзакционная таблица cards

Источник: `csp_afpc_sss_inc.cards_event`.

Бизнес-смысл: таблица хранит raw-историю карточных событий клиента: POS, e-commerce, ATM, авторизации, операции по карте, merchant/MCC, терминальные признаки, карточные скоринги и технические ответы карточного контура. Используется для восстановления полной карточной последовательности до/после сработки. Строка в этой таблице не означает антифрод-сработку.

Зерно данных: одна строка описывает одно raw-событие карточного канала.

Связи с hits:

- `event_id` - идентификатор raw-события. Может совпадать с `event_id` в hits, если карточное событие породило или связано с антифрод-сработкой.
- `epk_id` - клиентский ключ для сопоставления с hits и другими raw-таблицами.
- `event_dt` и `event_time` - дата и время raw-события для временного сопоставления со сработкой.
- `event_channel`, `event_type`, `sub_type`, `type_operation` - признаки карточного сценария.

Поля IP и геолокации:

- `token_device_ip` - IP токенизированного устройства или карточного события.
- `user_ip_location_city` - город пользователя, определенный по IP.
- `user_ip_location_country` - страна пользователя, определенная по IP.

Поля:

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
