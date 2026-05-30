---
name: uko-event-table
description: "Raw-история не карточного канала: ДБО, СБП, переводы, операции по счетам, устройства и связь со сработкой из hits."
---

# Транзакционная таблица uko

Источник: `csp_afpc_sss_inc.uko_event`.

Смысл: raw-история не карточного контура — ДБО, СБП, переводы, операции по счетам, действия в приложении, устройство, IP, признаки получателя, правила и технические параметры. Для восстановления поведения в UKO/ДБО-сценариях до/после сработки. Строка ≠ антифрод-сработка.

Зерно: одна строка = одно raw-событие не карточного канала.

## Ключи связи с hits

- `event_id` - может совпадать с `event_id` в hits.
- `epk_id` - клиентский ключ для сопоставления.
- `event_dt` (`YYYYMMDD`) - предпочтительное поле фильтрации и связи.
- `event_channel`, `sub_channel`, `event_type`, `sub_type`, `type_operation` - признаки ДБО/UKO-сценария.

## Связывание с hits

1. Точечный поиск: `event_id` из hits без фильтра по `event_time`.
2. Fallback по клиенту: `epk_id` + `event_dt` (не `event_time` из hits).
3. Точное время в UKO — через `event_dttm_readable` или `event_dt`.

## Ограничения

- `event_time` хранится в Unix epoch **в миллисекундах** (`1769276343000`). **Не фильтруй его значением `event_time` из hits — форматы разные.**
- IP/гео поля: `ip_device`, `user_ip_location_city`, `user_ip_location_region`, `user_ip_location_country_code`, `longitude_ip`, `latitude_ip`.

## Поля

```text
index
event_id
event_time
event_dttm_readable
event_dt
load_dt
own_dttm
user_id
epk_id
event_channel
sub_channel
event_type
sub_type
type_operation
event_description
first_name
last_name
middle_name
mobile_phone_number
client_phone_number
dul_number
client_card_number
payer_card_number
payer_account_number
number_acc
transaction_sender_account_number
transaction_amount
transaction_amount_currency
transaction_beneficiar_account_number
transaction_beneficiar_bik
recipient_bank_name
payee_phone_number
recepient_fio
transaction_beneficiar_nick_name
operation_id
member_id
sbp_id
user_ip_location_country_code
user_ip_location_city
user_ip_location_region
ip_device
longitude_ip
latitude_ip
hardware_id
os_id
device_time
app_version
name_os
phone_brand
phone_model
user_login_id
card_expire_date
birth_date_client
segment_client
client_groups
user_mobile_hardware_id_days_since_first_hit
device_mobile_days_since_first_hit
payment_new_ip_provider
device_source_sdk
final_marker_payer
tfm_client_marker
client_made_payment_to_recipient
client_accepted_transfer_to_recipient_ignite
main_rule
rules
subrules
risk_score_dsl
kafka_input_time
kafka_output_time
indicators_vk_max
scoring_oss
indicators_sbp
params
```
