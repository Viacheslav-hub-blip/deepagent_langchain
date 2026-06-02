# Поля uko

Источник: `uko`.

## Идентификаторы и время

- `index` - технический индекс.
- `event_id` - идентификатор raw-события.
- `event_time` - Unix epoch в миллисекундах.
- `event_dttm_readable` - читаемое время.
- `event_dt` - дата `YYYYMMDD`.
- `load_dt` - дата загрузки.
- `own_dttm` - дата-время загрузки.
- `user_id` - идентификатор пользователя.
- `epk_id` - клиентский ключ.

## Канал и операция

- `event_channel` - канал.
- `sub_channel` - подканал.
- `event_type` - тип события.
- `sub_type` - подтип.
- `type_operation` - тип операции.
- `event_description` - описание операции.
- `transaction_amount` - сумма.
- `transaction_amount_currency` - валюта.

## Клиент

- `first_name` - имя.
- `last_name` - фамилия.
- `middle_name` - отчество.
- `mobile_phone_number` - мобильный телефон.
- `client_phone_number` - телефон клиента.
- `dul_number` - номер документа.
- `client_card_number` - карта клиента.
- `birth_date_client` - дата рождения.
- `segment_client` - сегмент.
- `client_groups` - группы клиента.

## Счета, получатель и банк

- `payer_card_number` - карта плательщика.
- `payer_account_number` - счет плательщика.
- `number_acc` - счет.
- `transaction_sender_account_number` - счет отправителя.
- `transaction_beneficiar_account_number` - счет получателя.
- `transaction_beneficiar_bik` - БИК получателя.
- `recipient_bank_name` - банк получателя.
- `payee_phone_number` - телефон получателя.
- `recepient_fio` - ФИО получателя в исходном поле.
- `transaction_beneficiar_nick_name` - ник или название получателя.
- `operation_id` - идентификатор операции.
- `member_id` - участник или банк.
- `sbp_id` - идентификатор СБП.

## IP, гео и устройство

- `user_ip_location_country_code` - код страны по IP.
- `user_ip_location_city` - город по IP.
- `user_ip_location_region` - регион по IP.
- `ip_device` - IP устройства.
- `longitude_ip` - долгота по IP.
- `latitude_ip` - широта по IP.
- `hardware_id` - hardware id устройства.
- `os_id` - OS id.
- `device_time` - время на устройстве.
- `app_version` - версия приложения.
- `name_os` - ОС.
- `phone_brand` - бренд телефона.
- `phone_model` - модель телефона.
- `user_login_id` - идентификатор логина.
- `card_expire_date` - срок действия карты.
- `user_mobile_hardware_id_days_since_first_hit` - возраст hardware id.
- `device_mobile_days_since_first_hit` - возраст устройства.
- `payment_new_ip_provider` - признак нового IP-провайдера.
- `device_source_sdk` - данные SDK устройства.

## Правила и маркеры

- `final_marker_payer` - итоговый маркер плательщика.
- `tfm_client_marker` - маркер клиента TFM.
- `client_made_payment_to_recipient` - клиент платил этому получателю.
- `client_accepted_transfer_to_recipient_ignite` - клиент принимал перевод получателю.
- `main_rule` - основное правило raw-события.
- `rules` - список правил.
- `subrules` - список подправил.
- `risk_score_dsl` - risk score.
- `kafka_input_time` - время входа в Kafka.
- `kafka_output_time` - время выхода из Kafka.
- `indicators_vk_max` - VK-индикаторы.
- `scoring_oss` - скоринг OSS.
- `indicators_sbp` - индикаторы СБП.
- `params` - параметры события.

