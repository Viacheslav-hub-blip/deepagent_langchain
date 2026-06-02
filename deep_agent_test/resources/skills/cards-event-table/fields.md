# Поля cards

Источник: `cards`.

## Идентификаторы и время

- `index` - технический индекс.
- `event_id` - идентификатор raw-события.
- `user_id` - идентификатор пользователя.
- `epk_id` - клиентский ключ.
- `event_dt` - дата `YYYYMMDD`.
- `event_time` - время `YYYY-MM-DD HH:MM:SS`.
- `own_dt` - дата загрузки.
- `client_transaction_id` - клиентский идентификатор транзакции.

## Клиент и карта

- `card_number` - номер карты.
- `card_owner` - владелец карты.
- `client_lastname` - фамилия клиента.
- `client_firstname` - имя клиента.
- `client_patronymicname` - отчество клиента.
- `client_id_document_number` - номер документа.
- `client_inn` - ИНН клиента.
- `client_phone` - телефон клиента.
- `card_bin` - BIN карты.
- `card_type` - тип карты.
- `card_ps` - платежная система.
- `card_brand` - бренд карты.

## Операция

- `event_description` - описание операции.
- `event_channel` - канал.
- `event_type` - тип события.
- `sub_type` - подтип.
- `type_operation` - тип операции.
- `transaction_amount` - сумма.
- `transaction_amount_in_rub` - сумма в рублях.
- `transaction_amount_currency` - валюта.
- `transaction_sender_account_number` - счет отправителя.
- `transaction_beneficiar_account_number` - счет получателя.
- `response_code` - код ответа.
- `cards_response_code_1` - код ответа карточного контура.

## Merchant, MCC, ATM/POS

- `atm_merchant_name` - merchant или ТСТ.
- `atm_mcc` - MCC.
- `atm_mcc_name` - название MCC.
- `atm_city` - город устройства или ТСТ.
- `atm_address` - адрес устройства или ТСТ.
- `atm_country` - страна устройства или ТСТ.
- `atm_acquiring_country` - страна эквайера.
- `atm_terminal_id` - терминал.
- `atm_id` - банкомат.
- `atm_merchant_id` - merchant id.
- `atm_acquiring_iic` - идентификатор эквайера.
- `time_transaction_local` - локальное время транзакции.
- `data_transaction_local` - локальная дата транзакции.

## IP, устройство и скоринги

- `user_ip_location_country` - страна пользователя по IP.
- `user_ip_location_city` - город пользователя по IP.
- `token_device_ip` - IP токенизированного устройства.
- `cards_dsl_model_risk_score` - карточный risk score.
- `cards_dsl_model_receiver_score` - score получателя.
- `cards_dsl_nspk_fraud_score` - fraud-score НСПК.
- `cards_client_markers` - карточные маркеры клиента.
- `cards_fs_comprpid_marker` - маркер Cards FS ComPRPID.
- `dbo_client_markers` - маркеры клиента ДБО.
- `phone_os` - ОС телефона.
- `version_mp` - версия мобильного приложения.
- `channel_ext_system` - внешняя система канала.

