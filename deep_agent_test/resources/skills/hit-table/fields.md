# Поля hits

Источник: `hits`.

Используй этот файл только если краткого списка в `/skills/hit-table/SKILL.md` недостаточно.

## Идентификаторы и время

- `index` - технический индекс строки.
- `event_id` - id сработки.
- `event_time` - читаемое время сработки `YYYY-MM-DD HH:MM:SS`.
- `event_dt` - дата сработки `YYYYMMDD`.
- `own_dt` - дата загрузки или партиции.
- `own_loading_id` - идентификатор загрузки.

## Клиент

- `epk_id` - клиентский ключ.
- `user_id` - идентификатор пользователя.
- `fio` - ФИО клиента.
- `segment` - сегмент клиента.
- `age` - возраст.
- `age_category` - возрастная категория.
- `phone` - телефон клиента.
- `phone_operator` - оператор телефона.
- `region_phone_operator` - регион оператора телефона.
- `dul_number` - номер документа.
- `dul_type` - тип документа.
- `payer_inn` - ИНН плательщика.

## Операция и сумма

- `transaction_amount` - сумма операции.
- `transaction_amount_in_rub` - сумма операции в рублях.
- `transaction_amount_currency` - валюта.
- `client_balance` - баланс клиента.
- `event_description` - текстовое описание события.
- `purpose` - назначение операции.
- `surface` - клиентская поверхность.
- `product` - продукт.
- `product_type` - тип продукта.
- `payment_transaction_flag` - признак платежной операции.

## Канал и тип события

- `event_channel` - канал события.
- `sub_channel` - подканал.
- `event_type` - тип события.
- `sub_type` - подтип.
- `type_operation` - тип операции.

## Карты, счета и получатель

- `card_number` - карта клиента.
- `transaction_sender_account_number` - счет отправителя.
- `p2p_sender_account_number` - счет отправителя P2P.
- `payer_account_number` - счет плательщика.
- `payer_card_number` - карта плательщика.
- `mobile_phone_number` - мобильный телефон.
- `payer_transfer_type` - тип перевода плательщика.
- `payee_transfer_type` - тип перевода получателю.
- `transaction_beneficiar_account_number` - счет получателя.
- `recipient_bik` - БИК получателя.
- `payee_bank_name` - банк получателя.
- `member_id` - идентификатор участника или банка.
- `sbp_id` - идентификатор СБП.
- `operation_id` - идентификатор операции.
- `recipient_info` - данные получателя в JSON/структуре.
- `recipient_inn` - ИНН получателя.

## Антифрод и резолюции

- `policy_action` - решение антифрод-системы.
- `main_rule` - основное правило.
- `tree_info` - данные сценария подтверждения.
- `type_accept` - тип подтверждения.
- `source_type_accept` - источник подтверждения.
- `resolution_first` - первая резолюция.
- `resolution_first_dttm` - время первой резолюции.
- `resolution_last` - последняя резолюция.
- `resolution_last_dttm` - время последней резолюции.
- `accept_time_sec` - время подтверждения в секундах.
- `has_claim` - признак жалобы.
- `is_save` - признак предотвращенного мошенничества.
- `marked_as_not_save_reason` - причина исключения из save.

## Дополнительный контекст

- `card_info` - данные карты.
- `trust_info` - данные доверенности.
- `atm_merchant_name` - merchant или ТСТ.
- `merchant_info` - данные merchant.
- `pos_info` - данные POS.
- `link_cf` - признаки связей и CF.
- `mobile_sdk_info` - данные мобильного SDK.
- `scoring_oss` - скоринг OSS.
- `previous_events` - краткие события до сработки.
- `posterious_events` - краткие события после сработки.
- `previous_events_additional_info` - детали событий до сработки.
- `posterious_events_additional_info` - детали событий после сработки.
- `hits_extra_facts` - дополнительные факты сработки.

