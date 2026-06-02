---
name: uko-event-table
description: "Краткая карточка источника uko: raw-история ДБО/СБП/переводов, счетов, устройства, IP и связь с hits."
---

# Таблица uko

Источник для `load_data`: `uko`.

Когда использовать:

- нужна raw-история не карточного канала;
- пользователь спрашивает про ДБО, СБП, переводы, счета, мобильное приложение, устройство, IP;
- нужно восстановить UKO/ДБО-поведение до/после сработки из `hits`;
- нужны признаки получателя, IP/гео, hardware/device, правила raw-события.

Зерно: одна строка = одно raw-событие не карточного канала. Это не строка антифрод-сработки.

## Ключи

- `event_id` - может совпадать с `hits.event_id`.
- `epk_id` - клиентский ключ.
- `event_dt` - дата `YYYYMMDD`.
- `event_dttm_readable` - читаемое время `YYYY-MM-DD HH:MM:SS`.
- `event_time` - Unix epoch в миллисекундах, не совместим с `hits.event_time`.

## Главные поля

- `event_id`
- `event_dt`
- `event_dttm_readable`
- `event_time`
- `epk_id`
- `user_id`
- `event_description`
- `event_channel`
- `sub_channel`
- `event_type`
- `sub_type`
- `type_operation`
- `transaction_amount`
- `transaction_amount_currency`
- `payer_account_number`
- `number_acc`
- `transaction_sender_account_number`
- `transaction_beneficiar_account_number`
- `transaction_beneficiar_bik`
- `recipient_bank_name`
- `payee_phone_number`
- `recepient_fio`
- `transaction_beneficiar_nick_name`
- `ip_device`
- `user_ip_location_city`
- `user_ip_location_region`
- `user_ip_location_country_code`
- `hardware_id`
- `os_id`
- `phone_brand`
- `phone_model`

## Ограничения

- Никогда не фильтруй `uko.event_time` значением `event_time` из `hits`.
- Для точного читаемого времени используй `event_dttm_readable`.
- Для связи с `hits` используй `event_id`, либо fallback `epk_id` + `event_dt`.

## Дополнительный контекст

- `/skills/uko-event-table/fields.md` - полный список полей `uko`.
- `/skills/hit-table/joins.md` - маршрут связи `hits` -> `cards` / `uko`.

Читай `fields.md`, если нужны редкие поля устройства, IP, правил raw-события или schema error.

