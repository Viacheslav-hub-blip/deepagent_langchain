---
name: cards-event-table
description: "Краткая карточка источника cards: raw-история карточного канала, POS, e-commerce, ATM, merchant/MCC и связь с hits."
---

# Таблица cards

Источник для `load_data`: `cards`.

Когда использовать:

- нужна raw-история карточной операции;
- пользователь спрашивает про POS, e-commerce, ATM, MCC, merchant, терминал, карточные скоринги;
- нужно восстановить карточное поведение до/после сработки из `hits`;
- по `hits` видно карточный канал или карточный тип операции.

Зерно: одна строка = одно raw-событие карточного канала. Это не строка антифрод-сработки.

## Ключи

- `event_id` - может совпадать с `hits.event_id`.
- `epk_id` - клиентский ключ для fallback-связи.
- `event_dt` - дата события `YYYYMMDD`.
- `event_time` - читаемое время `YYYY-MM-DD HH:MM:SS`.

## Главные поля

- `event_id`
- `event_dt`
- `event_time`
- `epk_id`
- `user_id`
- `event_description`
- `event_channel`
- `event_type`
- `sub_type`
- `type_operation`
- `transaction_amount`
- `transaction_amount_in_rub`
- `transaction_amount_currency`
- `card_number`
- `atm_merchant_name`
- `atm_mcc`
- `atm_mcc_name`
- `atm_city`
- `atm_country`
- `response_code`
- `token_device_ip`
- `user_ip_location_city`
- `user_ip_location_country`

## Ограничения

- Для связи с `hits` обычно достаточно `event_id`; fallback - `epk_id` + `event_dt`.
- IP/гео полей меньше, чем в `uko`.
- Не используй `cards` как источник антифрод-резолюций; резолюции находятся в `hits`.

## Дополнительный контекст

- `/skills/cards-event-table/fields.md` - полный список полей `cards`.
- `/skills/hit-table/joins.md` - маршрут связи `hits` -> `cards` / `uko`.

Читай `fields.md`, если нужно редкое карточное поле, MCC/merchant-разбивка, IP/гео или schema error.

