# Связь hits с raw-таблицами

Используй этот файл, когда нужно восстановить поведение клиента до/после сработки или получить raw-поля, которых нет в `hits`.

## Общий маршрут

1. Найди сработку в `hits`.
2. Возьми `event_id`, `epk_id`, `event_dt`, `event_channel`, `sub_channel`, `event_type`, `sub_type`, `type_operation`.
3. Выбери raw-таблицу:
   - `cards` для карточных операций, POS, e-commerce, ATM, merchant/MCC;
   - `uko` для ДБО, СБП, переводов, операций по счетам, устройства и IP.
4. Сначала пробуй связь по `event_id`.
5. Если совпадения нет, используй `epk_id` + `event_dt`.

## Важные запреты

- Не фильтруй `uko.event_time` значением из `hits.event_time`.
- Не делай точное сравнение времени между `hits` и `uko` без проверки форматов.
- Для дневной связи используй `event_dt`.

## Когда читать другие файлы

- Для полей `cards` читай `/skills/cards-event-table/fields.md`.
- Для полей `uko` читай `/skills/uko-event-table/fields.md`.

