---
name: antifraud-hit-table
description: "Сработки антифрода по hits: event_id, policy_action, main_rule, резолюции, is_save, жалобы и выбор raw-истории."
---

# Hit Table

Описание: точка входа для вопросов по таблице сработок.

Источник: `cspfs_repo_features3.hits_extra_info_129372427_view`.

Открой:

- `../_shared/data-sources.md` - поля, ключи, связь с raw-историей;
- `../_shared/antifraud-core.md` - смысл `policy_action`, резолюций и ограничений;
- `../antifraud-matrix-2-0/reference.md` - если задача про `is_save`, `previous_events` или `posterious_events`.

Правила:

- hits - это только сработки, не полная история клиента;
- поведение до/после проверяется через `cards_event` или `uko_event`;
- `policy_action` не является резолюцией.
