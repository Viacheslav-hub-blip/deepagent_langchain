---
name: antifraud-cards-event-table
description: "Raw-история карточного канала: POS, e-commerce, ATM, авторизации, токены и связь со сработкой из hits."
---

# Cards Event Table

Описание: точка входа для карточной raw-истории.

Источники: `csp_afpc_sss_inc.cards_event`, при необходимости `cspfs_repo_features3.cards_event`.

Используй для POS, e-commerce, ATM, авторизаций, проверки карты, токенизации, NFC и карточных событий до/после сработки.

Открой `../_shared/data-sources.md` для ключей, полей и правил связывания.

Не используй для UKO/ДБО/ВСП сценариев. Для решения, правила и резолюции возвращайся в hits.
