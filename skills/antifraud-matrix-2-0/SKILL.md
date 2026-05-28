---
name: antifraud-matrix-2-0
description: "Matrix 2.0: hits_extra_info_129372427_view, is_save, резолюции, purpose/surface/product, previous_events и posterious_events."
---

# Antifraud Matrix 2.0

Описание: точка входа для вопросов по Matrix 2.0.

Используй для `hits_extra_info_129372427_view`, `is_save`, `marked_as_not_save_reason`, `purpose`/`surface`/`product`, `has_claim`, `previous_events`, `posterious_events`, резолюций и `sdp_datastore_fs`.

Открой:

- `reference.md` - поля, кодировки и методология Matrix 2.0;
- `../_shared/antifraud-core.md` - бизнес-логика;
- `../_shared/data-sources.md` - hits/raw источники.

Ключевые правила:

- основная витрина: `cspfs_repo_features3.hits_extra_info_129372427_view`;
- покрытие: с `2026-01-01`;
- основной save-флаг: `is_save`;
- сумма save: `client_balance`, иначе `transaction_amount_in_rub`;
- при сравнении со старыми отчетами упоминай изменение логики с 2Q2026;
- чувствительные поля маскируй;
- код доступа к данным давай только по прямому запросу.
