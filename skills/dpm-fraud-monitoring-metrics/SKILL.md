---
name: dpm-fraud-monitoring-metrics
description: "KPI ДПМ: effectiveness, FBP, FP, AR, CSI, save/lost, false positives и отличия от метрик ЦБ."
---

# DPM Fraud Monitoring Metrics

Описание: точка входа для KPI и методологии ДПМ.

Используй для effectiveness, FBP, FBP по самопереводам, FP, AR, CSI, save, lost, false positives, prevented/lost fraud и сравнений с ЦБ.

Открой:

- `reference.md` - формулы, включения и исключения;
- `../_shared/antifraud-core.md` - бизнес-логика;
- `../_shared/data-sources.md` - источники данных.

Порядок:

1. Определи метрику, популяцию и канал.
2. Примени включения/исключения из `reference.md`.
3. Раздели `save`, `lost`, accepted fraud, false positive и неопределенные кейсы.
4. Если данных не хватает, перечисли недостающие поля.

Не включай accepted fraud в `lost`, события без резолюции в FP и mandatory stop в AR.
