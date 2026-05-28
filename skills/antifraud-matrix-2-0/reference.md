# Описание файла

Справочник Matrix 2.0. Содержит смысловые блоки: назначение витрины, правила доступа, аналитические поля, JSON-структуры, кодировки событий до/после, таксономию операций, паттерны анализа и типичные ошибки.

## Назначение

Matrix 2.0 - набор витрин и FS-потоков для KPI ФМ, статистики, срочных запросов и расследования антифрод-сработок. В сравнении с Matrix 1.0 изменились кластер расчетов, зависимости FS-потоков, формат базовых витрин, разметка `purpose`/`surface`/`product` и детализация неплатежных операций.

## Доступ и имена таблиц

- Старый контур: `datastore_sdp`.
- Новый контур: `sdp_datastore_fs`.
- Префикс таблиц нового контура: `cspfs_`.
- Основная витрина: `cspfs_repo_features3.hits_extra_info_129372427_view`.
- Старт покрытия: `2026-01-01`.

Если пользователь просит код доступа, укажи нужный кластер и префикс, но не вставляй полный Spark-шаблон без запроса.

## Аналитические правила

- `is_save` - главный предрасчитанный флаг предотвращенного мошенничества.
- Не добавляй дополнительные фильтры по типу операции, если задача не требует сужения выборки.
- Для суммы предотвращенного ущерба используй `client_balance`, а при его отсутствии `transaction_amount_in_rub`.
- Если `is_save = False`, объясняй исключение через `marked_as_not_save_reason`; при необходимости разбирай `posterious_events`.
- `payment_transaction_flag = True` для покупок/оплат, P2P, Me2Me и снятия наличных.
- `has_claim = True`, если жалоба клиента попала в окно от 2 недель до сработки до 2 недель после нее.
- При сравнении с историческими отчетами упоминай методологическое изменение с 2Q2026.

## Группы полей витрины

| Группа | Поля |
|---|---|
| Дата и событие | `event_dt`, `year`, `month`, `week`, `event_time`, `event_id` |
| Сумма и валюта | `transaction_amount`, `transaction_amount_in_rub`, `client_balance`, `transaction_amount_currency` |
| Канал и тип | `event_channel`, `sub_channel`, `event_type`, `sub_type`, `type_operation`, `event_description` |
| Клиент | `epk_id`, `user_id`, `segment`, `age`, `age_category`, `phone_operator`, `region_phone_operator` |
| Правила | `policy_action`, `main_rule`, `tree_info` |
| Подтверждение | `type_accept`, `source_type_accept`, `resolution_first`, `resolution_last`, `accept_time_sec` |
| Разметка | `purpose`, `surface`, `product`, `product_type`, `payment_transaction_flag` |
| Save и жалобы | `is_save`, `marked_as_not_save_reason`, `has_claim` |
| История | `previous_events`, `posterious_events`, `hits_extra_facts`, `previous_events_additional_info`, `posterious_event_additional_info` |

## Чувствительные поля

К чувствительным относятся `fio`, `phone`, `dul_number`, `card_number`, `payer_card_number`, `payer_account_number`, `transaction_sender_account_number`, `transaction_beneficiar_account_number`, `recipient_info`, `payer_inn`, `recipient_inn` и идентифицирующие реквизиты внутри JSON.

## JSON-поля

| Поле | Что искать |
|---|---|
| `main_rule` | `rule_name`, `rule_id`, `rule_category`, `description`, `tech_rule_flag` |
| `tree_info` | `event_id`, `verdict`, `client_balance`, `tree_aim`, `question_amount`, `q_and_a_json` |
| `branch_info` | `branch_terbank`, `branch_number`, `branch_office_number`, `branch_operator_sap_id`, `branch_operator_vsp`, `branch_one_hand` |
| `recipient_info` | `epk_id`, `user_id`, `fio`, `inn`, `number_card_recepient`, `account_number_of_recipient`, `payee_phone_number`, `brand_name`, `recipient_bank_name` |
| `card_info` | `card_type`, `card_linked_account`, `momentum_flg`, `virt_flg`, `payment_system` |
| `trust_info` | `truster_epk_id`, `trustee_epk_id`, `operation_person`, `dul_trustee` |
| `merchant_info` | `atm_id`, `atm_terminal_id`, `atm_merchant_id`, `atm_mcc`, `atm_mcc_name` |
| `pos_info` | `pos_data_input_mode`, `pos_cardholder_auth_method`, `pos_type`, `elcomm_cvv2_data`, `sbp_type_message`, `response_code` |
| `link_cf` | `recipient_from_address_book`, `export_user_payee_days_since_first_hit` |
| `mobile_sdk_info` | `phone_brand`, `name_os`, `app_version` |

## `posterious_events`

Массив успешных расходных операций после сработки. Окно: 5 дней после hit. Формат элемента:

`Event type | Recipient type | Link age | Amount bucket | Amount sign | Time bucket | Same-channel flag | Posterior channel | Reason flag`

| Атрибут | Значения |
|---|---|
| Event type | `WITHDRAW`, `PURCHASE`, `P2P`, `ME2ME`, `BOOKING`, `CASHOUT` |
| Recipient type | `SB` - тот же получатель/ТСТ; `--` - другой |
| Link age | `NEW` - связь меньше 11 дней; `OLD` - больше 11 дней |
| Amount bucket | `0_1p`, `1_5p`, `5_10p`, `10_15p`, `15_20p`, `20_25p`, `25_50p`, `50_100p` |
| Amount sign | `+` - последующая сумма больше; `-` - меньше |
| Time bucket | `0_1h`, `1_12h`, `12_24h`, `24_48h`, `48_72h`, `72_120h` |
| Same-channel flag | `SC` - тот же канал; `--` - другой |
| Posterior channel | `VSP`, `DBO`, `CARDS` |
| Reason flag | `main_reason`, `reason`, `--` |

`main_reason` - главный фактор исключения из предотвращенного мошенничества; максимум один на массив. `reason` - дополнительный фактор.

## `previous_events`

Массив событий до сработки: прошлые hits, пополнения/взносы и взятие кредита. Окна: hits и пополнения - 3 дня до hit; кредиты - 7 дней до hit. Формат элемента:

`Event type | Recipient type | Same-purpose flag | Amount bucket | Amount sign | Time bucket`

| Атрибут | Значения |
|---|---|
| Event type | `hit`, `me2me_deposit_sbp`, `p2p_deposit_sbp`, `p2p_deposit_internal`, `me2me_deposit_cash`, `p2p_deposit_cash`, `atm_deposit_cash`, `p2p_atm_deposit`, `credit` |
| Recipient type | `SB`, `--` |
| Same-purpose flag | `SP` - тот же `purpose`; `--` - другое |
| Amount bucket | `0_1p`, `1_5p`, `5_10p`, `10_20p`, `20_25p`, `25_50p`, `50_100p` |
| Amount sign | `+`, `-` |
| Time bucket | `0_1h`, `1_12h`, `12_24h`, `24_48h`, `48_96h`, `96_168h` |

## Таксономия операций

- Платежные purpose: покупка/оплата услуг, P2P, Me2Me, снятие наличных.
- Пополнение: взнос/пополнение, включая ВСП, ДБО, ATM, POS и e-commerce варианты.
- Кредиты и заказ наличных: отдельные purpose, не смешивать с платежными оборотами.
- Неплатежные и сервисные операции: заявки, активация/верификация карт, обновление данных, управление услугами, авторизация, запрос документов, токенизация, разблокировка профиля и прочие.
- `Дубль события` и `Сгенерированное событие ФС` учитывай как технические/специальные категории.

Для платежного скоупа сначала проверяй `payment_transaction_flag`, затем уточняй `purpose`, `surface`, `product`.

## Паттерны анализа

- Предотвращенное мошенничество: фильтр `is_save = True`, сумма через `client_balance` с fallback на `transaction_amount_in_rub`, группировки по нужным измерениям.
- Почему кейс не save: `is_save = False`, затем `marked_as_not_save_reason`, затем расшифровка `posterious_events`.
- Разбор по `event_id`: идентификация, канал, сумма, `purpose`/`surface`/`product`, `main_rule`, резолюции, `accept_time_sec`, `is_save`, история до/после.
- Ложные сработки и клиентское трение: `has_claim`, `is_save`, `marked_as_not_save_reason`, `accept_time_sec`, `resolution_last`, канал и разметка операции.

## Типичные ошибки

- Сравнивать Matrix 2.0 со старыми отчетами без оговорки про 2Q2026.
- Считать `transaction_amount_in_rub` единственной суммой предотвращенного ущерба.
- Предполагать, что все поля заполнены во всех каналах.
- Принимать `previous_events` за полную историю клиента.
- Принимать `posterious_events` за все последующие события клиента.
- Раскрывать чувствительные поля без маскирования.
- Исправлять физические опечатки полей в логике запроса.
