"""Генератор синтетического antifraud-датасета для локальных примеров.

Содержит:
- ClientProfile: описание синтетического клиента.
- Operation: описание одного raw-события клиента.
- TableTemplates: набор шаблонных строк и колонок исходных CSV.
- event_dt: преобразование даты в формат YYYYMMDD.
- uuid_str: генерация строкового UUID.
- hex_id: генерация hex-идентификатора.
- digits: генерация цифровой строки.
- js: сериализация JSON без ASCII-экранирования.
- read_templates: чтение текущих CSV как источника схем и шаблонов.
- build_clients: создание 10 согласованных профилей клиентов.
- build_hit_dates: создание расписания сработок по заданной плотной схеме.
- choose_operation_kind: выбор типа операции для raw-события.
- build_operations: генерация 100 операций на каждого клиента.
- fill_common_client_fields: заполнение клиентских полей в строке таблицы.
- make_cards_row: создание строки карточного события.
- make_uko_row: создание строки UKO/DBO-события.
- make_hits_row: создание строки таблицы сработок Matrix 2.0.
- make_history_row: создание строки истории авторазметки.
- build_tables: сбор пяти CSV-таблиц.
- validate_tables: проверка целостности и контрольных метрик.
- write_tables: запись CSV-файлов в examples/data.
- main: точка входа для локальной генерации.
"""

from __future__ import annotations

import csv
import json
import random
import uuid
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import pandas as pd


DATA_DIR = Path(__file__).resolve().parent / "data"

HITS_FILE = "cspfs_repo_features3.hits_extra_info_129372427_view.csv"
CARDS_FILE = "csp_afpc_sss_inc.cards_event.csv"
UKO_FILE = "csp_afpc_sss_inc.uko_event.csv"
HISTORY_FILE = "csp_repo_features.history_automarking_big_148078_155487.csv"
TIMELINE_FILE = "demo_client_timeline.csv"

START_DATE = datetime(2026, 1, 24, 8, 15, 0)
END_DATE = datetime(2026, 3, 9, 21, 30, 0)
OPERATIONS_PER_CLIENT = 100
RANDOM_SEED = 20260516

HIT_GROUPS = {
    "spread": {"clients": 5, "hits": 8},
    "single": {"clients": 2, "hits": 1},
    "regular": {"clients": 3, "hits": 15},
}

EDUCATION_SCENARIOS = [
    {
        "description": "Оплата обучения",
        "merchant": "Университет Синергия",
        "mcc": "8299",
        "mcc_name": "Educational Services",
        "amounts": (28500.0, 98000.0),
    },
    {
        "description": "Оплата онлайн-курсов",
        "merchant": "Яндекс Практикум",
        "mcc": "8299",
        "mcc_name": "Educational Services",
        "amounts": (14900.0, 79000.0),
    },
    {
        "description": "Оплата учебных материалов",
        "merchant": "ЛитРес Образование",
        "mcc": "5942",
        "mcc_name": "Book Stores",
        "amounts": (1200.0, 12500.0),
    },
    {
        "description": "Оплата экзамена",
        "merchant": "Центр тестирования",
        "mcc": "8299",
        "mcc_name": "Educational Services",
        "amounts": (3500.0, 24000.0),
    },
    {
        "description": "Оплата занятий с репетитором",
        "merchant": "Профи Обучение",
        "mcc": "8299",
        "mcc_name": "Educational Services",
        "amounts": (2500.0, 32000.0),
    },
    {
        "description": "Оплата образовательной платформы",
        "merchant": "Skillbox",
        "mcc": "8299",
        "mcc_name": "Educational Services",
        "amounts": (9900.0, 155000.0),
    },
]

OTHER_SCENARIOS = [
    {
        "description": "Оплата мобильной связи",
        "merchant": "МегаФон",
        "mcc": "4814",
        "mcc_name": "Telecommunication Services",
        "amounts": (450.0, 2200.0),
    },
    {
        "description": "Оплата услуг ЖКХ",
        "merchant": "Мосэнергосбыт",
        "mcc": "4900",
        "mcc_name": "Utilities",
        "amounts": (2800.0, 14500.0),
    },
    {
        "description": "Покупка по карте в торговой точке",
        "merchant": "Пятёрочка",
        "mcc": "5411",
        "mcc_name": "Retail",
        "amounts": (600.0, 8500.0),
    },
    {
        "description": "Перевод С2С из Сбербанка в сторонний Банк",
        "merchant": "Получатель перевода",
        "mcc": "6012",
        "mcc_name": "Financial Institutions",
        "amounts": (1500.0, 120000.0),
    },
]

BANKS = [
    ("ПАО Сбербанк", "044525225"),
    ("Альфа-Банк", "044525593"),
    ("ВТБ", "044525187"),
    ("Т-Банк", "044525974"),
    ("Газпромбанк", "044525823"),
]

RULES = [
    ("DENY новый образовательный получатель и высокий риск устройства", "New Payee"),
    ("DENY оплата обучения после смены устройства", "Device Anomaly"),
    ("CARD_DENY крупная покупка образовательных услуг после cash-in", "Behavior Anomaly"),
    ("DENY нетипичная сумма оплаты курсов", "Amount Anomaly"),
    ("DENY регулярные платежи образовательной платформе с высоким риском", "Regular Pattern"),
]

RECIPIENTS = [
    ("Иванова Мария Игоревна", "Образование", "+7 925 331-19-82"),
    ("Петров Денис Алексеевич", "Репетитор", "+7 903 742-88-10"),
    ("Смирнов Павел Максимович", "Куратор", "+7 916 114-22-75"),
    ("Соколова Анна Дмитриевна", "Методист", "+7 977 541-04-19"),
]


@dataclass(frozen=True)
class ClientProfile:
    """Описывает одного синтетического клиента.

    Args:
        user_id: Идентификатор клиента в raw-таблицах.
        epk_id: ЕПК/PPRB идентификатор клиента.
        fio: Полное имя клиента.
        last_name: Фамилия клиента.
        first_name: Имя клиента.
        middle_name: Отчество клиента.
        age: Возраст клиента.
        age_category: Возрастная категория клиента.
        phone: Телефон в формате с пробелами.
        phone_masked: Телефон в формате, используемом UKO-таблицей.
        dul_number: Номер документа без разделителей.
        dul_number_uko: Номер документа в формате UKO-таблицы.
        inn: ИНН клиента.
        card_number: Номер карты клиента.
        account_number: Номер основного счета клиента.
        number_acc: Номер вклада/дополнительного счета клиента.
        hardware_id: Идентификатор устройства клиента.
        os_id: Идентификатор ОС устройства клиента.
        user_login_id: Логин клиента.
        birth_date_client: Дата рождения в формате UKO-таблицы.
        client_birthdate: Дата рождения в формате истории авторазметки.
        segment: Сегмент клиента.
        group: Группа сценария сработок.

    Returns:
        Неизменяемый профиль клиента для генерации согласованных строк.
    """

    user_id: str
    epk_id: str
    fio: str
    last_name: str
    first_name: str
    middle_name: str
    age: int
    age_category: str
    phone: str
    phone_masked: str
    dul_number: str
    dul_number_uko: str
    inn: str
    card_number: str
    account_number: str
    number_acc: str
    hardware_id: str
    os_id: str
    user_login_id: str
    birth_date_client: str
    client_birthdate: str
    segment: str
    group: str


@dataclass(frozen=True)
class Operation:
    """Описывает одно raw-событие клиента.

    Args:
        event_id: Уникальный идентификатор события.
        client: Профиль клиента, которому принадлежит событие.
        table_type: Целевая raw-таблица: cards или uko.
        dt: Дата и время события.
        amount: Сумма операции в рублях.
        event_type: Тип события.
        sub_type: Подтип события.
        type_operation: Детальный тип операции.
        description: Человекочитаемое описание операции.
        scenario: Справочник сценария операции.
        is_hit: Признак антифрод-сработки.
        policy_action: Решение политики антифрода.
        rule_name: Название правила сработки.
        rule_category: Категория правила сработки.
        risk_score: Риск-скор операции.
        resolution_last: Последняя резолюция сработки.
        reason: Краткая причина для timeline и auto-marking.

    Returns:
        Неизменяемое описание операции для сборки строк таблиц.
    """

    event_id: str
    client: ClientProfile
    table_type: str
    dt: datetime
    amount: float
    event_type: str
    sub_type: str
    type_operation: str
    description: str
    scenario: dict[str, Any]
    is_hit: bool
    policy_action: str
    rule_name: str
    rule_category: str
    risk_score: int
    resolution_last: str
    reason: str


@dataclass(frozen=True)
class TableTemplates:
    """Хранит шаблоны строк и порядок колонок исходных CSV.

    Args:
        hits: DataFrame текущей таблицы сработок.
        cards: DataFrame текущей карточной raw-таблицы.
        uko: DataFrame текущей UKO/DBO raw-таблицы.
        history: DataFrame текущей таблицы авторазметки.
        timeline: DataFrame текущей timeline-таблицы.

    Returns:
        Объект с исходными схемами и шаблонными строками для генерации.
    """

    hits: pd.DataFrame
    cards: pd.DataFrame
    uko: pd.DataFrame
    history: pd.DataFrame
    timeline: pd.DataFrame


def event_dt(dt: datetime) -> str:
    """Преобразует дату в компактный формат YYYYMMDD.

    Args:
        dt: Дата и время операции.

    Returns:
        Строка с датой в формате YYYYMMDD.
    """

    return dt.strftime("%Y%m%d")


def uuid_str() -> str:
    """Генерирует строковый UUID.

    Args:
        Отсутствуют.

    Returns:
        UUID в строковом формате.
    """

    return str(uuid.uuid4())


def hex_id(length: int = 24) -> str:
    """Генерирует hex-идентификатор заданной длины.

    Args:
        length: Требуемая длина итоговой строки.

    Returns:
        Строка из hex-символов.
    """

    return uuid.uuid4().hex[:length]


def digits(length: int) -> str:
    """Генерирует цифровую строку заданной длины.

    Args:
        length: Количество цифр в строке.

    Returns:
        Строка из случайных цифр.
    """

    return "".join(random.choice("0123456789") for _ in range(length))


def js(obj: Any) -> str:
    """Сериализует объект в компактный JSON.

    Args:
        obj: JSON-совместимый объект.

    Returns:
        JSON-строка без ASCII-экранирования.
    """

    return json.dumps(obj, ensure_ascii=False, separators=(",", ":"))


def read_templates(data_dir: Path) -> TableTemplates:
    """Читает текущие CSV для сохранения порядка колонок и базовых шаблонов.

    Args:
        data_dir: Каталог с исходными CSV-файлами.

    Returns:
        Набор DataFrame с текущими схемами таблиц.
    """

    read_kwargs = {"dtype": str, "keep_default_na": False, "low_memory": False}
    return TableTemplates(
        hits=pd.read_csv(data_dir / HITS_FILE, **read_kwargs),
        cards=pd.read_csv(data_dir / CARDS_FILE, **read_kwargs),
        uko=pd.read_csv(data_dir / UKO_FILE, **read_kwargs),
        history=pd.read_csv(data_dir / HISTORY_FILE, **read_kwargs),
        timeline=pd.read_csv(data_dir / TIMELINE_FILE, **read_kwargs),
    )


def build_clients() -> list[ClientProfile]:
    """Создает 10 профилей клиентов для плотного тестового датасета.

    Args:
        Отсутствуют.

    Returns:
        Список из 10 согласованных клиентских профилей.
    """

    raw_people = [
        ("7770421986", "2099007770421986000001", "Кузнецов Андрей Сергеевич", "Кузнецов", "Андрей", "Сергеевич", 42, "40-45 лет", "+7 916 742-18-65", "+7 (916) 742-18-65", "4519884216", "451-98-84216", "772845913248", "5599004387129866", "40817810400077421865", "42307810900077421865", "A7F98B33D014CC2E93AA7B20477D11F8", "F68D12C9237B4C8EA5D03184C930B6AF", "login777042198600001", "14/05/1984", "1984-05-14"),
        ("7770421987", "2099007770421987000001", "Иванова Елена Павловна", "Иванова", "Елена", "Павловна", 35, "35-40 лет", "+7 925 104-22-91", "+7 (925) 104-22-91", "4520113377", "452-01-13377", "772845913249", "5599004387129873", "40817810400077104921", "42307810900077104921", "C8F11B33D014CC2E93AA7B20477D22C1", "A72D12C9237B4C8EA5D03184C930B6B2", "login777042198700001", "22/07/1990", "1990-07-22"),
        ("7770421988", "2099007770421988000001", "Смирнов Олег Викторович", "Смирнов", "Олег", "Викторович", 29, "25-30 лет", "+7 903 512-44-08", "+7 (903) 512-44-08", "4521445091", "452-14-45091", "772845913250", "5599004387129880", "40817810400077512440", "42307810900077512440", "D9A22C44E125DD3F04BB8C31588E33D2", "B83E23D0348C5D9FB6E14295D041C7C3", "login777042198800001", "03/11/1996", "1996-11-03"),
        ("7770421989", "2099007770421989000001", "Петрова Марина Олеговна", "Петрова", "Марина", "Олеговна", 48, "45-50 лет", "+7 977 771-09-13", "+7 (977) 771-09-13", "4517763205", "451-77-63205", "772845913251", "5599004387129897", "40817810400077771091", "42307810900077771091", "E0B33D55F236EE4A15CC9D42699F44E3", "C94F34E1459D6E0AC7F25306E152D8D4", "login777042198900001", "19/02/1978", "1978-02-19"),
        ("7770421990", "2099007770421990000001", "Соколова Анна Дмитриевна", "Соколова", "Анна", "Дмитриевна", 31, "30-35 лет", "+7 916 218-77-64", "+7 (916) 218-77-64", "4522851740", "452-28-51740", "772845913252", "5599004387129903", "40817810400077218776", "42307810900077218776", "F1C44E660347FF5B26DDAE537AAF55F4", "DA5045F256AE7F1BD8036417F263E9E5", "login777042199000001", "28/04/1994", "1994-04-28"),
        ("7770421991", "2099007770421991000001", "Морозов Илья Андреевич", "Морозов", "Илья", "Андреевич", 54, "50-55 лет", "+7 968 333-50-12", "+7 (968) 333-50-12", "4516540029", "451-65-40029", "772845913253", "5599004387129910", "40817810400077333501", "42307810900077333501", "A2D55F771458006C37EEBF648BB066A5", "EB61560367BF802CE91475280374FAF6", "login777042199100001", "11/09/1971", "1971-09-11"),
        ("7770421992", "2099007770421992000001", "Федорова Наталья Игоревна", "Федорова", "Наталья", "Игоревна", 39, "35-40 лет", "+7 999 441-80-77", "+7 (999) 441-80-77", "4523991844", "452-39-91844", "772845913254", "5599004387129927", "40817810400077441807", "42307810900077441807", "B3E660882569117D48FFC0759CC177B6", "FC72671478C0913DFA25863914850B07", "login777042199200001", "05/01/1987", "1987-01-05"),
        ("7770421993", "2099007770421993000001", "Орлов Павел Максимович", "Орлов", "Павел", "Максимович", 26, "25-30 лет", "+7 901 882-31-49", "+7 (901) 882-31-49", "4524672831", "452-46-72831", "772845913255", "5599004387129934", "40817810400077882314", "42307810900077882314", "C4F771993670228E59AAD186ADD288C7", "0D83782589D1A24E0B36974025961C18", "login777042199300001", "17/06/1999", "1999-06-17"),
        ("7770421994", "2099007770421994000001", "Васильева Ольга Романовна", "Васильева", "Ольга", "Романовна", 45, "45-50 лет", "+7 926 615-72-20", "+7 (926) 615-72-20", "4517399016", "451-73-99016", "772845913256", "5599004387129941", "40817810400077615722", "42307810900077615722", "D50882AA4781339F60BBE297BEE399D8", "1E94893690E2B35F1C47085136A72D29", "login777042199400001", "30/12/1980", "1980-12-30"),
        ("7770421995", "2099007770421995000001", "Никитин Сергей Алексеевич", "Никитин", "Сергей", "Алексеевич", 33, "30-35 лет", "+7 912 550-14-36", "+7 (912) 550-14-36", "4525880142", "452-58-80142", "772845913257", "5599004387129958", "40817810400077550143", "42307810900077550143", "E61993BB589244A071CCF308CFF4AAE9", "2FA5904701F3C4602D58196247B83E3A", "login777042199500001", "24/08/1992", "1992-08-24"),
    ]
    groups = ["spread"] * 5 + ["single"] * 2 + ["regular"] * 3
    return [ClientProfile(*person, segment="МВС", group=group) for person, group in zip(raw_people, groups, strict=True)]


def build_hit_dates(client_index: int, group: str) -> set[int]:
    """Создает множество дней периода, в которые у клиента будут сработки.

    Args:
        client_index: Порядковый номер клиента в списке профилей.
        group: Группа сценария сработок: spread, single или regular.

    Returns:
        Множество смещений дней от START_DATE.
    """

    if group == "spread":
        base = [1, 7, 13, 19, 25, 31, 37, 43]
        return {min(44, day + (client_index % 3) - 1) for day in base}
    if group == "single":
        return {12 + client_index * 5}
    if group == "regular":
        start = client_index % 3
        return {start + step * 3 for step in range(15)}
    raise ValueError(f"Неизвестная группа сработок: {group}")


def choose_operation_kind(client_index: int, op_index: int, is_hit: bool) -> tuple[str, dict[str, Any], str, str, str]:
    """Выбирает канал и тип операции для события.

    Args:
        client_index: Порядковый номер клиента.
        op_index: Порядковый номер операции клиента.
        is_hit: Признак того, что операция должна быть сработкой.

    Returns:
        Кортеж table_type, scenario, event_type, sub_type, type_operation.
    """

    scenario_pool = EDUCATION_SCENARIOS if is_hit or op_index % 3 != 0 else OTHER_SCENARIOS
    scenario = scenario_pool[(client_index + op_index) % len(scenario_pool)]
    table_type = "cards" if (client_index + op_index) % 2 == 0 else "uko"

    if table_type == "cards":
        return table_type, scenario, "PURCHASE", "PURCHASE", "CARD_PURCHASE"
    if "Перевод" in scenario["description"]:
        return table_type, scenario, "PAYMENT", "RURPAYMENT", "PAYMENT_SBP"
    if "мобильной" in scenario["description"]:
        return table_type, scenario, "PAYMENT", "PHONE", "PHONE_PAYMENT"
    if "ЖКХ" in scenario["description"]:
        return table_type, scenario, "PAYMENT", "UTILITY", "UTILITY_PAYMENT"
    return table_type, scenario, "PAYMENT", "EDUCATION", "EDUCATION_PAYMENT"


def build_operations(clients: list[ClientProfile]) -> list[Operation]:
    """Генерирует 100 операций на каждого клиента.

    Args:
        clients: Список клиентских профилей.

    Returns:
        Отсортированный по времени список операций всех клиентов.
    """

    operations: list[Operation] = []
    for client_index, client in enumerate(clients):
        hit_days = build_hit_dates(client_index, client.group)
        daily_hit_usage: set[int] = set()
        for op_index in range(OPERATIONS_PER_CLIENT):
            day_offset = (op_index * 7 + client_index * 3) % 45
            is_hit = day_offset in hit_days and day_offset not in daily_hit_usage
            if is_hit:
                daily_hit_usage.add(day_offset)

            table_type, scenario, event_type, sub_type, type_operation = choose_operation_kind(
                client_index,
                op_index,
                is_hit,
            )
            amount_min, amount_max = scenario["amounts"]
            amount = round(random.uniform(amount_min, amount_max), 2)
            if is_hit:
                amount = round(amount * random.uniform(1.35, 2.4), 2)
            period_start = START_DATE.replace(hour=0, minute=0, second=0, microsecond=0)
            dt = period_start + timedelta(
                days=day_offset,
                hours=8 + (op_index * 5 + client_index) % 12,
                minutes=(op_index * 11 + client_index * 7) % 60,
                seconds=(op_index * 13 + client_index * 3) % 60,
            )
            rule_name, rule_category = RULES[(client_index + op_index) % len(RULES)]
            resolution = "deny" if is_hit and (client_index + op_index) % 4 == 0 else "allow"
            reason = "education_payment_regular_hit" if client.group == "regular" else "education_payment_risk"
            operations.append(
                Operation(
                    event_id=uuid_str(),
                    client=client,
                    table_type=table_type,
                    dt=dt,
                    amount=amount,
                    event_type=event_type,
                    sub_type=sub_type,
                    type_operation=type_operation,
                    description=scenario["description"],
                    scenario=scenario,
                    is_hit=is_hit,
                    policy_action="deny" if is_hit else "allow",
                    rule_name=rule_name if is_hit else "",
                    rule_category=rule_category if is_hit else "",
                    risk_score=random.randint(650, 999) if is_hit else random.randint(40, 320),
                    resolution_last=resolution if is_hit else "",
                    reason=reason if is_hit else "",
                )
            )

    return sorted(operations, key=lambda item: (item.dt, item.client.user_id, item.event_id))


def fill_common_client_fields(row: dict[str, Any], client: ClientProfile) -> None:
    """Заполняет общие клиентские поля, если они присутствуют в строке.

    Args:
        row: Изменяемая строка целевой таблицы.
        client: Профиль клиента.

    Returns:
        None.
    """

    mapping = {
        "user_id": client.user_id,
        "epk_id": client.epk_id,
        "fio": client.fio,
        "last_name": client.last_name,
        "first_name": client.first_name,
        "middle_name": client.middle_name,
        "client_lastname": client.last_name,
        "client_firstname": client.first_name,
        "client_patronymicname": client.middle_name,
        "client_id_document_number": client.dul_number,
        "dul_number": client.dul_number,
        "client_inn": client.inn,
        "payer_inn": client.inn,
        "card_number": client.card_number,
        "client_card_number": client.card_number,
        "payer_card_number": client.card_number,
        "phone": client.phone,
        "client_phone": client.phone,
        "mobile_phone_number": client.phone_masked,
        "client_phone_number": client.phone_masked,
        "transaction_sender_account_number": client.account_number,
        "payer_account_number": client.account_number,
        "p2p_sender_account_number": client.account_number,
        "number_acc": client.number_acc,
        "age": str(client.age),
        "age_category": client.age_category,
        "segment": client.segment,
        "client_birthdate": client.client_birthdate,
        "birth_date_client": client.birth_date_client,
        "hardware_id": client.hardware_id,
        "os_id": client.os_id,
        "user_login_id": client.user_login_id,
    }
    for column, value in mapping.items():
        if column in row:
            row[column] = value
    if "dul_number" in row and "event_dttm_readable" in row:
        row["dul_number"] = client.dul_number_uko


def make_cards_row(template: dict[str, Any], op: Operation, index: int) -> dict[str, Any]:
    """Создает строку карточной raw-таблицы.

    Args:
        template: Шаблонная строка с полным набором колонок.
        op: Операция, для которой создается строка.
        index: Значение технического поля index.

    Returns:
        Словарь строки `csp_afpc_sss_inc.cards_event`.
    """

    row = dict(template)
    fill_common_client_fields(row, op.client)
    row.update(
        {
            "index": index,
            "event_id": op.event_id,
            "epk_id": op.client.epk_id,
            "event_type": op.event_type,
            "sub_type": op.sub_type,
            "type_operation": op.type_operation,
            "client_transaction_id": hex_id(44),
            "card_owner": op.client.fio,
            "event_description": op.description,
            "event_channel": "ISSUER",
            "transaction_amount": f"{op.amount:.2f}",
            "transaction_amount_in_rub": f"{op.amount:.2f}",
            "transaction_amount_currency": "RUB",
            "transaction_beneficiar_account_number": digits(20),
            "atm_merchant_name": op.scenario["merchant"],
            "atm_mcc": op.scenario["mcc"],
            "atm_mcc_name": op.scenario["mcc_name"],
            "atm_city": "MOSCOW",
            "atm_address": f"MOSCOW, {op.scenario['merchant']}, online",
            "atm_country": "RU",
            "atm_acquiring_country": "RU",
            "user_ip_location_country": "RU",
            "user_ip_location_city": "Moscow",
            "token_device_ip": f"95.31.{random.randint(1, 254)}.{random.randint(1, 254)}",
            "atm_terminal_id": "T" + digits(9),
            "atm_id": "ATM" + digits(7),
            "atm_merchant_id": "M" + digits(9),
            "atm_acquiring_iic": digits(6),
            "card_bin": op.client.card_number[:6],
            "card_type": random.choice(["DC", "CC"]),
            "card_ps": random.choice(["MIR", "VISA", "MASTERCARD"]),
            "card_brand": "SBERCARD",
            "time_transaction_local": op.dt.strftime("%H:%M:%S"),
            "data_transaction_local": op.dt.strftime("%d.%m"),
            "response_code": "05" if op.is_hit and op.resolution_last == "deny" else "00",
            "cards_response_code_1": "05" if op.is_hit and op.resolution_last == "deny" else "00",
            "cards_dsl_model_risk_score": str(op.risk_score),
            "cards_dsl_model_receiver_score": f"{random.random():.6f}",
            "cards_dsl_nspk_fraud_score": f"{random.random():.6f}",
            "cards_client_markers": "---|SUSP|---|---|----|---|---|---|-|---|---|---|---|---|---|---|education_payment|---|---|---|---|---|pos_CRD|---|adjOUT|-----------" if op.is_hit else "---|NEP|---|---|----|---|---|---|-|---|---|---|---|---|---|---|education_payment|---|---|---|---|---|pos_CRD|---|adjOUT|-----------",
            "cards_fs_comprpid_marker": "HIGH_RISK" if op.is_hit else "NORM",
            "phone_os": "Android",
            "version_mp": "16.13.0",
            "channel_ext_system": "CARDS",
            "own_dt": event_dt(op.dt),
            "event_dt": event_dt(op.dt),
            "event_time": op.dt.strftime("%Y-%m-%d %H:%M:%S"),
        }
    )
    return row


def make_uko_row(template: dict[str, Any], op: Operation, index: int) -> dict[str, Any]:
    """Создает строку UKO/DBO raw-таблицы.

    Args:
        template: Шаблонная строка с полным набором колонок.
        op: Операция, для которой создается строка.
        index: Значение технического поля index.

    Returns:
        Словарь строки `csp_afpc_sss_inc.uko_event`.
    """

    row = dict(template)
    fill_common_client_fields(row, op.client)
    bank, bik = BANKS[(int(op.client.user_id[-1]) + op.dt.day) % len(BANKS)]
    recipient_fio, nickname, payee_phone = RECIPIENTS[(op.dt.day + op.dt.hour) % len(RECIPIENTS)]
    epoch_ms = int(op.dt.replace(tzinfo=timezone.utc).timestamp() * 1000)
    row.update(
        {
            "index": index,
            "event_id": op.event_id,
            "event_time": str(epoch_ms),
            "event_dttm_readable": op.dt.strftime("%Y-%m-%d %H:%M:%S"),
            "event_dt": event_dt(op.dt),
            "load_dt": event_dt(op.dt),
            "own_dttm": (op.dt + timedelta(minutes=17)).strftime("%Y-%m-%d %H:%M:%S.%f"),
            "event_channel": "MOBILE",
            "sub_channel": "UFS.MOBILEAPI",
            "event_type": op.event_type,
            "sub_type": op.sub_type,
            "type_operation": op.type_operation,
            "event_description": op.description,
            "transaction_amount": f"{op.amount:.2f}",
            "transaction_amount_currency": "RUB",
            "transaction_beneficiar_account_number": digits(20),
            "transaction_beneficiar_bik": bik,
            "recipient_bank_name": bank,
            "payee_phone_number": payee_phone,
            "recepient_fio": recipient_fio,
            "transaction_beneficiar_nick_name": nickname,
            "operation_id": hex_id(24),
            "member_id": digits(14),
            "sbp_id": hex_id(24) if op.type_operation == "PAYMENT_SBP" else "",
            "user_ip_location_country_code": "RU",
            "user_ip_location_city": "Moscow",
            "user_ip_location_region": "MOW",
            "ip_device": f"95.31.{random.randint(1, 254)}.{random.randint(1, 254)}",
            "longitude_ip": "37.6171",
            "latitude_ip": "55.7483",
            "device_time": "AndroidT16",
            "app_version": "16.13.0",
            "name_os": "Android",
            "phone_brand": "Samsung Galaxy S23",
            "phone_model": "Samsung Galaxy S23",
            "segment_client": "2",
            "client_groups": "95,2",
            "user_mobile_hardware_id_days_since_first_hit": str(random.randint(10, 220)),
            "device_mobile_days_since_first_hit": str(random.randint(10, 220)),
            "payment_new_ip_provider": str(random.randint(1000, 9999)),
            "device_source_sdk": js(
                {
                    "AccessibilityServices": {"enabled": False},
                    "AgentAppInfo": "SberBank 16.13.0 arm64-v8a",
                    "AgentBrand": "Samsung Galaxy S23",
                    "AgentConnectionType": random.choice(["4G", "5G", "WiFi"]),
                    "Compromised": 0,
                    "Debugger": 0,
                    "DeveloperTools": 0,
                    "DeviceModel": "Samsung Galaxy S23",
                    "DeviceSystemName": "Android",
                    "DeviceSystemVersion": "14",
                    "HardwareID": op.client.hardware_id,
                    "OS_ID": op.client.os_id,
                    "TIMESTAMP": op.dt.strftime("%Y-%m-%dT%H:%M:%SZ"),
                    "TimeZone": "MSK",
                    "VpnConnection": False,
                }
            ),
            "final_marker_payer": "B20|U00|---|---|----|---|--------|-------|---|RM31|---|---|--|------|RI00|---|-C2F-|-F2C-|-C2T-|-T2C-|Y0304|----|---|---|-----|--|N00|----|--|----|----|----|-------",
            "tfm_client_marker": "G73|B20|ABS-------|---|M050|----|---|---------|----|--------|Y0203|edu|-------|------|U00|RI00|DIGIT01110|----------|---------|TFM_EPK|CBCUR0|ABREL--|-----------",
            "client_made_payment_to_recipient": "true",
            "client_accepted_transfer_to_recipient_ignite": "ДА" if not op.is_hit else "НЕТ",
            "main_rule": js({"rule_name": op.rule_name, "rule_id": uuid_str(), "rule_category": op.rule_category}) if op.is_hit else "",
            "rules": js([op.rule_name]) if op.is_hit else "[]",
            "subrules": "[]",
            "risk_score_dsl": str(op.risk_score),
            "kafka_input_time": str(int(op.dt.replace(tzinfo=timezone.utc).timestamp()) + 20),
            "kafka_output_time": str(int(op.dt.replace(tzinfo=timezone.utc).timestamp()) + 80),
            "indicators_vk_max": "{}",
            "scoring_oss": "{}",
            "indicators_sbp": "{}",
            "params": js({"demo_client_trace": True, "synthetic_education_dataset": True}),
        }
    )
    return row


def make_hits_row(
    template: dict[str, Any],
    op: Operation,
    index: int,
    client_hits: list[Operation],
) -> dict[str, Any]:
    """Создает строку таблицы сработок Matrix 2.0.

    Args:
        template: Шаблонная строка с полным набором колонок.
        op: Операция-сработка.
        index: Значение технического поля index.
        client_hits: Все сработки клиента для расчета history-признаков.

    Returns:
        Словарь строки `hits_extra_info_129372427_view`.
    """

    row = dict(template)
    fill_common_client_fields(row, op.client)
    bank, bik = BANKS[(int(op.client.user_id[-1]) + op.dt.day) % len(BANKS)]
    recipient_fio, nickname, payee_phone = RECIPIENTS[(op.dt.day + op.dt.hour) % len(RECIPIENTS)]
    previous_hits_72h = [
        hit for hit in client_hits
        if hit.dt <= op.dt and timedelta(0) <= op.dt - hit.dt <= timedelta(hours=72)
    ]
    days_with_hits = len({hit.dt.date() for hit in client_hits})
    is_card = op.table_type == "cards"
    is_save = op.resolution_last == "deny"
    previous_events = [
        "hit|SB|SP|0_1p|0_1h" if len(previous_hits_72h) > 1 else "p2p_deposit_sbp|--|--|--|48_96h",
        "atm_deposit_cash|--|--|--|12_24h",
    ]
    posterious_events = [
        "PURCHASE|--|OLD|0_1p|+|12_24h|SC|CARDS|main_reason",
        "P2P|--|OLD|15_20p|-|24_48h|--|DBO|--",
    ]
    row.update(
        {
            "index": index,
            "event_time": op.dt.strftime("%Y-%m-%d %H:%M:%S"),
            "event_id": op.event_id,
            "transaction_amount": f"{op.amount:.2f}",
            "transaction_amount_in_rub": f"{op.amount:.2f}",
            "client_balance": f"{round(op.amount + random.uniform(25000.0, 180000.0), 2):.2f}",
            "transaction_amount_currency": "RUB",
            "event_channel": "CARDS" if is_card else "MOBILE",
            "sub_channel": "ISSUER" if is_card else "UFS.MOBILEAPI",
            "event_type": op.event_type,
            "sub_type": op.sub_type,
            "type_operation": op.type_operation,
            "event_description": op.description,
            "tree_info": "",
            "policy_action": op.policy_action,
            "main_rule": js(
                {
                    "rule_name": op.rule_name,
                    "rule_id": uuid_str(),
                    "rule_category": op.rule_category,
                    "description": f"Синтетическая сработка по операции: {op.description}",
                }
            ),
            "phone_operator": "МТС",
            "region_phone_operator": "77",
            "dul_type": "21",
            "payer_transfer_type": "Карта" if is_card else "Счет",
            "payee_transfer_type": "Мерчант" if is_card else "Счет",
            "transaction_beneficiar_account_number": digits(20),
            "recipient_bik": bik,
            "payee_bank_name": bank,
            "member_id": digits(14),
            "sbp_id": hex_id(24) if op.type_operation == "PAYMENT_SBP" else "",
            "operation_id": hex_id(24),
            "recipient_info": js(
                {
                    "epk_id": None,
                    "user_id": None,
                    "fio": recipient_fio,
                    "inn": None,
                    "payee_phone_number": payee_phone,
                    "brand_name": op.scenario["merchant"],
                    "legal_name_of_service_provider": f"ООО {op.scenario['merchant']}",
                    "full_name_org": f"ООО {op.scenario['merchant']}",
                    "recipient_bank_name": bank,
                    "transaction_beneficiar_nick_name": nickname,
                }
            ),
            "card_info": js(
                {
                    "card_type": "DC",
                    "card_linked_account": op.client.account_number,
                    "momentum_flg": "false",
                    "virt_flg": "false",
                    "payment_system": "Sberbank",
                }
            ),
            "trust_info": js({"trusted_device": True, "trusted_payee": False, "device_age_days": random.randint(10, 220)}),
            "recipient_inn": digits(12) if is_card else "",
            "atm_merchant_name": op.scenario["merchant"] if is_card else "",
            "merchant_info": js(
                {
                    "merchant_group": "education",
                    "atm_mcc": op.scenario["mcc"],
                    "atm_mcc_name": op.scenario["mcc_name"],
                }
            ) if is_card else "",
            "pos_info": js({"pos_type": "ECOM", "response_code": "05" if is_save else "00"}) if is_card else "",
            "link_cf": js({"recipient_from_address_book": "НЕТ", "export_user_payee_days_since_first_hit": str(random.randint(1, 220))}),
            "mobile_sdk_info": js({"phone_brand": "Samsung Galaxy S23", "name_os": "Android", "app_version": "16.13.0"}),
            "scoring_oss": js({"phone_reciver_ScoreMTS_se_2024": f"{random.random():.6f}"}),
            "type_accept": "Black List",
            "source_type_accept": "rule",
            "resolution_first": "deny",
            "resolution_first_dttm": (op.dt + timedelta(minutes=2)).strftime("%Y-%m-%d %H:%M:%S"),
            "resolution_last": op.resolution_last,
            "resolution_last_dttm": (op.dt + timedelta(minutes=7)).strftime("%Y-%m-%d %H:%M:%S"),
            "accept_time_sec": str(random.randint(180, 420)),
            "purpose": "Покупка/Оплата услуг",
            "surface": "Карты" if is_card else "ДБО",
            "product": "Карта" if is_card else "СБОЛ МП",
            "product_type": "Образовательные услуги",
            "payment_transaction_flag": "True",
            "has_claim": "True" if index % 11 == 0 else "False",
            "is_save": "True" if is_save else "False",
            "marked_as_not_save_reason": "Сработка требует проверки" if is_save else "Операция подтверждена клиентом",
            "posterious_events": str(posterious_events),
            "previous_events": str(previous_events),
            "hits_extra_facts": js({"demo_client_trace": True, "day_n": op.dt.strftime("%Y-%m-%d"), "has_180d_history": True}),
            "posterious_events_additional_info": js({"visit_event_dt": None, "visit_event_id": None, "db_visit_dttm": None}),
            "previous_events_additional_info": js(
                {
                    "hit_cnt_per_client_72h": len(previous_hits_72h),
                    "operations_cnt_180d": OPERATIONS_PER_CLIENT,
                    "days_with_hits_180d": days_with_hits,
                }
            ),
            "own_loading_id": digits(8),
            "own_dt": event_dt(op.dt),
            "event_dt": event_dt(op.dt),
        }
    )
    return row


def make_history_row(template: dict[str, Any], op: Operation, index: int) -> dict[str, Any]:
    """Создает строку истории авторазметки для сработки.

    Args:
        template: Шаблонная строка с полным набором колонок.
        op: Операция-сработка.
        index: Значение технического поля index.

    Returns:
        Словарь строки `history_automarking`.
    """

    row = dict(template)
    fill_common_client_fields(row, op.client)
    row.update(
        {
            "index": index,
            "event_id": op.event_id.replace("-", ""),
            "source_event_id": op.event_id,
            "entity_id": op.client.epk_id,
            "event_time": op.dt.strftime("%Y-%m-%d %H:%M:%S"),
            "event_type": op.event_type,
            "sub_type": op.sub_type,
            "event_description": op.description,
            "client_transaction_id": hex_id(32),
            "atm_merchant_name": op.scenario["merchant"],
            "atm_terminal_id": "T" + digits(9),
            "atm_mcc": op.scenario["mcc"],
            "terbank_code": "99",
            "atm_city": "MOSCOW",
            "atm_address": f"MOSCOW, {op.scenario['merchant']}, online",
            "risk_score": str(op.risk_score),
            "transaction_amount": f"{op.amount:.2f}",
            "transaction_amount_currency": "RUB",
            "rule_name": op.rule_name,
            "rule_num": digits(5),
            "rule_order": digits(4),
            "mark": random.choice(["F", "L", "G"]),
            "mcc_group": "E",
            "resolution": op.resolution_last,
            "sub_channel": "ISSUER" if op.table_type == "cards" else "UFS.MOBILEAPI",
            "status": "processed",
            "reason": op.reason,
            "atm_country": "RU",
            "atm_acquiring_country": "RU",
            "atm_acquiring_iic": digits(6),
            "marking_time": (op.dt + timedelta(days=1, minutes=20)).strftime("%Y-%m-%d %H:%M:%S.%f")[:-3],
            "is_tech_rule": "False",
            "own_loading_id": digits(8),
            "own_dt": event_dt(op.dt),
            "load_dt": event_dt(op.dt),
        }
    )
    return row


def build_tables(templates: TableTemplates, operations: list[Operation]) -> dict[str, pd.DataFrame]:
    """Собирает пять связанных CSV-таблиц из списка операций.

    Args:
        templates: Набор исходных схем и шаблонных строк.
        operations: Список raw-операций всех клиентов.

    Returns:
        Словарь имя CSV-файла -> DataFrame.
    """

    card_template = templates.cards.iloc[0].to_dict()
    uko_template = templates.uko.iloc[0].to_dict()
    card_hit_template_df = templates.hits[templates.hits["event_channel"].astype(str).eq("CARDS")]
    uko_hit_template_df = templates.hits[templates.hits["event_channel"].astype(str).eq("MOBILE")]
    card_hit_template = (card_hit_template_df.iloc[0] if not card_hit_template_df.empty else templates.hits.iloc[0]).to_dict()
    uko_hit_template = (uko_hit_template_df.iloc[0] if not uko_hit_template_df.empty else templates.hits.iloc[0]).to_dict()
    history_template = templates.history.iloc[0].to_dict()

    client_hits: dict[str, list[Operation]] = {}
    for op in operations:
        if op.is_hit:
            client_hits.setdefault(op.client.user_id, []).append(op)
    for hits in client_hits.values():
        hits.sort(key=lambda item: item.dt)

    cards_rows: list[dict[str, Any]] = []
    uko_rows: list[dict[str, Any]] = []
    hits_rows: list[dict[str, Any]] = []
    history_rows: list[dict[str, Any]] = []
    timeline_rows: list[dict[str, Any]] = []

    for op in operations:
        if op.table_type == "cards":
            cards_rows.append(make_cards_row(card_template, op, len(cards_rows)))
            hit_template = card_hit_template
        else:
            uko_rows.append(make_uko_row(uko_template, op, len(uko_rows)))
            hit_template = uko_hit_template

        if op.is_hit:
            hits_rows.append(make_hits_row(hit_template, op, len(hits_rows), client_hits[op.client.user_id]))
            history_rows.append(make_history_row(history_template, op, len(history_rows)))

        timeline_rows.append(
            {
                "event_time": op.dt.strftime("%Y-%m-%d %H:%M:%S"),
                "event_id": op.event_id,
                "target_table": "cards_event" if op.table_type == "cards" else "uko_event",
                "is_hit": str(op.is_hit),
                "event_type": op.event_type,
                "sub_type": op.sub_type,
                "type_operation": op.type_operation,
                "amount": f"{op.amount:.2f}",
                "policy_action": op.policy_action if op.is_hit else "",
                "resolution_last": op.resolution_last if op.is_hit else "",
                "risk_score": str(op.risk_score) if op.is_hit else "",
                "rule": op.rule_name if op.is_hit else "",
                "reason": op.reason if op.is_hit else "",
            }
        )

    tables = {
        HITS_FILE: pd.DataFrame(hits_rows, columns=templates.hits.columns),
        CARDS_FILE: pd.DataFrame(cards_rows, columns=templates.cards.columns),
        UKO_FILE: pd.DataFrame(uko_rows, columns=templates.uko.columns),
        HISTORY_FILE: pd.DataFrame(history_rows, columns=templates.history.columns),
        TIMELINE_FILE: pd.DataFrame(timeline_rows, columns=templates.timeline.columns),
    }
    return tables


def validate_tables(tables: dict[str, pd.DataFrame], clients: list[ClientProfile], templates: TableTemplates) -> dict[str, Any]:
    """Проверяет схемы и контрольные метрики сгенерированного датасета.

    Args:
        tables: Словарь сгенерированных DataFrame.
        clients: Список профилей клиентов.
        templates: Исходные схемы для сравнения порядка колонок.

    Returns:
        Словарь с контрольными метриками генерации.
    """

    expected_columns = {
        HITS_FILE: list(templates.hits.columns),
        CARDS_FILE: list(templates.cards.columns),
        UKO_FILE: list(templates.uko.columns),
        HISTORY_FILE: list(templates.history.columns),
        TIMELINE_FILE: list(templates.timeline.columns),
    }
    for name, expected in expected_columns.items():
        actual = list(tables[name].columns)
        if actual != expected:
            raise AssertionError(f"Схема {name} изменилась: {actual} != {expected}")

    hits = tables[HITS_FILE]
    cards = tables[CARDS_FILE]
    uko = tables[UKO_FILE]
    history = tables[HISTORY_FILE]
    timeline = tables[TIMELINE_FILE]

    raw = pd.concat(
        [
            cards[["event_id", "user_id"]].assign(raw_table="cards"),
            uko[["event_id", "user_id"]].assign(raw_table="uko"),
        ],
        ignore_index=True,
    )
    raw_counts = raw.groupby("user_id")["event_id"].count().to_dict()
    expected_users = {client.user_id for client in clients}
    if set(raw_counts) != expected_users:
        raise AssertionError("В raw-таблицах найден некорректный набор клиентов")
    if any(count != OPERATIONS_PER_CLIENT for count in raw_counts.values()):
        raise AssertionError(f"У клиента должно быть ровно {OPERATIONS_PER_CLIENT} raw-операций: {raw_counts}")
    if len(timeline) != OPERATIONS_PER_CLIENT * len(clients):
        raise AssertionError("Некорректное число строк timeline")
    if len(hits) != 87 or len(history) != 87:
        raise AssertionError("Некорректное число сработок или строк авторазметки")

    hit_ids = set(hits["event_id"].astype(str))
    card_ids = set(cards["event_id"].astype(str))
    uko_ids = set(uko["event_id"].astype(str))
    exactly_one = sum((event_id in card_ids) ^ (event_id in uko_ids) for event_id in hit_ids)
    if exactly_one != len(hit_ids):
        raise AssertionError("Не каждая сработка найдена ровно в одной raw-таблице")
    if set(history["source_event_id"].astype(str)) != hit_ids:
        raise AssertionError("История авторазметки не совпадает со сработками")

    hits_by_client = hits.groupby("user_id")["event_id"].count().to_dict()
    group_counts = {client.group: [] for client in clients}
    for client in clients:
        group_counts[client.group].append(int(hits_by_client.get(client.user_id, 0)))
    if sorted(group_counts["spread"]) != [8, 8, 8, 8, 8]:
        raise AssertionError(f"Некорректные spread-сработки: {group_counts['spread']}")
    if sorted(group_counts["single"]) != [1, 1]:
        raise AssertionError(f"Некорректные single-сработки: {group_counts['single']}")
    if sorted(group_counts["regular"]) != [15, 15, 15]:
        raise AssertionError(f"Некорректные regular-сработки: {group_counts['regular']}")

    education_pattern = "обуч|курс|учеб|экзам|репетитор|образ"
    raw_descriptions = pd.concat([cards["event_description"], uko["event_description"]], ignore_index=True)
    education_hits = hits["event_description"].astype(str).str.contains(education_pattern, case=False, regex=True).sum()
    education_raw = raw_descriptions.astype(str).str.contains(education_pattern, case=False, regex=True).sum()
    if education_hits == 0 or education_raw == 0:
        raise AssertionError("Не найдены образовательные операции")

    return {
        "clients": len(clients),
        "raw_rows": len(raw),
        "timeline_rows": len(timeline),
        "hits_rows": len(hits),
        "history_rows": len(history),
        "cards_rows": len(cards),
        "uko_rows": len(uko),
        "hits_by_group": {group: sorted(values) for group, values in group_counts.items()},
        "old_epk_hits": int((hits["epk_id"].astype(str) == "2099007770421986000001").sum()),
    }


def write_tables(tables: dict[str, pd.DataFrame], data_dir: Path) -> None:
    """Записывает сгенерированные таблицы в каталог examples/data.

    Args:
        tables: Словарь имя CSV-файла -> DataFrame.
        data_dir: Каталог для записи CSV.

    Returns:
        None.
    """

    for name, df in tables.items():
        df.to_csv(data_dir / name, index=False, quoting=csv.QUOTE_MINIMAL, encoding="utf-8")


def main() -> None:
    """Генерирует и записывает плотный синтетический antifraud-датасет.

    Args:
        Отсутствуют.

    Returns:
        None.
    """

    random.seed(RANDOM_SEED)
    templates = read_templates(DATA_DIR)
    clients = build_clients()
    operations = build_operations(clients)
    tables = build_tables(templates, operations)
    report = validate_tables(tables, clients, templates)
    write_tables(tables, DATA_DIR)
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
