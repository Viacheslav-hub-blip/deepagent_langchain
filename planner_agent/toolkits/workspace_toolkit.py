"""
workspace_tools.py
==================
Модуль инструментов рабочего пространства для LangChain-агента.

Предоставляет набор async-инструментов (tools) для работы с файлами,
датафреймами и контекстными документами в изолированной sandbox-среде.

Минимальный набор инструментов, отдаваемый worker-у сейчас:
    - list_loaded_dataframes_in_virtual_environment — список датафреймов в sandbox
    - show_current_dataframe                        — предпросмотр активного датафрейма
    - workspace_read_file                           — чтение текстового файла/части файла
    - workspace_write_file                          — запись текстового файла
    - save_simple_variable_in_virtual_env_to_file   — сохранение переменной в файл
    - load_additional_context                       — загрузка контекстного документа
    - load_additional_source_database_table         — preview/load источника данных
"""

import json
import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import aiofiles
import pandas as pd
from langchain_core.tools import BaseTool, tool

from ..runtime.sandbox import PythonSandboxProtocol

#: Расширения файлов, распознаваемых как датафреймы
DATAFRAME_EXTENSIONS: frozenset[str] = frozenset(
    {".csv", ".parquet", ".pq", ".json", ".jsonl", ".pkl", ".pickle"}
)

#: Расширения текстовых контекстных файлов
CONTEXT_EXTENSIONS: frozenset[str] = frozenset({".txt", ".md"})

#: Расширения файлов, сохраняемых как текст
TEXT_EXTENSIONS: frozenset[str] = frozenset({".txt", ".md", ".log"})

#: Кодировки, перебираемые при чтении текстового файла
TEXT_ENCODINGS: tuple[str, ...] = ("utf-8", "utf-8-sig", "cp1251")

EXECUTABLE_DENYLIST: frozenset[str] = frozenset(
    {
        ".exe",
        ".dll",
        ".bat",
        ".cmd",
        ".ps1",
        ".sh",
        ".com",
        ".scr",
        ".msi",
        ".vbs",
        ".js",
        ".jar",
        ".pyz",
        ".app",
        ".dmg",
    }
)

#: Поддиректория источников данных по умолчанию
DEFAULT_SOURCES_SUBDIR: str = "sources"

#: Поддиректория контекстных файлов по умолчанию
DEFAULT_CONTEXTS_SUBDIR: str = "contexts"

#: Имя переменной активного датафрейма по умолчанию
DEFAULT_CURRENT_DF_VAR: str = "df_current"

#: Имя переменной дополнительного датафрейма по умолчанию
DEFAULT_ADDITIONAL_DF_VAR: str = "df_additional"

#: Максимальное число строк при предпросмотре активного датафрейма
MAX_PREVIEW_ROWS_CURRENT: int = 100

#: Максимальное число строк при предпросмотре источника данных
MAX_PREVIEW_ROWS_SOURCE: int = 200

MAX_TEXT_READ_CHARS: int = 8_000

DEFAULT_WORKSPACE_TOOL_NAMES: frozenset[str] = frozenset(
    {
        "list_loaded_dataframes_in_virtual_environment",
        "show_current_dataframe",
        "workspace_read_file",
        "workspace_write_file",
        "save_simple_variable_in_virtual_env_to_file",
        "load_additional_context",
        "load_additional_source_database_table",
    }
)


@dataclass
class WorkspaceRuntime:
    """
    Контейнер состояния рабочего пространства агента.

    Attributes:
        sandbox:            Изолированная Python-среда выполнения.
        workspace_root:     Корневая директория рабочего пространства.
        sources_dir:        Директория с исходными наборами данных.
        contexts_dir:       Директория с контекстными документами.
        allowed_roots:      Разрешённые корневые пути для доступа к файлам.
        current_dataframe:  Имя переменной активного датафрейма (опционально).
    """

    sandbox: PythonSandboxProtocol
    workspace_root: Path
    sources_dir: Path
    contexts_dir: Path
    allowed_roots: tuple[Path, ...]
    current_dataframe: Optional[str] = None


class WorkspacePathError(ValueError):
    """Исключение, возникающее при обращении к пути вне разрешённых корней."""


def _is_subpath(path: Path, root: Path) -> bool:
    """
    Проверяет, является ли `path` подпутём `root`.

    Args:
        path: Проверяемый путь.
        root: Корневой путь.

    Returns:
        True, если `path` находится внутри `root`, иначе False.
    """
    try:
        path.resolve().relative_to(root.resolve())
        return True
    except Exception:
        return False


def _is_path_allowed(path: Path, allowed_roots: tuple[Path, ...]) -> bool:
    """
    Проверяет, разрешён ли доступ к пути согласно списку корневых директорий.

    Args:
        path:          Проверяемый путь.
        allowed_roots: Кортеж разрешённых корневых директорий.

    Returns:
        True, если путь находится внутри хотя бы одного из разрешённых корней.
    """
    return any(_is_subpath(path, root) for root in allowed_roots)


def _resolve_runtime_dir(
        workspace_root: Path,
        directory: Optional[str],
        default_subdir: str,
) -> Path:
    """
    Определяет абсолютный путь к рабочей директории.

    Если `directory` не задан, возвращает `workspace_root / default_subdir`.
    Абсолютные пути используются как есть, относительные — разрешаются
    относительно `workspace_root`.

    Args:
        workspace_root: Корень рабочего пространства.
        directory:      Явно указанная директория (опционально).
        default_subdir: Имя поддиректории по умолчанию.

    Returns:
        Абсолютный путь к директории.
    """
    if directory is None or not str(directory).strip():
        return (workspace_root / default_subdir).resolve()

    candidate = Path(directory)
    if candidate.is_absolute():
        return candidate.resolve()
    return (workspace_root / candidate).resolve()


def _resolve_path(
        runtime: WorkspaceRuntime,
        input_path: str,
        base_dir: Optional[Path] = None,
) -> Path:
    """
    Разрешает путь к файлу и проверяет его допустимость.

    Относительные пути разрешаются относительно `base_dir` (или
    `workspace_root`, если `base_dir` не задан). После разрешения путь
    проверяется на принадлежность к `allowed_roots`.

    Args:
        runtime:    Состояние рабочего пространства.
        input_path: Путь к файлу (абсолютный или относительный).
        base_dir:   Базовая директория для относительных путей (опционально).

    Returns:
        Абсолютный разрешённый путь.

    Raises:
        WorkspacePathError: Если путь выходит за пределы разрешённых корней.
    """
    candidate = Path(input_path)
    if not candidate.is_absolute():
        start = base_dir if base_dir is not None else runtime.workspace_root
        candidate = start / candidate

    resolved = candidate.resolve()
    if not _is_path_allowed(resolved, runtime.allowed_roots):
        allowed = ", ".join(str(root) for root in runtime.allowed_roots)
        raise WorkspacePathError(
            f"Path is outside allowed roots: {input_path}. Allowed roots: {allowed}"
        )
    return resolved


def _is_executable_path(path: Path) -> bool:
    return path.suffix.lower() in EXECUTABLE_DENYLIST


def _load_dataframe(path: Path) -> pd.DataFrame:
    """
    Загружает датафрейм из файла по его расширению.

    Args:
        path: Путь к файлу датафрейма.

    Returns:
        Загруженный объект pd.DataFrame.

    Raises:
        ValueError: Если расширение файла не поддерживается.
    """
    suffix = path.suffix.lower()
    if suffix == ".csv":
        return pd.read_csv(path)
    if suffix in {".parquet", ".pq"}:
        return pd.read_parquet(path)
    if suffix in {".json", ".jsonl"}:
        return pd.read_json(path)
    if suffix in {".pkl", ".pickle"}:
        return pd.read_pickle(path)
    raise ValueError(f"Unsupported dataframe format: {suffix}")


def _save_dataframe(df: pd.DataFrame, path: Path) -> None:
    """
    Сохраняет датафрейм в файл по его расширению.

    Args:
        df:   Сохраняемый датафрейм.
        path: Путь к целевому файлу.

    Raises:
        ValueError: Если расширение файла не поддерживается.
    """
    suffix = path.suffix.lower()
    if suffix == ".csv":
        df.to_csv(path, index=False)
        return
    if suffix in {".parquet", ".pq"}:
        df.to_parquet(path, index=False)
        return
    if suffix in {".json", ".jsonl"}:
        df.to_json(path, orient="records", force_ascii=False)
        return
    if suffix in {".pkl", ".pickle"}:
        df.to_pickle(path)
        return
    raise ValueError(f"Unsupported dataframe output format: {suffix}")


def _build_allowed_roots(
        workspace_root: Path,
        sources_dir: Path,
        contexts_dir: Path,
) -> tuple[Path, ...]:
    """
    Формирует дедуплицированный кортеж разрешённых корневых директорий.

    Args:
        workspace_root: Корень рабочего пространства.
        sources_dir:    Директория источников данных.
        contexts_dir:   Директория контекстных файлов.

    Returns:
        Кортеж уникальных абсолютных путей разрешённых корней.
    """
    roots: list[Path] = []
    for root in (
            workspace_root.resolve(),
            sources_dir.resolve(),
            contexts_dir.resolve(),
    ):
        if root not in roots:
            roots.append(root)
    return tuple(roots)


def _list_context_files(contexts_dir: Path) -> list[str]:
    """
    Возвращает отсортированный список имён контекстных файлов в директории.

    Args:
        contexts_dir: Директория с контекстными файлами.

    Returns:
        Список имён файлов с расширениями из CONTEXT_EXTENSIONS.
    """
    return sorted(
        p.name
        for p in contexts_dir.glob("*")
        if p.is_file() and p.suffix.lower() in CONTEXT_EXTENSIONS
    )


def _resolve_context_file(runtime: WorkspaceRuntime, context_file: str) -> Path:
    """
    Находит контекстный файл в директории `contexts_dir` с учётом
    возможного отсутствия расширения и регистронезависимого сравнения.

    Args:
        runtime:      Состояние рабочего пространства.
        context_file: Имя или путь к контекстному файлу.

    Returns:
        Абсолютный путь к найденному файлу.

    Raises:
        FileNotFoundError: Если файл не найден ни одним из способов.
    """
    candidate = _resolve_path(runtime, context_file, base_dir=runtime.contexts_dir)
    if candidate.exists() and candidate.is_file():
        return candidate

    requested = Path(context_file)
    if requested.suffix == "":
        for suffix in CONTEXT_EXTENSIONS:
            try_path = _resolve_path(
                runtime,
                f"{context_file}{suffix}",
                base_dir=runtime.contexts_dir,
            )
            if try_path.exists() and try_path.is_file():
                return try_path

    target_name = requested.name.lower()
    target_stem = requested.stem.lower()
    for item in runtime.contexts_dir.glob("*"):
        if not item.is_file():
            continue
        name = item.name.lower()
        stem = item.stem.lower()
        if name == target_name or stem == target_name or stem == target_stem:
            resolved = _resolve_path(
                runtime, str(item), base_dir=runtime.contexts_dir
            )
            if resolved.exists() and resolved.is_file():
                return resolved

    raise FileNotFoundError(context_file)


async def _read_text_with_fallback(path: Path) -> str:
    """
    Асинхронно читает текстовый файл, последовательно перебирая кодировки
    из TEXT_ENCODINGS. Если ни одна не подошла — читает с игнорированием
    ошибок декодирования.

    Args:
        path: Путь к текстовому файлу.

    Returns:
        Содержимое файла в виде строки.
    """
    for encoding in TEXT_ENCODINGS:
        try:
            async with aiofiles.open(path, encoding=encoding) as fh:
                return await fh.read()
        except UnicodeDecodeError:
            continue
    # Финальный fallback: читаем с игнорированием ошибок
    async with aiofiles.open(path, encoding="utf-8", errors="ignore") as fh:
        return await fh.read()


# ---------------------------------------------------------------------------
# Фабрика инструментов
# ---------------------------------------------------------------------------


def build_workspace_tools(
        sandbox: PythonSandboxProtocol,
        workspace_root: str = ".",
        sources_dir: Optional[str] = None,
        contexts_dir: Optional[str] = None,
        enabled_tool_names: Optional[set[str]] = None,
) -> list[BaseTool]:
    """
    Создаёт и возвращает список LangChain-инструментов для работы с
    рабочим пространством агента.

    Args:
        sandbox:        Изолированная среда выполнения Python-кода.
        workspace_root: Путь к корневой директории рабочего пространства.
        sources_dir:    Путь к директории источников данных (опционально).
        contexts_dir:   Путь к директории контекстных документов (опционально).
        enabled_tool_names: Явный список инструментов для выдачи worker-у.
            По умолчанию используется минимальный MVP surface.

    Returns:
        Список инициализированных инструментов (BaseTool).
    """
    workspace_root_path = Path(workspace_root).resolve()
    resolved_sources_dir = _resolve_runtime_dir(
        workspace_root=workspace_root_path,
        directory=sources_dir,
        default_subdir=DEFAULT_SOURCES_SUBDIR,
    )
    resolved_contexts_dir = _resolve_runtime_dir(
        workspace_root=workspace_root_path,
        directory=contexts_dir,
        default_subdir=DEFAULT_CONTEXTS_SUBDIR,
    )

    runtime = WorkspaceRuntime(
        sandbox=sandbox,
        workspace_root=workspace_root_path,
        sources_dir=resolved_sources_dir,
        contexts_dir=resolved_contexts_dir,
        allowed_roots=_build_allowed_roots(
            workspace_root=workspace_root_path,
            sources_dir=resolved_sources_dir,
            contexts_dir=resolved_contexts_dir,
        ),
    )

    runtime.workspace_root.mkdir(parents=True, exist_ok=True)
    runtime.sources_dir.mkdir(parents=True, exist_ok=True)
    runtime.contexts_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Инструменты
    # ------------------------------------------------------------------

    @tool("get_list_files_from_root_directory")
    async def get_list_files_from_root_directory() -> str:
        """
        Используйте этот инструмент только для того, чтобы получить список
        сохранённых файлов, созданных во время работы агента.
        Эти данные могут содержать промежуточные наборы данных.

        Загружает список файлов с расширениями датафреймов из рабочей папки
        агента и возвращает их имена.
        """
        files = [
            p.name
            for p in runtime.workspace_root.glob("*")
            if p.is_file() and p.suffix.lower() in DATAFRAME_EXTENSIONS
        ]
        if not files:
            return "В рабочей директории нет наборов данных. Попробуйте проверить виртуальное окружение или пропустить шаг"
        return "\n".join(sorted(files))

    @tool("replace_current_dataframe")
    async def replace_current_dataframe(
            dataframe_file: str,
            variable_name: str = DEFAULT_CURRENT_DF_VAR,
    ) -> str:
        """
        Используйте этот инструмент, когда вы хотите заменить текущий
        dataframe (df_current) на dataframe из runtime-памяти агента (рабочей директории).

        Загружает набор данных по названию файла и заменяет активный
        dataframe в памяти.
        """
        path = _resolve_path(runtime, dataframe_file, base_dir=runtime.workspace_root)
        if not path.exists() or not path.is_file():
            return f"Dataframe не найден в рабочей директории: {dataframe_file}"

        df = _load_dataframe(path)
        await runtime.sandbox.add_variable(variable_name, df)
        runtime.current_dataframe = variable_name
        runtime.sandbox.last_dataframe_variable = variable_name

        return (
            f"Dataframe '{path.name}' загружен в переменную '{variable_name}'. Он находится в вирутальном окружении"
            f"Размерность={df.shape}. Названия колонок={list(df.columns)}"
        )

    @tool("list_loaded_dataframes_in_virtual_environment")
    async def list_loaded_dataframes_in_virtual_environment() -> str:
        """
        Используйте этот инструмент, чтобы показать список dataframes
        в текущем виртуальном окружении.

        Возвращает список имён загруженных наборов данных в виртуальном
        окружении.
	Название инструмента:
	list_loaded_dataframes_in_virtual_environment
        """
        loaded = [
            (
                f"Название переменной {name}: "
                f"размерность={value.shape} "
                f"Превью: {value.head(1).to_string()}"
            )
            for name, value in runtime.sandbox.globals.items()
            if isinstance(value, pd.DataFrame)
        ]
        if not loaded:
            return "Ни один dataframe не загружен в виртуальное окружение."
        return "\n".join(sorted(loaded))

    @tool("show_current_dataframe")
    async def show_current_dataframe(rows: int = 5) -> str:
        """
        Используйте этот инструмент, когда вам необходимо посмотреть
        первые n строк текущего (активного) dataframe (по умолчанию всегда
        имеет название переменной df_current), с которым в данный
        момент работает пользователь.

        Возвращает строку с описанием и предпросмотром активного dataframe.
        """
        current_name = (
                runtime.current_dataframe or runtime.sandbox.last_dataframe_variable
        )
        if not current_name:
            return "Current dataframe is not set."

        value = await runtime.sandbox.get_variable(current_name)
        if value is None or not isinstance(value, pd.DataFrame):
            return (
                f"Current dataframe variable '{current_name}' "
                "is missing or not a DataFrame."
            )

        rows = max(1, min(rows, MAX_PREVIEW_ROWS_CURRENT))
        preview = value.head(rows).to_string()
        return (
            f"Текущий DataFrame (df_current): {current_name}\n"
            f"Размерность: {value.shape}\n"
            f"Первые строки (превью):\n{preview}"
        )

    @tool("workspace_read_file")
    async def workspace_read_file(
            file_path: str,
            offset: int = 0,
            max_chars: int = MAX_TEXT_READ_CHARS,
    ) -> str:
        """
        Читает текстовый файл внутри разрешённых директорий workspace.

        Используйте offset и max_chars, чтобы читать большие файлы частями и
        не загружать весь файл в контекст.
        """
        path = _resolve_path(runtime, file_path)
        if not path.exists() or not path.is_file():
            return f"File not found: {file_path}"
        if _is_executable_path(path):
            return f"Reading executable file types is not allowed: {path.suffix}"

        content = await _read_text_with_fallback(path)
        safe_offset = max(0, offset)
        safe_max_chars = max(1, min(max_chars, MAX_TEXT_READ_CHARS))
        chunk = content[safe_offset:safe_offset + safe_max_chars]
        next_offset = safe_offset + len(chunk)
        has_more = next_offset < len(content)

        return (
            f"File: {path.name}\n"
            f"Offset: {safe_offset}\n"
            f"Returned chars: {len(chunk)}\n"
            f"Has more: {has_more}\n"
            f"Next offset: {next_offset if has_more else ''}\n"
            f"Content:\n{chunk}"
        )

    @tool("workspace_write_file")
    async def workspace_write_file(
            file_path: str,
            content: str,
            overwrite: bool = False,
    ) -> str:
        """
        Записывает текстовый файл внутри разрешённых директорий workspace.

        Запись executable-файлов запрещена. По умолчанию существующие файлы не
        перезаписываются; для обновления передайте overwrite=True.
        """
        path = _resolve_path(runtime, file_path)
        if _is_executable_path(path):
            return f"Writing executable file types is not allowed: {path.suffix}"
        if path.exists() and not overwrite:
            return f"File already exists: {file_path}. Use overwrite=True to replace it."

        path.parent.mkdir(parents=True, exist_ok=True)
        async with aiofiles.open(path, "w", encoding="utf-8") as fh:
            await fh.write(content)

        return f"File written: {path}"

    @tool("save_simple_variable_in_virtual_env_to_file")
    async def save_variable_to_file(variable_name: str, output_path: str) -> str:
        """
    	[Инструмент]: [сохранить] [объект].
[Тип]: [DataFrame], [.json], [.txt], [.md], [.log], [.pkl].
[Путь]: [директория] [рабочая] [локальная].
[Цель]: [доступ] [дальнейший], [модель], [пользователь].
[Граница]: [только] [данный] [набор].
[Прочее]: [иные] [форматы] — [иные] [инструменты].
save_simple_variable_in_virtual_env_to_file
        """
        value = await runtime.sandbox.get_variable(variable_name)
        if value is None:
            return f"Переменная '{variable_name}' не найдена."

        path = _resolve_path(runtime, output_path)
        path.parent.mkdir(parents=True, exist_ok=True)

        if isinstance(value, pd.DataFrame):
            _save_dataframe(value, path)
            return f"DataFrame '{variable_name}' сохранен в {path}"

        suffix = path.suffix.lower()
        answer  = f"Переменная  '{variable_name}' из виртруального окружения сохранена в  {path}"
        if suffix == ".json":
            async with aiofiles.open(path, "w", encoding="utf-8") as fh:
                await fh.write(
                    json.dumps(value, ensure_ascii=False, indent=2, default=str)
                )
        elif suffix in TEXT_EXTENSIONS:
            async with aiofiles.open(path, "w", encoding="utf-8") as fh:
                await fh.write(str(value))
        else:
            async with aiofiles.open(path, "wb") as fh:
                answer += " в формате pickle файла. ВАЖНО: если вам нужно было сохранить переменную в другом формате (не pickle) то попробуйте вызвать другой инструмент или сгенерировать код"
                await fh.write(pickle.dumps(value))

        return answer

    @tool("load_additional_context")
    async def load_additional_context(
            context_file: str,
            section: str = "full",
    ) -> str:
        """
         [Инструмент]: [загрузить] [данные].
[Что]: [бизнес-контекст], [источники].
[Где]: [директория] [пользователя].
[Когда]: [есть] [запрос] — [использовать].
load_additional_context
        """
        try:
            path = _resolve_context_file(runtime, context_file)
        except FileNotFoundError:
            available = _list_context_files(runtime.contexts_dir)
            if available:
                return (
                    f"Файл с названием: {context_file}. Не найден "
                    f"Доступные файлы с дополнительным контекстом: {', '.join(available)}"
 		    f"Проверьте название файла и попробуйте вызвать инструмент с другим названием файла"
                )
            return (
                f"Context file not found: {context_file}. "
                "No context files are available in contexts directory."
            )

        content = await _read_text_with_fallback(path)
        normalized = content.replace("\r\n", "\n")

        section_separator = "\n---\n"
        if section_separator in normalized:
            preview_part, full_part = normalized.split(section_separator, maxsplit=1)
        else:
            preview_part, full_part = normalized, normalized

        selected = full_part if section.lower() == "full" else preview_part
        return selected.strip()

    @tool("load_additional_source_database_table")
    async def load_additional_source_database_table(
            source_file: str,
            variable_name: str = DEFAULT_ADDITIONAL_DF_VAR,
            preview: bool = True,
            rows: int = 5,
    ) -> str:
        """
        Используйте этот инструмент только когда вам необходимо загрузить
        информацию из доступного дополнительного источника данных (базы данных, таблицы)

        Args:
            source_file:   Имя файла источника данных.
            variable_name: Имя переменной для сохранения в sandbox.
            preview:       При True — показывает первые строки без загрузки
                           в память. При False — загружает полный набор данных
                           в виртуальное окружение.
            rows:          Количество строк для предпросмотра.
        """
        path = _resolve_path(runtime, source_file, base_dir=runtime.sources_dir)
        if not path.exists() or not path.is_file():
            return f"Источник данных (таблица, база) не найдена: {source_file}. Проверьте название файла или выбранный инструмент, возможно, вам нужен инструмента для загрузки контекста а не для загрузки источника"

        df = _load_dataframe(path)

        if preview:
            rows = max(1, min(rows, MAX_PREVIEW_ROWS_SOURCE))
            columns = (
                ", ".join(df.columns.astype(str).tolist())
                if len(df.columns)
                else "<no columns>"
            )
            dtypes = ", ".join(
                f"{name}:{dtype}" for name, dtype in df.dtypes.items()
            )
            table_preview = df.head(rows).to_string(index=False)
            return (
		f"Превью файла: {path.name}\n"
                f"Размерность: {df.shape}\n"
                f"Колонки: {columns}\n"
                f"Типы данных в колонках: {dtypes}\n"
                f"Строки: {rows}\n"
                f"Превью:\n{table_preview}"
 		f"Для загрузки источника в переменную вирутального окружения вызовите этот инструмент (load_additional_source) с параметром preview=False "
            )

        await runtime.sandbox.add_variable(variable_name, df)
        return (
            f"Dataframe загружен полностью из '{source_file}' в переменную '{variable_name}'. Она находится в вирутальном окружении "
            f"Размерность набора данных в вирутальном окружении={df.shape}. Колонки={list(df.columns)}"
        )

    tools = [
        get_list_files_from_root_directory,
        replace_current_dataframe,
        list_loaded_dataframes_in_virtual_environment,
        show_current_dataframe,
        workspace_read_file,
        workspace_write_file,
        save_variable_to_file,
        load_additional_context,
        load_additional_source_database_table,
    ]

    selected_tool_names = enabled_tool_names or set(DEFAULT_WORKSPACE_TOOL_NAMES)
    return [workspace_tool for workspace_tool in tools if workspace_tool.name in selected_tool_names]
