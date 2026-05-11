"""
base_code_executor_tool.py

Базовый класс инструмента-исполнителя кода для LangChain-агента.

Описание:
    Содержит класс BaseCodeExecutorTool, который:
    - принимает MCP-инструмент и Python-песочницу;
    - преобразует JSON Schema MCP-инструмента в Pydantic-модель;
    - генерирует код через MCP, логирует его, выполняет в песочнице и возвращает JSON-ответ;
    - поддерживает контекст повторных попыток (предыдущий код и ошибка).
"""

import json
import re
import asyncio
from enum import Enum
from typing import Any, Dict, Optional, Type

import nest_asyncio
from pydantic import BaseModel, ConfigDict, Field, PrivateAttr, create_model
from langchain_core.tools import BaseTool

from .sandbox import ClientPythonSandbox, ExecutionResult

# Максимальная длина строкового превью переменной
_PREVIEW_MAX_LEN: int = 500

# Суффикс, добавляемый к усечённому превью
_PREVIEW_SUFFIX: str = "..."

# Сообщение, когда доступны только встроенные библиотеки Python
_NO_LIBRARIES_MSG: str = "Доступны только встроенные библиотеки Python"

# Сообщение, когда в песочнице нет переменных
_NO_VARIABLES_MSG: str = "Нет переменных"

# Ключи, по которым ищется код в словарном ответе MCP
_RESPONSE_CODE_KEYS: tuple[str, ...] = ("code", "text", "content", "result", "output")

# Количество строк датафрейма для превью в схеме
_DATAFRAME_HEAD_ROWS: int = 2

# Поля MCP-инструментов, в которых обычно передается текст задания на генерацию кода
_CODE_TASK_TEXT_KEYS: tuple[str, ...] = ("task", "instruction")

# Контракт, который добавляется к задаче перед вызовом внешнего генератора кода
_GROUNDED_CODE_CONTRACT: str = """

Контракт работы с данными:
- Сгенерированный код должен выполнять вычисления только по данным, которые уже доступны в schema_context, явно переданы в тексте задачи, доступны как переменные песочницы или указаны как artifact/file path.
- Не создавай демонстрационные, синтетические или примерные входные таблицы вместо реальных данных. Допустимы только небольшие служебные константы: пороги, названия колонок, пустые структуры результата и параметры фильтрации.
- Если нужных данных нет в schema_context, тексте задачи, переменных или artifact/file path, код должен записать в target_variable диагностический результат с описанием недостающих входов, а не имитировать данные.
- Итоговая переменная должна содержать не только числа, но и краткое описание использованных входов: имена переменных, artifact/file path или другой источник, который реально использовался кодом.
"""

# Заголовок диагностического блока с кодом, отправленным в песочницу.
_SANDBOX_CODE_LOG_TITLE: str = "[sandbox-code]"

# Заголовок диагностического блока с ошибкой выполнения кода.
_SANDBOX_ERROR_LOG_TITLE: str = "[sandbox-error]"


class BaseCodeExecutorTool(BaseTool):
    """
    Базовый LangChain-инструмент, выполняющий код в Python-песочнице.

    Принимает MCP-инструмент, который генерирует код, запускает его
    в изолированной песочнице и возвращает структурированный JSON-ответ.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)
    args_schema: Optional[Type[BaseModel]] = None

    mcp_tool: Any = Field(default=None, exclude=True, repr=False)
    sandbox: ClientPythonSandbox = Field(default=None, exclude=True, repr=False)
    used_libraries: Optional[str] = Field(default=None, exclude=True, repr=False)

    # Приватные атрибуты для контекста повторных попыток
    _previous_code: Optional[str] = PrivateAttr(default=None)
    _error_context: Optional[str] = PrivateAttr(default=None)

    def __init__(
            self,
            mcp_tool: Any,
            sandbox: ClientPythonSandbox,
            used_libraries: Optional[str] = None,
            **kwargs: Any,
    ) -> None:
        """
        Инициализация инструмента.

        Преобразует схему MCP-инструмента в Pydantic-модель и передаёт
        все необходимые зависимости родительскому классу.

        Args:
            mcp_tool: MCP-инструмент, генерирующий Python-код.
            sandbox: Клиент Python-песочницы для выполнения кода.
            used_libraries: Строка с перечнем используемых библиотек (опционально).
            **kwargs: Дополнительные аргументы для BaseTool.
        """
        mcp_schema = getattr(mcp_tool, "args_schema", None)
        kwargs["args_schema"] = self._convert_schema(mcp_schema) if mcp_schema else None
        kwargs["mcp_tool"] = mcp_tool
        kwargs["sandbox"] = sandbox
        kwargs["used_libraries"] = used_libraries
        super().__init__(**kwargs)

    @staticmethod
    def _convert_schema(schema: Any) -> Optional[Type[BaseModel]]:
        """
        Преобразует представление схемы MCP в класс Pydantic-модели.

        Args:
            schema: Схема в виде класса, словаря JSON Schema или None.

        Returns:
            Класс Pydantic-модели или None, если схема не распознана.
        """
        if isinstance(schema, type):
            return schema

        if isinstance(schema, dict):
            return BaseCodeExecutorTool._json_schema_to_pydantic(schema)

        return None

    @staticmethod
    def _json_schema_to_pydantic(json_schema: dict) -> Type[BaseModel]:
        """
        Строит динамическую Pydantic-модель из свойств JSON Schema.

        Args:
            json_schema: Словарь с ключами "properties" и "required".

        Returns:
            Динамически созданный класс Pydantic-модели.
        """
        properties = json_schema.get("properties", {})
        required = json_schema.get("required", [])

        # Соответствие типов JSON Schema и Python
        TYPE_MAP: Dict[str, type] = {
            "string": str,
            "integer": int,
            "number": float,
            "boolean": bool,
            "array": list,
            "object": dict,
        }

        fields: Dict[str, Any] = {}
        for name, info in properties.items():
            has_default = "default" in info
            is_optional = has_default or name not in required

            # Поля с anyOf всегда опциональны
            if "anyOf" in info:
                is_optional = True

            json_type = info.get("type", "string")
            python_type = TYPE_MAP.get(json_type, str)

            # Если поле содержит enum — создаём динамический Enum-класс
            enum_values = info.get("enum")
            description = info.get("description", "")
            if enum_values:
                enum_name = f"{name}_enum"
                enum_cls = Enum(enum_name, {v: v for v in enum_values})
                python_type = enum_cls
                description += f" Допустимые значения: {enum_values}"

            # Определяем тип и значение по умолчанию для поля
            if is_optional:
                field_type = Optional[python_type]
                default = info.get("default", None)
            else:
                field_type = python_type
                default = ...

            fields[name] = (
                field_type,
                Field(default=default, description=description),
            )

        return create_model("DynamicMCPSchema", **fields)

    def _run(self, **kwargs: Any) -> str:
        """
        Синхронная обёртка вокруг асинхронного метода _arun.

        Требуется интерфейсом BaseTool. Если цикл событий уже запущен,
        применяет nest_asyncio для вложенного выполнения корутины.

        Args:
            **kwargs: Аргументы, передаваемые в _arun.

        Returns:
            JSON-строка с результатом выполнения инструмента.
        """
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            loop = None

        if loop and loop.is_running():
            # Применяем nest_asyncio, чтобы вложить вызов в уже запущенный цикл
            nest_asyncio.apply()
            return loop.run_until_complete(self._arun(**kwargs))
        else:
            return asyncio.run(self._arun(**kwargs))

    async def _arun(self, **kwargs: Any) -> str:
        """
        Асинхронная точка входа: генерирует код через MCP,
        выполняет его в песочнице и возвращает JSON-ответ.

        Args:
            **kwargs: Аргументы инструмента, включая target_variable.

        Returns:
            JSON-строка с результатом выполнения или описанием ошибки.
        """
        mcp_args = self._prepare_mcp_args(**kwargs)
        generated_code = await self._invoke_mcp_and_parse(mcp_args)
        if generated_code is None:
            return self._make_error_response("Не удалось получить код от MCP")

        target_variable: str = kwargs.get("target_variable", "result")
        self._log_generated_code(
            code=generated_code,
            target_variable=target_variable,
        )
        result = await self.sandbox.execute(generated_code, target_variable=target_variable)
        self._log_execution_error(result)

        # Сохраняем имя последней успешно заполненной переменной в песочнице
        if result.success:
            self.sandbox.last_target_variable = target_variable

        # Если переменная является датафреймом — запоминаем её имя отдельно
        val = await self.sandbox.get_variable(target_variable)
        if val is not None and hasattr(val, "shape") and hasattr(val, "head"):
            self.sandbox.last_dataframe_variable = target_variable

        tool_response = self._format_response(result, generated_code, target_variable, **kwargs)
        return tool_response

    def _prepare_mcp_args(self, **kwargs: Any) -> Dict[str, Any]:
        """
        Подготавливает и фильтрует аргументы для вызова MCP-инструмента.

        Добавляет контекст схемы, доступных библиотек, предыдущего кода
        и ошибки. Лишние ключи, отсутствующие в схеме MCP, удаляются.

        Args:
            **kwargs: Исходные аргументы вызова инструмента.

        Returns:
            Отфильтрованный словарь аргументов для MCP.
        """
        # Исключаем None-значения из аргументов
        mcp_args = {k: v for k, v in kwargs.items() if v is not None}

        mcp_args["schema_context"] = self._get_current_schema()

        # Передаём контекст доступных библиотек
        mcp_args["used_libraries"] = self._get_used_library_context()

        self._apply_grounded_code_contract(mcp_args)

        # Добавляем предыдущий код и ошибку, если они есть и не переданы явно
        if "previous_code" not in mcp_args and self._previous_code:
            mcp_args["previous_code"] = self._previous_code
        if "error_context" not in mcp_args and self._error_context:
            mcp_args["error_context"] = self._error_context

        # Фильтруем аргументы по допустимым ключам схемы MCP
        allowed_keys = self._get_allowed_mcp_keys()
        if allowed_keys is not None:
            filtered = {k: v for k, v in mcp_args.items() if k in allowed_keys}
            removed = set(mcp_args.keys()) - set(filtered.keys())
            if removed:
                print(f"[MCP-FILTER] Удалены параметры, отсутствующие в схеме: {removed}")
            mcp_args = filtered

        return mcp_args

    @staticmethod
    def _apply_grounded_code_contract(mcp_args: Dict[str, Any]) -> None:
        """
        Добавляет к тексту задания универсальный контракт доказательной генерации кода.

        Args:
            mcp_args: Аргументы, подготовленные для внешнего MCP-генератора кода.

        Returns:
            ``None``. Словарь ``mcp_args`` изменяется на месте.
        """
        for key in _CODE_TASK_TEXT_KEYS:
            value = mcp_args.get(key)
            if isinstance(value, str) and value.strip():
                if "Контракт работы с данными:" not in value:
                    mcp_args[key] = f"{value.rstrip()}{_GROUNDED_CODE_CONTRACT}"
                return

    def _get_allowed_mcp_keys(self) -> Optional[set]:
        """
        Извлекает допустимые имена аргументов MCP из доступных схем.

        Проверяет args_schema (Pydantic v1/v2), input_schema и schema()
        в порядке приоритета.

        Returns:
            Множество допустимых ключей или None, если схема не найдена.
        """
        schema = getattr(self.mcp_tool, "args_schema", None)
        if schema is not None:
            if isinstance(schema, type) and hasattr(schema, "model_fields"):
                # Pydantic v2
                return set(schema.model_fields.keys())
            if isinstance(schema, type) and hasattr(schema, "__fields__"):
                # Pydantic v1
                return set(schema.__fields__.keys())
            if isinstance(schema, dict) and "properties" in schema:
                return set(schema["properties"].keys())

        # Проверяем input_schema (альтернативный формат MCP)
        input_schema = getattr(self.mcp_tool, "input_schema", None)
        if isinstance(input_schema, dict) and "properties" in input_schema:
            return set(input_schema["properties"].keys())

        # Пробуем вызвать метод schema() как запасной вариант
        if hasattr(self.mcp_tool, "schema"):
            try:
                s = self.mcp_tool.schema()
                if isinstance(s, dict) and "properties" in s:
                    return set(s["properties"].keys())
            except Exception:
                pass

        # Схема не найдена — фильтрация не применяется
        return None

    def _get_used_library_context(self) -> str:
        """
        Возвращает строку с перечнем библиотек, доступных в песочнице.

        Returns:
            Строка с именами разрешённых библиотек через запятую,
            или сообщение о том, что доступны только встроенные библиотеки.
        """
        if not self.sandbox.allowed_libraries:
            return _NO_LIBRARIES_MSG

        allowed = self.sandbox.allowed_libraries
        return ", ".join(allowed)

    async def _invoke_mcp_and_parse(self, mcp_args: Dict[str, Any]) -> Optional[str]:
        """
        Вызывает MCP-инструмент и извлекает код из ответа.

        Args:
            mcp_args: Подготовленные аргументы для MCP.

        Returns:
            Строка с Python-кодом или None в случае ошибки.
        """
        try:
            response = await self.mcp_tool.ainvoke(mcp_args)
            return self._extract_code_from_response(response)
        except Exception:
            return None

    def _extract_code_from_response(self, raw_response: Any) -> str:
        """
        Извлекает исполняемый Python-код из различных форматов ответа MCP.

        Поддерживает строки, объекты с атрибутом content, словари и списки.

        Args:
            raw_response: Сырой ответ от MCP-инструмента.

        Returns:
            Очищенная строка Python-кода.
        """
        if isinstance(raw_response, str):
            return self._clean_code(raw_response)

        if hasattr(raw_response, "content"):
            content = raw_response.content
            if isinstance(content, str):
                return self._clean_code(content)
            if isinstance(content, list) and len(content) > 0:
                for item in content:
                    if isinstance(item, dict) and "text" in item:
                        return self._clean_code(item["text"])
                    if isinstance(item, str):
                        return self._clean_code(item)

        if isinstance(raw_response, dict):
            # Перебираем стандартные ключи в порядке приоритета
            for key in _RESPONSE_CODE_KEYS:
                if key in raw_response and isinstance(raw_response[key], str):
                    return self._clean_code(raw_response[key])
            if "response" in raw_response:
                return self._extract_code_from_response(raw_response["response"])

        if isinstance(raw_response, list) and len(raw_response) > 0:
            return self._extract_code_from_response(raw_response[0])

        return self._clean_code(str(raw_response))

    @staticmethod
    def _clean_code(code: str) -> str:
        """
        Удаляет markdown-обёртки и возвращает чистый Python-код.

        Args:
            code: Строка кода, возможно обёрнутая в markdown-блок.

        Returns:
            Очищенная строка кода без markdown-разметки.
        """
        code = re.sub(r'```\w*\s*\n?', '', code)
        code = re.sub(r'```\n?', '', code)
        code = re.sub(r'^\s*python\s*\n?', '', code, flags=re.IGNORECASE)
        return code.strip()

    def _format_response(
            self,
            result: ExecutionResult,
            code: str,
            target_var: str,
            **kwargs: Any,
    ) -> str:
        """
        Сериализует результат выполнения в JSON-ответ инструмента.

        Args:
            result: Объект результата выполнения из песочницы.
            code: Сгенерированный и выполненный Python-код.
            target_var: Имя целевой переменной в пространстве имён песочницы.
            **kwargs: Дополнительные аргументы (не используются напрямую).

        Returns:
            JSON-строка с полным описанием результата выполнения.
        """
        response: Dict[str, Any] = {
            "success": result.success,
            "tool_name": self.name,
            "generated_code": code,
            "target_variable": target_var,
            "execution_time_ms": result.execution_time_ms,
            "new_variables": result.new_variable_schemas,
            "execution_output": result.output,
        }

        if result.success:
            response["variable_preview"] = self.sandbox._get_variable_preview(target_var)
            response["message"] = f"Переменная '{target_var}' успешно создана"
        else:
            response["error"] = result.error
            response["message"] = "Выполнение кода завершилось с ошибкой"

        # Сохраняем код и ошибку для возможного повторного вызова
        self._previous_code = code
        self._error_context = result.error

        return json.dumps(response, ensure_ascii=False)

    def _log_generated_code(self, *, code: str, target_variable: str) -> None:
        """
        Выводит в stdout сгенерированный код перед выполнением в песочнице.

        Args:
            code: Python-код, полученный от MCP-инструмента.
            target_variable: Имя целевой переменной, которую должен создать код.

        Returns:
            ``None``. Метод выполняет только диагностический вывод.
        """

        print(
            f"{_SANDBOX_CODE_LOG_TITLE} tool={self.name} target_variable={target_variable}\n"
            f"{code}",
            flush=True,
        )

    def _log_execution_error(self, result: ExecutionResult) -> None:
        """
        Выводит в stdout ошибку выполнения кода, если sandbox вернул неуспешный результат.

        Args:
            result: Результат выполнения кода в песочнице.

        Returns:
            ``None``. Метод ничего не печатает при успешном выполнении.
        """

        if result.success:
            return
        error_text = result.error or "Sandbox execution failed without error text."
        print(
            f"{_SANDBOX_ERROR_LOG_TITLE} tool={self.name}\n{error_text}",
            flush=True,
        )

    @staticmethod
    def _make_error_response(message: str) -> str:
        """
        Возвращает нормализованный JSON-ответ об ошибке.

        Args:
            message: Текст сообщения об ошибке.

        Returns:
            JSON-строка с полями success, error и message.
        """
        return json.dumps(
            {"success": False, "error": message, "message": message},
            ensure_ascii=False,
        )

    def _get_current_schema(self) -> str:
        """
        Формирует текстовый контекст схемы для MCP-промпта
        на основе переменных в пространстве имён песочницы.

        Для датафреймов выводит форму, типы столбцов, образец данных,
        статистику пропущенных значений и пустых строк.
        Для остальных переменных — тип и усечённое строковое представление.

        Returns:
            Многострочная строка с описанием переменных
            или сообщение об отсутствии переменных.
        """
        lines = []

        for name, value in self.sandbox.globals.items():
            # Пропускаем приватные имена и модули
            if name.startswith("_"):
                continue
            if type(value).__name__ == "module":
                continue

            try:
                if hasattr(value, "shape") and hasattr(value, "columns"):
                    # Формируем описание столбцов с типами
                    cols_info = ", ".join(
                        f"{col} ({dtype})" for col, dtype in value.dtypes.items()
                    )

                    # Анализируем пропущенные значения (NaN)
                    nan_counts = value.isna().sum()
                    total_rows = len(value)
                    nan_cols = nan_counts[nan_counts > 0]

                    if not nan_cols.empty:
                        nan_lines = []
                        for col, count in nan_cols.items():
                            pct = count / total_rows * 100
                            nan_lines.append(f" - {col}: {count} NaN ({pct:.1f}%)")
                        nan_info = (
                                "\nСтолбцы с пропущенными значениями (NaN):\n"
                                + "\n".join(nan_lines)
                        )
                    else:
                        nan_info = "\nПропущенных значений (NaN) нет."

                    # Анализируем пустые строки в строковых столбцах
                    str_cols = value.select_dtypes(include=["object", "string"]).columns
                    empty_str_parts = []
                    for col in str_cols:
                        empty_count = (value[col] == "").sum()
                        if empty_count > 0:
                            pct = empty_count / total_rows * 100
                            empty_str_parts.append(
                                f" - {col}: {empty_count} пустых строк ({pct:.1f}%)"
                            )

                    if empty_str_parts:
                        empty_str_info = (
                                "\nСтолбцы с пустыми строками (''):\n"
                                + "\n".join(empty_str_parts)
                        )
                    else:
                        empty_str_info = ""

                    lines.append(
                        f"{name}: pandas DataFrame, shape={value.shape}\n"
                        f" Столбцы: {cols_info}\n"
                        f" Образец данных:\n{value.head(_DATAFRAME_HEAD_ROWS).to_string()}"
                        f"{nan_info}"
                        f"{empty_str_info}"
                    )

                elif name not in self.sandbox._base_globals_names:
                    # Ограничиваем длину превью для нечисловых переменных
                    preview = str(value)
                    if len(preview) > _PREVIEW_MAX_LEN:
                        preview = preview[:_PREVIEW_MAX_LEN] + _PREVIEW_SUFFIX
                    lines.append(f"{name}: {type(value).__name__} = {preview}")

            except Exception as e:
                lines.append(f"{name}: ошибка ({e})")

        result = "\n".join(lines)
        return result or _NO_VARIABLES_MSG

    def with_context(
            self,
            previous_code: Optional[str] = None,
            error_context: Optional[str] = None,
    ) -> "BaseCodeExecutorTool":
        """
        Клонирует инструмент и прикрепляет контекст повторной попытки.

        Args:
            previous_code: Код предыдущего вызова (для контекста повтора).
            error_context: Текст ошибки предыдущего вызова.

        Returns:
            Новый экземпляр инструмента с заданным контекстом.
        """
        new_tool = self.__class__(
            mcp_tool=self.mcp_tool,
            sandbox=self.sandbox,
            name=self.name,
            description=self.description,
        )
        object.__setattr__(new_tool, "_previous_code", previous_code)
        object.__setattr__(new_tool, "_error_context", error_context)
        return new_tool

    def reset_context(self) -> None:
        """Сбрасывает кэшированный контекст повторной попытки (код и ошибку)."""
        self._previous_code = None
        self._error_context = None
