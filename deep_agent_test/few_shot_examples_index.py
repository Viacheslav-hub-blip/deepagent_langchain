"""Индекс few-shot примеров для аналитического DeepAgent.

Содержит:
- FewShotExampleDocument: внутреннее описание примера в индексе.
- FewShotIndexManifest: файл состояния индекса с хэшами примеров.
- FewShotSearchResult: результат векторного поиска по примерам.
- FewShotExamplesStore: хранилище индекса и векторный поиск top-k.
- FewShotExamplesStore.__init__: загрузка документов и векторов индекса.
- FewShotExamplesStore.search: поиск top-k похожих примеров.
- update_few_shot_examples_index: инкрементальное обновление индекса по измененным файлам.
- parse_few_shot_example_file: чтение name и описания из markdown-файла.
- collect_example_file_hashes: сбор sha256-хэшей markdown-файлов.
- compute_file_sha256: расчет sha256-хэша файла.
- load_few_shot_documents: чтение документов индекса.
- load_index_manifest: чтение manifest индекса.
- save_index_manifest: сохранение manifest индекса.
- _split_example_header: выделение верхней части markdown-файла до разделителя.
- _load_previous_index: загрузка прежних документов и векторов по относительному пути.
- _load_vectors: загрузка numpy-массива векторов.
- _build_vectors_array: сборка списка векторов в двумерный массив.
- _cosine_scores: расчет cosine similarity для векторного поиска.
- _write_json_atomic: атомарная запись JSON-файла.
- _write_jsonl_atomic: атомарная запись JSONL-файла.
- _write_vectors_atomic: атомарная запись numpy-векторов.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
from pathlib import Path
from typing import Any

import numpy as np
from pydantic import BaseModel, Field

INDEX_VERSION = 1
EXAMPLE_FILE_PATTERN = "*.md"
HEADER_SEPARATOR_PATTERN = re.compile(r"(?m)^---\s*$")


class FewShotExampleDocument(BaseModel):
    """Внутренний документ few-shot примера, сохраненный в индексе.

    Args:
        name: Человекочитаемое название примера из первой строки markdown-файла.
        relative_path: Относительный путь к markdown-файлу внутри папки примеров.
        absolute_path: Абсолютный путь к markdown-файлу для загрузки полного содержимого.
        description: Описание кейса из верхней части файла до разделителя.
        index_text: Текст, по которому строится embedding для поиска.

    Returns:
        Документ индекса, который используется для поиска и последующей загрузки полного примера.
    """

    name: str
    relative_path: str
    absolute_path: str
    description: str
    index_text: str


class FewShotIndexManifest(BaseModel):
    """Состояние индекса few-shot примеров.

    Args:
        index_version: Версия формата индекса.
        file_hashes: Хэши markdown-файлов на момент последней индексации.

    Returns:
        Manifest, по которому определяется список новых, измененных и удаленных файлов.
    """

    index_version: int = INDEX_VERSION
    file_hashes: dict[str, str] = Field(default_factory=dict)


class FewShotSearchResult(BaseModel):
    """Результат векторного поиска few-shot примера.

    Args:
        name: Название найденного примера.
        relative_path: Относительный путь к markdown-файлу примера.
        absolute_path: Абсолютный путь к markdown-файлу примера.
        description: Описание кейса из верхней части markdown-файла.
        vector_score: Оценка cosine similarity с пользовательским запросом.

    Returns:
        Кандидат, который можно передать LLM-реранжеру.
    """

    name: str
    relative_path: str
    absolute_path: str
    description: str
    vector_score: float


class FewShotExamplesStore:
    """Хранилище few-shot индекса с поиском top-k похожих примеров.

    Args:
        index_dir: Папка с ``documents.jsonl`` и ``vectors.npy``.
        embeddings: Embeddings-модель LangChain с методами ``embed_query`` и ``embed_documents``.

    Returns:
        Объект, который выполняет векторный поиск по готовому индексу.
    """

    def __init__(self, index_dir: Path, embeddings: Any) -> None:
        """Загружает индекс few-shot примеров с диска.

        Args:
            index_dir: Папка с файлами индекса.
            embeddings: Embeddings-модель для векторизации пользовательского запроса.

        Returns:
            None.
        """

        self.index_dir = index_dir
        self.embeddings = embeddings
        self.documents = load_few_shot_documents(index_dir / "documents.jsonl")
        self.vectors = _load_vectors(index_dir / "vectors.npy")

    def search(self, query: str, top_k: int = 10) -> list[FewShotSearchResult]:
        """Возвращает top-k похожих few-shot примеров.

        Args:
            query: Пользовательский запрос для поиска похожих примеров.
            top_k: Максимальное количество кандидатов.

        Returns:
            Список кандидатов, отсортированный по убыванию cosine similarity.
        """

        if not query.strip() or not self.documents or self.vectors.size == 0:
            return []

        query_vector = np.asarray(self.embeddings.embed_query(query), dtype=float)
        scores = _cosine_scores(query_vector=query_vector, vectors=self.vectors)
        top_indexes = np.argsort(scores)[::-1][:top_k]

        results: list[FewShotSearchResult] = []
        for index in top_indexes:
            document = self.documents[int(index)]
            results.append(
                FewShotSearchResult(
                    name=document.name,
                    relative_path=document.relative_path,
                    absolute_path=document.absolute_path,
                    description=document.description,
                    vector_score=float(scores[int(index)]),
                )
            )
        return results


def update_few_shot_examples_index(
    examples_dir: Path,
    index_dir: Path,
    embeddings: Any,
) -> None:
    """Инкрементально обновляет индекс few-shot примеров по измененным файлам.

    Args:
        examples_dir: Папка с markdown-файлами примеров.
        index_dir: Папка для хранения ``manifest.json``, ``documents.jsonl`` и ``vectors.npy``.
        embeddings: Embeddings-модель LangChain для построения векторов новых и измененных файлов.

    Returns:
        None. Функция сохраняет обновленный индекс на диск.
    """

    examples_dir.mkdir(parents=True, exist_ok=True)
    index_dir.mkdir(parents=True, exist_ok=True)

    manifest_path = index_dir / "manifest.json"
    documents_path = index_dir / "documents.jsonl"
    vectors_path = index_dir / "vectors.npy"

    current_hashes = collect_example_file_hashes(examples_dir)
    old_manifest = load_index_manifest(manifest_path)
    old_documents, old_vectors = _load_previous_index(documents_path, vectors_path)

    index_is_usable = (
        old_manifest is not None
        and old_manifest.index_version == INDEX_VERSION
        and documents_path.exists()
        and vectors_path.exists()
    )

    if index_is_usable and old_manifest.file_hashes == current_hashes:
        return

    documents_by_path = {document.relative_path: document for document in old_documents}
    vectors_by_path = {
        document.relative_path: old_vectors[index]
        for index, document in enumerate(old_documents)
        if index < len(old_vectors)
    }

    ordered_paths = sorted(current_hashes)
    new_documents: list[FewShotExampleDocument] = []
    new_vectors_by_path: dict[str, np.ndarray] = {}
    paths_to_embed: list[str] = []

    for relative_path in ordered_paths:
        file_hash = current_hashes[relative_path]
        old_hash = old_manifest.file_hashes.get(relative_path) if old_manifest else None
        can_reuse = (
            index_is_usable
            and old_hash == file_hash
            and relative_path in documents_by_path
            and relative_path in vectors_by_path
        )

        if can_reuse:
            document = documents_by_path[relative_path]
            vector = np.asarray(vectors_by_path[relative_path], dtype=float)
            new_documents.append(document)
            new_vectors_by_path[relative_path] = vector
            continue

        document = parse_few_shot_example_file(examples_dir / relative_path, examples_dir)
        new_documents.append(document)
        paths_to_embed.append(relative_path)

    if paths_to_embed:
        texts_to_embed = [
            document.index_text
            for document in new_documents
            if document.relative_path in set(paths_to_embed)
        ]
        embedded_vectors = embeddings.embed_documents(texts_to_embed)
        for relative_path, vector in zip(paths_to_embed, embedded_vectors, strict=True):
            new_vectors_by_path[relative_path] = np.asarray(vector, dtype=float)

    ordered_vectors = [
        new_vectors_by_path[document.relative_path]
        for document in new_documents
        if document.relative_path in new_vectors_by_path
    ]
    vectors_array = _build_vectors_array(ordered_vectors)

    _write_jsonl_atomic(documents_path, [document.model_dump(mode="json") for document in new_documents])
    _write_vectors_atomic(vectors_path, vectors_array)
    save_index_manifest(
        manifest_path,
        FewShotIndexManifest(index_version=INDEX_VERSION, file_hashes=current_hashes),
    )


def parse_few_shot_example_file(path: Path, examples_dir: Path) -> FewShotExampleDocument:
    """Читает markdown-файл few-shot примера.

    Args:
        path: Путь к markdown-файлу примера.
        examples_dir: Корневая папка примеров для вычисления относительного пути.

    Returns:
        Документ индекса с названием, описанием, путем и текстом для embedding.
    """

    full_text = path.read_text(encoding="utf-8")
    header = _split_example_header(full_text).strip()
    lines = [line.strip() for line in header.splitlines() if line.strip()]

    if not lines or not lines[0].startswith("name:"):
        raise ValueError(f"Файл {path} должен начинаться со строки 'name: ...'.")

    name = lines[0].removeprefix("name:").strip()
    description = "\n".join(lines[1:]).strip()

    if not name:
        raise ValueError(f"В файле {path} не заполнено название после 'name:'.")
    if not description:
        raise ValueError(f"В файле {path} не заполнено описание примера до разделителя '---'.")

    relative_path = path.relative_to(examples_dir).as_posix()
    index_text = f"{name}\n\n{description}"
    return FewShotExampleDocument(
        name=name,
        relative_path=relative_path,
        absolute_path=str(path.resolve()),
        description=description,
        index_text=index_text,
    )


def collect_example_file_hashes(examples_dir: Path) -> dict[str, str]:
    """Собирает sha256-хэши markdown-файлов примеров.

    Args:
        examples_dir: Папка с markdown-файлами few-shot примеров.

    Returns:
        Словарь ``относительный путь -> sha256`` для текущего состояния папки.
    """

    hashes: dict[str, str] = {}
    for path in sorted(examples_dir.rglob(EXAMPLE_FILE_PATTERN)):
        if not path.is_file():
            continue
        relative_path = path.relative_to(examples_dir).as_posix()
        hashes[relative_path] = compute_file_sha256(path)
    return hashes


def compute_file_sha256(path: Path) -> str:
    """Вычисляет sha256-хэш файла.

    Args:
        path: Путь к файлу, хэш которого нужно вычислить.

    Returns:
        Строка sha256-хэша содержимого файла.
    """

    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_few_shot_documents(path: Path) -> list[FewShotExampleDocument]:
    """Загружает документы few-shot индекса из JSONL-файла.

    Args:
        path: Путь к ``documents.jsonl``.

    Returns:
        Список документов индекса или пустой список, если файла нет.
    """

    if not path.exists():
        return []

    documents: list[FewShotExampleDocument] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        documents.append(FewShotExampleDocument.model_validate_json(line))
    return documents


def load_index_manifest(path: Path) -> FewShotIndexManifest | None:
    """Загружает manifest few-shot индекса.

    Args:
        path: Путь к ``manifest.json``.

    Returns:
        Manifest индекса или ``None``, если manifest отсутствует.
    """

    if not path.exists():
        return None
    return FewShotIndexManifest.model_validate_json(path.read_text(encoding="utf-8"))


def save_index_manifest(path: Path, manifest: FewShotIndexManifest) -> None:
    """Сохраняет manifest few-shot индекса.

    Args:
        path: Путь к ``manifest.json``.
        manifest: Manifest с текущими хэшами файлов.

    Returns:
        None.
    """

    _write_json_atomic(path, manifest.model_dump(mode="json"))


def _split_example_header(text: str) -> str:
    """Возвращает верхнюю часть markdown-примера до первого разделителя.

    Args:
        text: Полное содержимое markdown-файла.

    Returns:
        Текст до первой строки ``---`` или весь текст, если разделителя нет.
    """

    match = HEADER_SEPARATOR_PATTERN.search(text)
    if match is None:
        return text
    return text[: match.start()]


def _load_previous_index(
    documents_path: Path,
    vectors_path: Path,
) -> tuple[list[FewShotExampleDocument], np.ndarray]:
    """Загружает прежние документы и векторы индекса.

    Args:
        documents_path: Путь к ``documents.jsonl``.
        vectors_path: Путь к ``vectors.npy``.

    Returns:
        Кортеж из списка документов и массива векторов. При ошибке возвращает пустой индекс.
    """

    if not documents_path.exists() or not vectors_path.exists():
        return [], np.empty((0, 0), dtype=float)

    try:
        documents = load_few_shot_documents(documents_path)
        vectors = _load_vectors(vectors_path)
    except (OSError, ValueError, json.JSONDecodeError):
        return [], np.empty((0, 0), dtype=float)

    if len(documents) != len(vectors):
        return [], np.empty((0, 0), dtype=float)
    return documents, vectors


def _load_vectors(path: Path) -> np.ndarray:
    """Загружает numpy-массив векторов индекса.

    Args:
        path: Путь к ``vectors.npy``.

    Returns:
        Двумерный numpy-массив векторов или пустой массив.
    """

    if not path.exists():
        return np.empty((0, 0), dtype=float)
    vectors = np.load(path)
    if vectors.ndim == 1 and vectors.size == 0:
        return np.empty((0, 0), dtype=float)
    if vectors.ndim != 2:
        raise ValueError(f"Файл {path} должен содержать двумерный массив vectors.")
    return np.asarray(vectors, dtype=float)


def _build_vectors_array(vectors: list[np.ndarray]) -> np.ndarray:
    """Собирает список векторов в двумерный numpy-массив.

    Args:
        vectors: Список одномерных embedding-векторов.

    Returns:
        Двумерный массив ``N x D`` или пустой массив ``0 x 0``.
    """

    if not vectors:
        return np.empty((0, 0), dtype=float)
    return np.vstack([np.asarray(vector, dtype=float) for vector in vectors])


def _cosine_scores(query_vector: np.ndarray, vectors: np.ndarray) -> np.ndarray:
    """Считает cosine similarity между запросом и матрицей векторов.

    Args:
        query_vector: Вектор пользовательского запроса.
        vectors: Матрица векторов документов.

    Returns:
        Массив оценок cosine similarity для каждого документа.
    """

    if vectors.size == 0:
        return np.asarray([], dtype=float)

    query_norm = np.linalg.norm(query_vector)
    vector_norms = np.linalg.norm(vectors, axis=1)
    denominator = vector_norms * query_norm
    denominator = np.where(denominator == 0, 1.0, denominator)
    return vectors @ query_vector / denominator


def _write_json_atomic(path: Path, payload: Any) -> None:
    """Атомарно записывает JSON-файл.

    Args:
        path: Целевой путь JSON-файла.
        payload: JSON-сериализуемое содержимое.

    Returns:
        None.
    """

    path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = path.with_suffix(path.suffix + ".tmp")
    temp_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    os.replace(temp_path, path)


def _write_jsonl_atomic(path: Path, rows: list[dict[str, Any]]) -> None:
    """Атомарно записывает JSONL-файл.

    Args:
        path: Целевой путь JSONL-файла.
        rows: Список JSON-сериализуемых строк.

    Returns:
        None.
    """

    path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = path.with_suffix(path.suffix + ".tmp")
    content = "\n".join(json.dumps(row, ensure_ascii=False) for row in rows)
    if content:
        content += "\n"
    temp_path.write_text(content, encoding="utf-8")
    os.replace(temp_path, path)


def _write_vectors_atomic(path: Path, vectors: np.ndarray) -> None:
    """Атомарно записывает numpy-массив векторов.

    Args:
        path: Целевой путь ``vectors.npy``.
        vectors: Двумерный массив embedding-векторов.

    Returns:
        None.
    """

    path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = path.with_suffix(path.suffix + ".tmp")
    with temp_path.open("wb") as file:
        np.save(file, vectors)
    os.replace(temp_path, path)


__all__ = [
    "FewShotExampleDocument",
    "FewShotExamplesStore",
    "FewShotIndexManifest",
    "FewShotSearchResult",
    "collect_example_file_hashes",
    "compute_file_sha256",
    "load_few_shot_documents",
    "load_index_manifest",
    "parse_few_shot_example_file",
    "save_index_manifest",
    "update_few_shot_examples_index",
]
