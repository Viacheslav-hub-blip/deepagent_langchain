"""Файловое хранилище artifacts исследовательских запусков.

Содержит:
- ArtifactService: сервис записи, регистрации и чтения индекса artifacts.
"""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

from planner_agent.schemas.artifacts import Artifact, normalize_artifact_metadata

from ._json import append_jsonl, read_jsonl


class ArtifactService:
    """Управляет файловым хранилищем artifacts внутри каталога runs.

    Args:
        runs_dir: Корневой каталог, где хранятся ResearchRun и их artifacts.

    Returns:
        Экземпляр сервиса для записи и чтения artifacts.
    """

    def __init__(self, runs_dir: str | Path = "runs") -> None:
        """Сохраняет путь к каталогу запусков.

        Args:
            runs_dir: Корневой каталог runs.

        Returns:
            None.
        """

        self.runs_dir = Path(runs_dir)

    def write_artifact(
        self,
        *,
        run_id: str,
        node_id: str,
        kind: str,
        filename: str,
        content: str | bytes,
        mime_type: str = "text/plain",
        summary: str = "",
        metadata: dict[str, Any] | None = None,
        artifact_id: str | None = None,
    ) -> Artifact:
        """Создает новый файл artifact внутри каталога запуска и регистрирует его в индексе.

        Args:
            run_id: Идентификатор ResearchRun.
            node_id: Идентификатор lineage node, который создает artifact.
            kind: Универсальный тип artifact.
            filename: Относительный путь файла внутри каталога artifacts запуска.
            content: Текстовое или бинарное содержимое artifact.
            mime_type: MIME-тип содержимого.
            summary: Краткое описание artifact.
            metadata: Дополнительные metadata artifact.
            artifact_id: Опциональный человеко-читаемый идентификатор. Если задан и
                уже занят в текущем запуске, к нему добавляется короткий
                числовой суффикс для уникальности.

        Returns:
            Созданная запись Artifact.

        Raises:
            ValueError: Если filename пытается выйти за пределы каталога artifacts запуска.
        """

        artifacts_dir = self.runs_dir / run_id / "artifacts"
        target = (artifacts_dir / filename).resolve()
        if not self._is_subpath(target, artifacts_dir.resolve()):
            raise ValueError(f"Artifact path escapes run artifact directory: {filename}")

        target.parent.mkdir(parents=True, exist_ok=True)
        if isinstance(content, bytes):
            target.write_bytes(content)
            payload = content
        else:
            target.write_text(content, encoding="utf-8")
            payload = content.encode("utf-8")

        resolved_id = self._resolve_unique_artifact_id(run_id, artifact_id)
        artifact_kwargs: dict[str, Any] = {
            "run_id": run_id,
            "node_id": node_id,
            "kind": kind,
            "uri": str(target),
            "mime_type": mime_type,
            "summary": summary,
            "checksum": hashlib.sha256(payload).hexdigest(),
            "metadata": normalize_artifact_metadata(metadata, kind=kind),
        }
        if resolved_id:
            artifact_kwargs["artifact_id"] = resolved_id
        artifact = Artifact(**artifact_kwargs)
        append_jsonl(self._index_path(run_id), artifact)
        return artifact

    def register_artifact(
        self,
        *,
        run_id: str,
        node_id: str,
        kind: str,
        uri: str,
        mime_type: str = "application/octet-stream",
        summary: str = "",
        checksum: str = "",
        metadata: dict[str, Any] | None = None,
        artifact_id: str | None = None,
    ) -> Artifact:
        """Регистрирует уже существующий artifact URI без копирования файла.

        Args:
            run_id: Идентификатор ResearchRun.
            node_id: Идентификатор lineage node.
            kind: Универсальный тип artifact.
            uri: URI или локальный путь к artifact.
            mime_type: MIME-тип содержимого.
            summary: Краткое описание artifact.
            checksum: Контрольная сумма содержимого, если она известна.
            metadata: Дополнительные metadata artifact.

        Returns:
            Зарегистрированная запись Artifact.
        """

        resolved_id = self._resolve_unique_artifact_id(run_id, artifact_id)
        artifact_kwargs: dict[str, Any] = {
            "run_id": run_id,
            "node_id": node_id,
            "kind": kind,
            "uri": uri,
            "mime_type": mime_type,
            "summary": summary,
            "checksum": checksum,
            "metadata": normalize_artifact_metadata(metadata, kind=kind),
        }
        if resolved_id:
            artifact_kwargs["artifact_id"] = resolved_id
        artifact = Artifact(**artifact_kwargs)
        append_jsonl(self._index_path(run_id), artifact)
        return artifact

    def register_file_artifact(
        self,
        *,
        run_id: str,
        node_id: str,
        kind: str,
        path: str | Path,
        mime_type: str = "application/octet-stream",
        summary: str = "",
        metadata: dict[str, Any] | None = None,
        artifact_id: str | None = None,
    ) -> Artifact:
        """Регистрирует локальный файл как artifact с расчетом checksum.

        Args:
            run_id: Идентификатор ResearchRun.
            node_id: Идентификатор lineage node.
            kind: Универсальный тип artifact.
            path: Путь к существующему локальному файлу.
            mime_type: MIME-тип содержимого.
            summary: Краткое описание artifact.
            metadata: Дополнительные metadata artifact.

        Returns:
            Зарегистрированная запись Artifact.
        """

        source = Path(path).resolve()
        payload = source.read_bytes()
        return self.register_artifact(
            run_id=run_id,
            node_id=node_id,
            kind=kind,
            uri=str(source),
            mime_type=mime_type,
            summary=summary,
            checksum=hashlib.sha256(payload).hexdigest(),
            metadata=metadata,
            artifact_id=artifact_id,
        )

    def list_artifacts(self, run_id: str) -> list[Artifact]:
        """Возвращает все artifacts выбранного запуска.

        Args:
            run_id: Идентификатор ResearchRun.

        Returns:
            Список Artifact из индекса запуска.
        """

        artifacts: list[Artifact] = []
        for row in read_jsonl(self._index_path(run_id)):
            kind = str(row.get("kind") or "")
            row["metadata"] = normalize_artifact_metadata(
                row.get("metadata"),
                kind=kind,
            )
            artifacts.append(Artifact.model_validate(row))
        return artifacts

    def get_artifact(self, run_id: str, artifact_id: str) -> Artifact | None:
        """Возвращает artifact по идентификатору.

        Args:
            run_id: Идентификатор ResearchRun.
            artifact_id: Идентификатор artifact.

        Returns:
            Artifact или ``None``, если запись не найдена.
        """

        return next(
            (
                artifact
                for artifact in self.list_artifacts(run_id)
                if artifact.artifact_id == artifact_id
            ),
            None,
        )

    def list_node_artifacts(self, run_id: str, node_id: str) -> list[Artifact]:
        """Возвращает artifacts, созданные конкретным lineage node.

        Args:
            run_id: Идентификатор ResearchRun.
            node_id: Идентификатор lineage node.

        Returns:
            Список Artifact, где ``artifact.node_id`` совпадает с node_id.
        """

        return [
            artifact
            for artifact in self.list_artifacts(run_id)
            if artifact.node_id == node_id
        ]

    def register_branch_artifacts(
        self,
        *,
        source_run_id: str,
        target_run_id: str,
        target_node_id: str,
        artifact_ids: list[str],
        uri_overrides: dict[str, str] | None = None,
    ) -> list[Artifact]:
        """Регистрирует artifacts исходного запуска в новой ветке.

        Args:
            source_run_id: Идентификатор исходного ResearchRun.
            target_run_id: Идентификатор нового ResearchRun.
            target_node_id: Идентификатор branch node в новом запуске.
            artifact_ids: Список artifacts, которые нужно перенести в ветку.
            uri_overrides: Опциональная замена URI, если пользователь отредактировал artifact.

        Returns:
            Список зарегистрированных artifacts в целевом запуске.
        """

        uri_overrides = uri_overrides or {}
        registered: list[Artifact] = []
        for artifact_id in artifact_ids:
            source = self.get_artifact(source_run_id, artifact_id)
            if source is None:
                continue

            selected_uri = uri_overrides.get(artifact_id, source.uri)
            selected_path = Path(selected_uri)
            metadata = {
                **source.metadata,
                "branched_from_run_id": source_run_id,
                "branched_from_artifact_id": artifact_id,
                "branched_from_uri": source.uri,
                "branched_from_checksum": source.checksum,
                "branch_artifact": True,
                "edited_override": artifact_id in uri_overrides,
            }
            if selected_path.exists() and selected_path.is_file():
                registered.append(
                    self.register_file_artifact(
                        run_id=target_run_id,
                        node_id=target_node_id,
                        kind=source.kind,
                        path=selected_path,
                        mime_type=source.mime_type,
                        summary=source.summary,
                        metadata=metadata,
                    )
                )
                continue

            registered.append(
                self.register_artifact(
                    run_id=target_run_id,
                    node_id=target_node_id,
                    kind=source.kind,
                    uri=selected_uri,
                    mime_type=source.mime_type,
                    summary=source.summary,
                    checksum=source.checksum,
                    metadata=metadata,
                )
            )
        return registered

    def _resolve_unique_artifact_id(
        self,
        run_id: str,
        artifact_id: str | None,
    ) -> str | None:
        """Гарантирует уникальность пользовательского artifact_id внутри запуска.

        Args:
            run_id: Идентификатор ResearchRun.
            artifact_id: Желаемый человеко-читаемый идентификатор или ``None``.

        Returns:
            Свободный идентификатор: исходный, либо с числовым суффиксом при коллизии.
            Возвращает ``None``, если пользователь не передал artifact_id (тогда сработает
            UUID из default_factory модели Artifact).
        """

        if not artifact_id:
            return None
        existing_ids = {artifact.artifact_id for artifact in self.list_artifacts(run_id)}
        if artifact_id not in existing_ids:
            return artifact_id
        index = 2
        while True:
            candidate = f"{artifact_id}_{index}"
            if candidate not in existing_ids:
                return candidate
            index += 1

    def _index_path(self, run_id: str) -> Path:
        """Возвращает путь к JSONL-индексу artifacts запуска.

        Args:
            run_id: Идентификатор ResearchRun.

        Returns:
            Путь к ``artifacts.jsonl``.
        """

        return self.runs_dir / run_id / "artifacts.jsonl"

    @staticmethod
    def _is_subpath(path: Path, root: Path) -> bool:
        """Проверяет, находится ли path внутри root.

        Args:
            path: Проверяемый путь.
            root: Корневой каталог, за пределы которого нельзя выходить.

        Returns:
            ``True``, если path является дочерним путем root.
        """

        try:
            path.relative_to(root)
            return True
        except ValueError:
            return False


__all__ = ["ArtifactService"]
