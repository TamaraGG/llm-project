"""Извлечение сущностей и связей из чанка. Требование FR-1."""

import re

from pydantic import BaseModel, ValidationError

from assistant.graphrag.prompts.extraction_prompts import (
    EXAMPLE_CHUNK_EXTRACTION,
    EXAMPLE_TEXT,
    EXTRACTION_PROMPT,
)
from assistant.infra.config import vocabulary
from assistant.infra.grounding import CODE_TYPES, canonical_name, find_codes
from assistant.infra.llm import LLMClient
from assistant.infra.schemas import (
    ENTITY_TYPES,
    EXTRACTION_SCHEMA,
    RELATION_SIGNATURES,
    Chunk,
    Entity,
    ExtractionResult,
    Relation,
)


class RawEntity(BaseModel):
    type: str
    name: str


class RawRelation(BaseModel):
    subject_type: str
    subject: str
    type: str
    object_type: str
    object: str


class RawExtraction(BaseModel):
    entities: list[RawEntity] = []
    relations: list[RawRelation] = []


def _clean_json_text(text: str) -> str:
    """Очищает строку от markdown-блоков ```json ... ```."""
    stripped = text.strip()
    match = re.search(r"```(?:json)?\s*([\s\S]*?)\s*```", stripped)
    return match.group(1).strip() if match else stripped


def canonical_type(raw_type: str) -> str | None:
    if not isinstance(raw_type, str):
        return None
    type_map = {t.lower(): t for t in ENTITY_TYPES}
    return type_map.get(raw_type.strip().lower())


def check_one_entity(
    raw_type: str,
    raw_name: str,
    dictionary: dict[str, list[str]],
    valid_codes: set[str],
) -> Entity | None:
    """Возвращает заполненный Entity, если тип и имя правильные, иначе - None."""
    canon_type = canonical_type(raw_type)
    if not canon_type:
        return None

    canon_name = canonical_name(canon_type, raw_name, dictionary)
    if not canon_name or (canon_type in CODE_TYPES and canon_name not in valid_codes):
        return None

    return Entity(type=canon_type, name=canon_name)


def check_entities(raw_entities: list[RawEntity], chunk_codes: set[str]) -> tuple[list[Entity], int]:
    """Проверяет сущности по тексту чанка и словарю."""
    valid_entities: list[Entity] = []
    seen_node_ids: set[str] = set()
    dropped: int = 0
    dictionary = vocabulary()

    for raw in raw_entities:
        entity = check_one_entity(
            raw_type=raw.type,
            raw_name=raw.name,
            dictionary=dictionary,
            valid_codes=chunk_codes,
        )
        if entity is None:
            dropped += 1
        elif entity.node_id not in seen_node_ids:
            seen_node_ids.add(entity.node_id)
            valid_entities.append(entity)

    return valid_entities, dropped


def canonical_relation_type(raw_rel_type: str) -> str | None:
    if not isinstance(raw_rel_type, str):
        return None
    rel_type_map = {s.lower(): s for s in RELATION_SIGNATURES}
    return rel_type_map.get(raw_rel_type.strip().lower())


def check_relations(raw_relations: list[RawRelation], chunk_codes: set[str]) -> tuple[list[Relation], int]:
    """Проверяет связи по типам, схеме и тексту чанка."""
    dropped: int = 0
    valid_relations: list[Relation] = []
    seen_rel_keys: set[tuple[str, str, str]] = set()
    dictionary = vocabulary()

    for raw in raw_relations:
        object_entity: Entity | None = check_one_entity(raw.object_type, raw.object, dictionary, chunk_codes)
        if object_entity is None:
            dropped += 1
            continue

        subject_entity: Entity | None = check_one_entity(raw.subject_type, raw.subject, dictionary, chunk_codes)
        if subject_entity is None:
            dropped += 1
            continue

        canon_rel_type = canonical_relation_type(raw.type)
        if canon_rel_type is None:
            dropped += 1
            continue

        relation = Relation(subject=subject_entity, type=canon_rel_type, object=object_entity)
        if not relation.matches_schema():
            dropped += 1
            continue

        rel_key = (subject_entity.node_id, canon_rel_type, object_entity.node_id)
        if rel_key not in seen_rel_keys:
            seen_rel_keys.add(rel_key)
            valid_relations.append(relation)

    return valid_relations, dropped


def extract_chunk(chunk: Chunk, llm: LLMClient) -> ExtractionResult:
    """Возвращает сущности и связи чанка строго по схеме графа.

    Модель вызывается через `llm.chat(..., json_schema=EXTRACTION_SCHEMA)`. Запрос описывает схему
    с направлением каждой связи и содержит хотя бы один пример. Всё, что не подтверждается текстом,
    отбрасывается и считается в `dropped`: код, которого нет в чанке, название не из словаря,
    связь с отброшенным концом. Пригодятся `find_codes` и `canonical_name` из `assistant.infra.grounding`.
    Функция не падает ни на каком ответе модели: одна повторная попытка, затем пустой результат.
    """
    result = ExtractionResult(
        chunk_id=chunk.chunk_id,
        doc_id=chunk.doc_id,
        entities=[],
        relations=[],
        dropped=0,
    )

    messages = [
        {"role": "system", "content": EXTRACTION_PROMPT},
        {"role": "user", "content": f"Извлеки сущности и связи из текста:\n{EXAMPLE_TEXT}"},
        {"role": "assistant", "content": f"{EXAMPLE_CHUNK_EXTRACTION}"},
        {"role": "user", "content": f"Извлеки сущности и связи из текста:\n{chunk.text}"},
    ]

    for _ in range(2):
        output = llm.chat(messages=messages, json_schema=EXTRACTION_SCHEMA, max_tokens=4096)
        try:
            cleaned = _clean_json_text(output.content)
            extraction = RawExtraction.model_validate_json(cleaned)
        except (ValidationError, ValueError, TypeError):
            continue

        chunk_codes: set[str] = set(find_codes(chunk.text))
        entities, dropped_ent = check_entities(extraction.entities, chunk_codes)
        relations, dropped_rel = check_relations(extraction.relations, chunk_codes)

        all_entities = list(entities)
        seen_nodes = {e.node_id for e in all_entities}
        for r in relations:
            for end in (r.subject, r.object):
                if end.node_id not in seen_nodes:
                    seen_nodes.add(end.node_id)
                    all_entities.append(end)

        result.entities = all_entities
        result.relations = relations
        result.dropped = dropped_rel + dropped_ent
        break

    return result
