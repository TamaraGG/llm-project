"""Сборка графа знаний из результатов извлечения. Требование FR-2."""
from typing import Any

import networkx as nx

from assistant.infra.schemas import Entity, ExtractionResult, Relation


def _resolve_replacement_conflicts(
        graph: nx.DiGraph,
        nodes_in_replaced_by: set[tuple[str, str]],
        replaced_by_edges: dict[str, dict[str, Any]],
) -> None:
    for (subject_id, object_id) in nodes_in_replaced_by:
        forward = replaced_by_edges.get(subject_id, {}).get(object_id)
        reverse = replaced_by_edges.get(object_id, {}).get(subject_id)

        if forward is not None and reverse is not None:
            if forward["weight"] > reverse["weight"]:
                graph.add_edge(subject_id, object_id, **forward)
            elif forward["weight"] < reverse["weight"]:
                graph.add_edge(object_id, subject_id, **reverse)
            graph.graph["conflicts"] += 1
        elif forward is not None:
            graph.add_edge(subject_id, object_id, **forward)
        elif reverse is not None:
            graph.add_edge(object_id, subject_id, **reverse)


def build_graph(results: list[ExtractionResult]) -> nx.DiGraph:
    """Объединяет результаты по всем чанкам в один направленный граф.

    Узел: идентификатор `Entity.node_id`, атрибуты `type`, `name`, `sources` (список chunk_id).
    Ребро: атрибуты `type`, `weight` (число разных чанков, где связь найдена), `sources`.
    Связи, не соответствующие схеме (`Relation.matches_schema`), отбрасываются. Если между двумя
    деталями есть `REPLACED_BY` в обе стороны, остаётся связь с большим весом, при равенстве
    отбрасываются обе. Число таких конфликтов записывается в `graph.graph["conflicts"]`.
    """
    # узлы, рёбра с весами, проверка схемы, разбор встречных замен

    replaced_by_edges : dict[str, dict[str, Any]] = {}
    nodes_in_replaced_by: set[tuple[str, str]] = set()

    graph : nx.DiGraph = nx.DiGraph()
    graph.graph["conflicts"] = 0

    for result in results:
        chunk_id = result.chunk_id

        for entity in result.entities:

            if entity.node_id not in graph:
                graph.add_node(entity.node_id, type=entity.type, name=entity.name, sources=[chunk_id])
            elif chunk_id not in graph.nodes[entity.node_id]["sources"]:
                graph.nodes[entity.node_id]["sources"].append(chunk_id)

        for relation in result.relations:
            if not relation.matches_schema():
                continue

            object_id = relation.object.node_id
            subject_id = relation.subject.node_id
            rel_type = relation.type

            if rel_type == "REPLACED_BY":
                nodes_in_replaced_by.add((object_id, subject_id) if object_id < subject_id else (subject_id, object_id))
                if subject_id not in replaced_by_edges.keys():
                    replaced_by_edges[subject_id] = {object_id: {"type": "REPLACED_BY", "weight": 1, "sources": [chunk_id]}}
                elif object_id not in replaced_by_edges[subject_id].keys():
                    replaced_by_edges[subject_id][object_id] = {"type": "REPLACED_BY", "weight": 1, "sources": [chunk_id]}
                elif chunk_id not in replaced_by_edges[subject_id][object_id]["sources"]:
                    replaced_by_edges[subject_id][object_id]["weight"] += 1
                    replaced_by_edges[subject_id][object_id]["sources"] += [chunk_id]
            else:
                if graph.has_edge(subject_id, object_id):
                    if chunk_id not in graph[subject_id][object_id]["sources"]:
                        graph[subject_id][object_id]["weight"] += 1
                        graph[subject_id][object_id]["sources"] += [chunk_id]
                else:
                    graph.add_edge(subject_id, object_id, type=rel_type, weight=1, sources=[chunk_id])

    _resolve_replacement_conflicts(graph, nodes_in_replaced_by, replaced_by_edges)

    return graph
