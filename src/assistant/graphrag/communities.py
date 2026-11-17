"""Сообщества графа и их сводки. Требование FR-3."""
import json
from typing import Any

import networkx as nx
from networkx.algorithms.community import louvain_communities

from assistant.graphrag.prompts.community_prompts import COMMUNITY_SUMMARY_PROMPT, COMMUNITY_SUMMARY_EXAMPLES
from assistant.infra.llm import SEED, LLMClient
from assistant.infra.schemas import Community

MAX_TOKENS: int = int(150 * 2.2)

def detect_communities(graph: nx.DiGraph) -> list[Community]:
    """Разбивает узлы на сообщества алгоритмом Лувена с фиксированным `seed`.

    Каждый узел входит ровно в одно сообщество. Поле `summary` здесь остаётся пустым.
    """
    # снять направление рёбер, разбить с учётом weight, вернуть Community с номерами по порядку
    undirected_graph: nx.Graph = graph.to_undirected()

    community_sets: list[set[Any]] = louvain_communities(undirected_graph, weight="weight", seed=SEED)

    communities: list[Community] = [Community(community_id=i, nodes=list(c)) for i, c in enumerate(community_sets)]

    return communities


def build_community_summary_messages(facts: dict[str, Any]) -> list[dict[str, str]]:
    messages = [{"role": "system", "content": COMMUNITY_SUMMARY_PROMPT}]
    for example, summary in COMMUNITY_SUMMARY_EXAMPLES:
        messages.append({"role": "user", "content": json.dumps(example, ensure_ascii=False, sort_keys=True)})
        messages.append({"role": "assistant", "content": summary})
    messages.append({"role": "user", "content": json.dumps(facts, ensure_ascii=False, sort_keys=True)})
    return messages


def summarize_community(community: Community, graph: nx.DiGraph, llm: LLMClient) -> str:
    """Сводка сообщества не длиннее 150 слов: какие сущности входят, как связаны, что встречается чаще.

    Все числа считает код по весам рёбер и передаёт модели готовыми. Модель их только формулирует.
    """
    entities: list[str] = community.nodes

    subgraph: nx.DiGraph = nx.subgraph(graph, entities)

    relation_with_object: dict[str, dict[str, int]] = {}
    relation_with_subject: dict[str, dict[str, int]] = {}

    relation_totals: dict[str, int] = {}

    all_relations: dict[tuple[str, str], dict[str, Any]] = {}

    for u, v, data in subgraph.edges(data=True):
        if data.get("weight", 0) == 0 or data.get("type") is None:
            continue
        rel_type = data["type"]
        rel_weight = data["weight"]

        relation_with_object[rel_type] = relation_with_object.get(rel_type, {})
        relation_with_object[rel_type][v] = relation_with_object[rel_type].get(v, 0)
        relation_with_object[rel_type][v] += rel_weight

        relation_with_subject[rel_type] = relation_with_subject.get(rel_type, {})
        relation_with_subject[rel_type][u] = relation_with_subject[rel_type].get(u, 0)
        relation_with_subject[rel_type][u] += rel_weight

        relation_totals[rel_type] = relation_totals.get(rel_type, 0) + data["weight"]

        all_relations[(u, v)] = {"type": rel_type, "weight": rel_weight}

    facts = {
        "entities": sorted(entities),
        "all_relations": [
            {"subject": subject, "object": object_, **data}
            for (subject, object_), data in sorted(all_relations.items())
        ],
        "relation_totals": relation_totals,
        "relation_with_object": relation_with_object,
        "relation_with_subject": relation_with_subject,
    }

    response = llm.chat(messages=build_community_summary_messages(facts), max_tokens=MAX_TOKENS)

    return " ".join(response.content.split()[:150])
