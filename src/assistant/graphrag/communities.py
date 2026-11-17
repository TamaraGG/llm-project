"""Сообщества графа и их сводки. Требование FR-3."""
from typing import Any

import networkx as nx
from networkx.algorithms.community import louvain_communities

from assistant.infra.llm import LLMClient, SEED
from assistant.infra.schemas import Community


def detect_communities(graph: nx.DiGraph) -> list[Community]:
    """Разбивает узлы на сообщества алгоритмом Лувена с фиксированным `seed`.

    Каждый узел входит ровно в одно сообщество. Поле `summary` здесь остаётся пустым.
    """
    # снять направление рёбер, разбить с учётом weight, вернуть Community с номерами по порядку
    undirected_graph: nx.Graph = graph.to_undirected()

    community_sets: list[set[Any]] = louvain_communities(undirected_graph, weight="weight", seed=SEED)

    communities: list[Community] = [Community(community_id=i, nodes=list(c)) for i, c in enumerate(community_sets)]

    return communities


def summarize_community(community: Community, graph: nx.DiGraph, llm: LLMClient) -> str:
    """Сводка сообщества не длиннее 150 слов: какие сущности входят, как связаны, что встречается чаще.

    Все числа считает код по весам рёбер и передаёт модели готовыми. Модель их только формулирует.
    """
    # TODO: собрать факты и подсчёты сообщества, попросить модель написать сводку
    raise NotImplementedError("Реализуйте summarize_community")
