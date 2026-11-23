import itertools
from typing import Any

import networkx as nx

from assistant.graphrag.prompts.search_prompts import LOCAL_SEARCH_EXAMPLES, LOCAL_SEARCH_PROMPT


def get_path_edges(graph: nx.DiGraph, path: list[str]) -> list[dict]:
    """Из массивов нод (путь, например, [node1, node2, node3])
    делает список словарей, которые соответствуют ребрам из графа"""

    result: list[dict] = []
    for u, v in zip(path, path[1:], strict=False):
        if graph.has_edge(u, v):
            result.append({"subject": u, "object": v, **graph[u][v]})
        if u != v and graph.has_edge(v, u):
            result.append({"subject": v, "object": u, **graph[v][u]})
    return result


def find_directional_replacement_chains(node: str, graph: nx.DiGraph, is_direction_out: bool) -> list[list[str]]:
    """Находит цепочки со связью ребер REPLACED_BY, содержащие узел node. Только в одном направлении"""

    if node not in graph:
        return []

    chains: list[list[str]] = []
    stack: list[list[str]] = [[node]]

    edges = graph.out_edges if is_direction_out else graph.in_edges

    while stack:
        current_path = stack.pop()

        next_nodes: list[str] = []
        for source, target, data in edges(current_path[-1], data=True):
            next_node = target if is_direction_out else source
            if data.get("type") == "REPLACED_BY" and next_node not in current_path:
                next_nodes.append(next_node)

        if not next_nodes:
            if len(current_path) > 1:
                chains.append(current_path)
        else:
            stack.extend(current_path + [next_node] for next_node in next_nodes)

    return chains


def find_replacement_chains(node: str, graph: nx.DiGraph) -> list[list[str]]:
    """Находит все цепочки со связью ребер REPLACED_BY, содержащие узел node"""

    if node not in graph:
        return []

    in_chains = [
        chain[::-1]
        for chain in find_directional_replacement_chains(node, graph, False)
    ] or [[node]]
    out_chains = find_directional_replacement_chains(node, graph, True) or [[node]]

    chains: list[list[str]] = []
    for in_chain, out_chain in itertools.product(in_chains, out_chains):
        chain = in_chain + out_chain[1:]
        if len(chain) > 1 and len(chain) == len(set(chain)):
            chains.append(chain)

    return chains

def find_ego_paths(graph: nx.DiGraph, center_node: Any, radius=2) -> list[list]:
    """Находит окрестность ноды center_node на глубину до двух узлов.
    Возвращает список путей всех, состоящий из названий нод"""

    ego_subgraph = nx.ego_graph(graph, n=center_node, radius=radius, undirected=True).to_undirected(as_view=True)
    targets = [node for node in ego_subgraph if node != center_node]

    paths: list[list[Any]] = list(
        nx.all_simple_paths(ego_subgraph, source=center_node, target=targets, cutoff=radius)
    )

    unique_paths = list(dict.fromkeys(map(tuple, paths)))

    subpaths: set[tuple[Any, ...]] = set()
    for path in unique_paths:
        n = len(path)
        for i in range(n):
            for j in range(i + 1, n + 1):
                if j - i < n:
                    subpaths.add(path[i:j])

    return [list(path) for path in unique_paths if path not in subpaths]

def find_paths_between_nodes(graph: nx.DiGraph, nodes: list[Any], max_len = 3) -> list[list]:
    """Находит цепочки, соединяющие ноды, возвращает список таких цепочек"""
    paths: list[list[str]] = []

    undirected = graph.to_undirected()
    for u, v in list(itertools.combinations(nodes, 2)):
        for path in nx.all_simple_paths(undirected, source=u, target=v, cutoff=max_len):
            paths.append(path)

    return paths

def build_local_search_context(
    ego_paths: list[list[dict[str, Any]]],
    connection_paths: list[list[dict[str, Any]]],
    replaced_by_paths: list[list[dict[str, Any]]],
) -> tuple[list[str], list[str]]:
    """Составляет из списков путей факты для LLM, а также список ресурсов"""

    facts: list[str] = []
    sources: set[str] = set()
    seen: set[tuple[tuple[str, str, str], ...]] = set()
    total_chars = 0

    other_paths = [
        list(edges)
        for path in itertools.chain(ego_paths, connection_paths)
        for keep, edges in itertools.groupby(path, key=lambda edge: edge["type"] != "REPLACED_BY")
        if keep
    ]
    paths = itertools.chain(
        ((path, True) for path in replaced_by_paths),
        ((path, False) for path in other_paths),
    )
    for path, required in paths:
        if not path:
            continue

        key = tuple((edge["subject"], edge["type"], edge["object"]) for edge in path)
        if key not in seen:
            current = path[0]["subject"]
            if len(path) > 1 and current in (path[1]["subject"], path[1]["object"]):
                current = path[0]["object"]
            parts = [current.split(":", 1)[-1]]
            for edge in path:
                subject, relation, object_ = edge["subject"], edge["type"], edge["object"]
                if current == subject:
                    parts.append(f" -{relation}-> {object_.split(':', 1)[-1]}")
                    current = object_
                elif current == object_:
                    parts.append(f" <-{relation}- {subject.split(':', 1)[-1]}")
                    current = subject
                else:
                    parts.append(f"; {subject.split(':', 1)[-1]} -{relation}-> {object_.split(':', 1)[-1]}")
                    current = object_

            fact = f"{len(facts) + 1}. {''.join(parts)}"
            extra_chars = len(fact) + (1 if facts else 0)
            if total_chars + extra_chars > 8000:
                if required:
                    return [], []
                continue

            facts.append(fact)
            seen.add(key)
            total_chars += extra_chars

        for edge in path:
            sources.update(source.split("#", 1)[0] for source in edge.get("sources", []))

    return facts, sorted(sources)

def format_local_search_input(question: str, facts: list[str]) -> str:
    fact_text = "\n".join(facts)
    return f"ВОПРОС:\n{question}\n\nФАКТЫ:\n{fact_text}"


def build_local_search_messages(question: str, facts: list[str]) -> list[dict[str, str]]:
    """Составляет сообщения для LLM с системным промптом, few-shot, вопросом пользователя и фактами"""

    messages = [{"role": "system", "content": LOCAL_SEARCH_PROMPT}]
    for example, answer in LOCAL_SEARCH_EXAMPLES:
        example_facts = [f"{index}. {fact}" for index, fact in enumerate(example["facts"], start=1)]
        messages.append({"role": "user", "content": format_local_search_input(example["question"], example_facts)})
        messages.append({"role": "assistant", "content": answer})
    messages.append({"role": "user", "content": format_local_search_input(question, facts)})
    return messages
