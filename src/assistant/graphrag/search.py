"""Два режима поиска по графу. Требования FR-4, FR-5 и FR-10."""
from pydantic import BaseModel, Field, ValidationError

from assistant.graphrag.extra_functions.search_functions import (
    build_global_reduce_messages,
    build_global_search_messages,
    build_local_search_context,
    build_local_search_messages,
    clean_sources_attribute,
    find_ego_paths,
    find_paths_between_nodes,
    find_replacement_chains,
    get_path_edges,
)
from assistant.graphrag.extraction import _clean_json_text
from assistant.graphrag.prompts.search_prompts import REFUSAL_SEARCH_ANSWER
from assistant.infra.grounding import code_type, find_codes, find_names
from assistant.infra.llm import LLMClient
from assistant.infra.schemas import MAP_SCHEMA, Answer, KnowledgeBase


def local_search(question: str, kb: KnowledgeBase, llm: LLMClient) -> Answer:
    """Отвечает на вопрос о конкретных сущностях по фактам из графа.

    Опорные сущности ищутся без модели: коды через `find_codes`, названия через `find_names`.
    При двух и более опорных сущностях берутся факты на путях длиной до трёх связей между ними
    и цепочки `REPLACED_BY` от деталей на этих путях. При одной берётся окрестность на глубину
    до двух связей. Модель получает пронумерованный список фактов и отвечает только по нему.
    Если в вопросе есть код, которого нет в графе, или фактов нет, возвращается
    `Answer(..., refused=True)` без вызова модели.
    """
    # опорные сущности, отбор фактов, запрос к модели, источники

    graph = kb.graph

    nodes: list[str] = [f"{code_type(code)}:{code}" for code in find_codes(question)]
    nodes += [f"{kind}:{name}" for kind, name in find_names(question, kb.vocabulary)]

    if not nodes or any(node not in graph for node in nodes):
        return Answer(text=REFUSAL_SEARCH_ANSWER, refused=True)

    ego_paths: list[list[dict]] = []
    connection_paths: list[list[dict]] = []
    replaced_by_paths: list[list[dict]] = []

    if len(nodes) == 1:
        raw_ego_paths = find_ego_paths(graph, nodes[0], radius=2)
        ego_paths = [get_path_edges(graph, path) for path in raw_ego_paths]
        replacement_roots = sorted({
            node
            for path in [nodes, *raw_ego_paths]
            for node in path
            if graph.nodes[node].get("type") == "Part"
        })
    else:
        raw_connection_paths = find_paths_between_nodes(graph, nodes, max_len=3)
        connection_paths = [get_path_edges(graph, path) for path in raw_connection_paths]
        replacement_roots = sorted({
            node
            for path in raw_connection_paths
            for node in path
            if graph.nodes[node].get("type") == "Part"
        })

    for node in replacement_roots:
        replaced_by_paths.extend(
            get_path_edges(graph, path) for path in find_replacement_chains(node, graph)
        )

    if len(ego_paths) == 0 and len(connection_paths) == 0 and len(replaced_by_paths) == 0:
        return Answer(text=REFUSAL_SEARCH_ANSWER, refused=True)

    facts, sources = build_local_search_context(ego_paths, connection_paths, replaced_by_paths)

    if not facts:
        return Answer(text=REFUSAL_SEARCH_ANSWER, refused=True)

    response = llm.chat(messages=build_local_search_messages(question, facts))

    normalized_response = response.content.strip().rstrip(".!?…").rstrip().casefold()
    if normalized_response == REFUSAL_SEARCH_ANSWER:
        return Answer(text=REFUSAL_SEARCH_ANSWER, sources=sources, facts=facts, refused=True)
    return Answer(text=response.content, sources=sources, facts=facts, refused=False)

class GlobalSearchOutput(BaseModel):
    answer: str
    score: int = Field(ge=0, le=10)

def global_search(question: str, kb: KnowledgeBase, llm: LLMClient) -> Answer:
    """Отвечает на обзорный вопрос по сводкам сообществ в два шага.

    Шаг map: для каждой сводки модель даёт частичный ответ и оценку полезности от 0 до 10.
    Шаг reduce: частичные ответы с оценкой выше 0 объединяются в итоговый ответ.
    Если таких ответов нет, возвращается `Answer(..., refused=True)`.
    """
    partial_answers: list[GlobalSearchOutput] = []
    sources: set[str] = set()

    for community in kb.communities:
        output = llm.chat(messages=build_global_search_messages(question, community.summary), json_schema=MAP_SCHEMA)
        cleaned: str = _clean_json_text(output.content)
        try:
            partial_answer: GlobalSearchOutput = GlobalSearchOutput.model_validate_json(cleaned)
        except ValidationError:
            continue

        if partial_answer.score == 0:
            continue

        partial_answers.append(partial_answer)
        sources.update([source
                        for node in community.nodes
                        for source in clean_sources_attribute(kb.graph.nodes[node].get("sources", []))
                        ])

    if len(partial_answers) == 0:
        return Answer(text=REFUSAL_SEARCH_ANSWER, refused=True)

    response = llm.chat(messages=build_global_reduce_messages(
        question, [partial_answer.answer for partial_answer in partial_answers],
    ))
    final_answer = response.content
    refused = final_answer.strip().rstrip(".!?…").rstrip().casefold() == REFUSAL_SEARCH_ANSWER

    return Answer(text=final_answer, sources=sorted(sources), refused=refused)
