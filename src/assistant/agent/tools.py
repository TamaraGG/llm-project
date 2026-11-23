"""Инструменты агента. Требование FR-6."""

from collections.abc import Callable

from assistant.agent.extra_functions.tools_functions import check_part_id, function_schema
from assistant.graphrag.search import global_search, local_search
from assistant.infra.erp_client import ErpClient, ErpNotFound, ErpRejected
from assistant.infra.llm import LLMClient
from assistant.infra.schemas import KnowledgeBase

WAREHOUSES = ["Москва", "Санкт-Петербург", "Казань", "Новосибирск"]

# Описания пяти инструментов в формате function calling:
# search_knowledge, summarize_topic, check_stock, get_price, create_service_order.
TOOL_SCHEMAS: list[dict] = [
    function_schema(
        name="search_knowledge",
        description=(
            "Отвечает на вопросы об оборудовании, кодах ошибок, деталях и их заменах, "
            "используя связанные факты графа знаний. Возвращает ответ, подтверждающие факты и документы-источники."
        ),
        properties={
            "question": {
                "type": "string",
                "description": "Вопрос об оборудовании, неисправности, детали или порядке обслуживания.",
            },
        },
        required=["question"],
    ),
    function_schema(
        name="summarize_topic",
        description=(
            "Подготавливает обзор по широкой теме на основе сводок базы знаний, "
            "например о частых неисправностях или общих рекомендациях по обслуживанию. "
            "Возвращает ответ и источники."
        ),
        properties={
            "question": {
                "type": "string",
                "description": "Тема или обзорный вопрос для обобщения сведений из базы знаний.",
            },
        },
        required=["question"],
    ),
    function_schema(
        name="check_stock",
        description="Возвращает остаток детали на выбранном складе и признак снятия детали с производства.",
        properties={
            "part_id": {
                "type": "string",
                "description": "Номер детали, например P-1042 или P-1042B.",
                "pattern": "^P-[0-9]{4}[A-Z]?$",
            },
            "warehouse": {
                "type": "string",
                "description": "Город склада, на котором нужно проверить наличие детали.",
                "enum": WAREHOUSES,
            },
        },
        required=["part_id", "warehouse"],
    ),
    function_schema(
        name="get_price",
        description="Возвращает цену детали в рублях и признак снятия детали с производства из ERP.",
        properties={
            "part_id": {
                "type": "string",
                "description": "Номер детали, например P-1042 или P-1042B.",
                "pattern": "^P-[0-9]{4}[A-Z]?$",
            },
        },
        required=["part_id"],
    ),
    function_schema(
        name="create_service_order",
        description=(
            "Создаёт сервисный заказ на указанное количество деталей с выбранного склада "
            "для заданной модели оборудования после явного подтверждения пользователя."
        ),
        properties={
            "part_id": {
                "type": "string",
                "description": "Номер заказываемой детали, например P-1042 или P-1042B.",
                "pattern": "^P-[0-9]{4}[A-Z]?$",
            },
            "warehouse": {
                "type": "string",
                "description": "Город склада, с которого заказывается деталь.",
                "enum": WAREHOUSES,
            },
            "qty": {
                "type": "integer",
                "description": "Количество заказываемых деталей; целое число не меньше одного.",
                "minimum": 1,
            },
            "model": {
                "type": "string",
                "description": "Код модели оборудования, для которой заказывается деталь, например AR-210.",
            },
        },
        required=["part_id", "warehouse", "qty", "model"],
    ),
]


def build_tools(kb: KnowledgeBase, llm: LLMClient, erp: ErpClient) -> dict[str, Callable[..., dict]]:
    """Словарь «имя инструмента -> функция». Каждая функция возвращает словарь.

    Функция проверяет свои аргументы до обращения к ERP: склад из списка, номер детали вида P-1042
    или P-1042B, количество больше нуля. При ошибке возвращает {"error": "..."} и ничего не выполняет.
    `ErpNotFound` и `ErpRejected` превращаются в {"error": "..."} с разным текстом, чтобы исходы
    оставались различимыми. `ErpUnavailable` передаётся дальше: повторы делает агент (FR-8).
    """
    # пять функций поверх local_search, global_search и клиента ERP

    def search_knowledge(question: str) -> dict[str, str | list[str] | bool]:
        result = local_search(question, kb, llm)
        return {"answer": result.text, "facts": result.facts, "sources": result.sources, "refused": result.refused}

    def summarize_topic(question: str) -> dict[str, str | list[str] | bool]:
        result = global_search(question, kb, llm)
        return {"answer": result.text, "sources": result.sources, "refused": result.refused}

    def check_stock(part_id: str, warehouse: str) -> dict[str, int | bool | str]:
        errors: list[str] = []

        is_part_id_correct, error = check_part_id(part_id)
        if not is_part_id_correct:
            errors.append(error)

        if warehouse not in WAREHOUSES:
            errors.append("Склада нет в списке доступных.")

        if errors:
            return {"error": "\n".join(errors)}

        try:
            result = erp.get_stock(part_id, warehouse)
            if result.get("qty") is None or not isinstance(result.get("discontinued"), bool):
                return {"error": "ERP вернул неверный ответ."}
            return {"qty": result.get("qty"), "discontinued": result.get("discontinued")}
        except (ErpNotFound, ErpRejected) as e:
            return {"error": str(e)}

    def get_price(part_id: str) -> dict[str, float | bool | str]:

        is_part_id_correct, error = check_part_id(part_id)
        if not is_part_id_correct:
            return {"error": error}

        try:
            result = erp.get_price(part_id)
            if result.get("price_rub") is None or not isinstance(result.get("discontinued"), bool):
                return {"error": "ERP вернул неверный ответ."}
            return {"price_rub": result.get("price_rub"), "discontinued": result.get("discontinued")}
        except (ErpNotFound, ErpRejected) as e:
            return {"error": str(e)}

    def create_service_order(part_id: str, warehouse: str, qty: int, model: str) -> dict[str, int | str]:
        errors: list[str] = []

        is_part_id_correct, error = check_part_id(part_id)
        if not is_part_id_correct:
            errors.append(error)
        if warehouse not in WAREHOUSES:
            errors.append("Склада нет в списке доступных.")
        if type(qty) is not int or qty < 1:
            errors.append("Количество должно быть целым числом не меньше 1.")

        if errors:
            return {"error": "\n".join(errors)}

        try:
            result = erp.create_order(part_id, warehouse, qty, model)
            if result.get("order_id") is None:
                return {"error": "ERP вернул неверный ответ."}
            return {"order_id": result.get("order_id")}
        except (ErpNotFound, ErpRejected) as e:
            return {"error": str(e)}

    tools_functions: list[Callable] = [
        search_knowledge,
        summarize_topic,
        check_stock,
        get_price,
        create_service_order,
    ]

    return {tool.__name__: tool for tool in tools_functions}
