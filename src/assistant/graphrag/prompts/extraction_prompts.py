import json


EXAMPLE_TEXT = """# Сервисный бюллетень SB-2025-01

Деталь P-1046 (вентилятор испарителя) снята с производства. Вместо неё устанавливается деталь P-1046B.

Поставщик детали P-1046B – «ВентПром». Детали взаимозаменяемы по посадочным местам и разъёмам, доработка оборудования не требуется.

Новые заказы оформляются только на P-1046B."""

EXAMPLE_CHUNK_EXTRACTION = json.dumps(
    {
        "entities": [
            {"type": "Part", "name": "P-1046"},
            {"type": "Part", "name": "P-1046B"},
            {"type": "Component", "name": "вентилятор испарителя"},
            {"type": "Supplier", "name": "ВентПром"},
        ],
        "relations": [
            {
                "subject_type": "Part",
                "subject": "P-1046",
                "type": "REPLACED_BY",
                "object_type": "Part",
                "object": "P-1046B",
            },
            {
                "subject_type": "Part",
                "subject": "P-1046",
                "type": "IS_A",
                "object_type": "Component",
                "object": "вентилятор испарителя",
            },
            {
                "subject_type": "Part",
                "subject": "P-1046B",
                "type": "SUPPLIED_BY",
                "object_type": "Supplier",
                "object": "ВентПром",
            },
        ],
    },
    ensure_ascii=False,
    indent=2,
)

EXTRACTION_PROMPT = f"""# ЗАДАЧА
Извлеки из текста документа сущности и связи между ними строго по схеме графа знаний. Формат ответа — JSON.

## Схема графа знаний
Допустимые типы сущностей (type):
- ProductLine (линейка оборудования: Арктика, Бриз, Гольфстрим)
- Model (код модели: AR-210, BZ-110, GS-410 и т.д.)
- Component (название узла строчными буквами: вентилятор испарителя, пусковое реле, компрессор и т.д.)
- Part (номер детали: P-1046, P-1046B, P-1042 и т.д.)
- ErrorCode (код ошибки: E-17 и т.д.)
- Procedure (номер регламента ремонта: SP-104 и т.д.)
- Supplier (поставщик: ВентПром, ХолодКомплект и т.д.)

Направления связей (subject -> type -> object):
- Model -> BELONGS_TO -> ProductLine
- Model -> HAS_PART -> Part
- Part -> IS_A -> Component
- ErrorCode -> INDICATES -> Component
- ErrorCode -> FIXED_BY -> Procedure
- Part -> REPLACED_BY -> Part
- Part -> SUPPLIED_BY -> Supplier
- Model -> FAILED -> Component

Правила:
1. Извлекай только то, что явно упоминается в тексте. Не придумывай сущности.
2. Коды пиши латинскими буквами в верхнем регистре (P-1046, E-17, SP-104).
3. Названия узлов (Component) пиши строчными буквами в именительном падеже (например, «вентилятор испарителя»).
"""
