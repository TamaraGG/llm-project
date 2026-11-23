from assistant.infra.grounding import code_type, normalize_code


def function_schema(name: str, description: str, properties: dict, required: list[str]) -> dict:
    return {
        "type": "function",
        "function": {
            "name": name,
            "description": description,
            "parameters": {
                "type": "object",
                "properties": properties,
                "required": required,
                "additionalProperties": False,
            },
        },
    }


def check_part_id(part_id: str) -> tuple[bool, str]:
    if not isinstance(part_id, str):
        return False, "part_id должен быть строкой"
    normalized = normalize_code(part_id)
    if normalized is None or normalized != part_id or code_type(normalized) != "Part":
        return False, "Некорректный номер детали, ожидается P-1042 или P-1042B"

    return True, ""
