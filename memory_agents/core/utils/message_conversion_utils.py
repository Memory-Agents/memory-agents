import logging

logger = logging.getLogger()


def ensure_message_content_is_str(content: str | list[str | dict] | None) -> str:
    if not isinstance(content, str):
        content = str(content)
        logger.error(
            "The retrieved message content is not a str, this might be unexpected behavior"
        )
        return content
    return content
