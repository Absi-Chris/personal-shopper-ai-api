import os
import re
from functools import lru_cache

from guardrails import Guard, OnFailAction
from guardrails.settings import settings

_CART_CHANGE_CLAIM = re.compile(r"\b(?:added|removed|replaced|swapped|updated|deleted)\b", re.I)
_SUCCESSFUL_CART_TOOL = re.compile(
    r"\b(?:added\b|removed from cart\b|replaced\b|swapped\b)", re.I
)
_TOPIC_VALIDATION_MIN_WORDS = 4
_SHOPPING_INTENT = re.compile(
    r"\b(?:recommend|suggest|find|search|compare|buy|purchase|add|remove|replace|swap|show|list)(?:\s+down)?\b|\bgive\s+me\b",
    re.I,
)
_PRODUCT_TERM = re.compile(
    r"\b(?:laptop|computer|headphones?|earbuds?|phone|smartphone|monitor|keyboard|mouse|tablet|camera|smartwatch|product|item|brand|model)\b",
    re.I,
)
_INPUT_BLOCKED = "I can help with shopping and product-related questions. Please rephrase your request."
_OUTPUT_BLOCKED = "I couldn't provide that response safely. Please rephrase your request."
_OUTPUT_PII_ENTITIES = ("EMAIL_ADDRESS", "PHONE_NUMBER")


def _configure_guardrails():
    settings.use_server = False


@lru_cache(maxsize=1)
def _jailbreak_guard() -> Guard:
    from guardrails_ai.detect_jailbreak import DetectJailbreak

    _configure_guardrails()
    return Guard(use_server=False).use(
        DetectJailbreak(use_local=True, on_fail=OnFailAction.EXCEPTION),
    )


@lru_cache(maxsize=1)
def _topic_guard() -> Guard:
    from guardrails_ai.restricttotopic import RestrictToTopic

    _configure_guardrails()
    return Guard(use_server=False).use(
        RestrictToTopic(
            valid_topics=[
                "shopping",
                "consumer products",
                "product information",
                "product comparisons",
                "shopping carts and orders",
                "product documents and warranties",
                "greetings and assistant capabilities",
            ],
            use_local=True,
            disable_llm=True,
            on_fail=OnFailAction.EXCEPTION,
        ),
    )


@lru_cache(maxsize=1)
def _output_guard() -> Guard:
    from guardrails_ai.guardrails_pii import GuardrailsPII
    from guardrails_ai.profanity_free import ProfanityFree

    _configure_guardrails()
    validators = [
        ProfanityFree(on_fail=OnFailAction.EXCEPTION),
        GuardrailsPII(
            entities=list(_OUTPUT_PII_ENTITIES),
            on_fail=OnFailAction.EXCEPTION,
        ),
    ]
    competitors = [
        name.strip()
        for name in os.getenv("GUARDRAIL_COMPETITORS", "").split(",")
        if name.strip()
    ]
    if competitors:
        from guardrails_ai.competitor_check import CompetitorCheck

        validators.append(CompetitorCheck(competitors, on_fail=OnFailAction.EXCEPTION))
    return Guard(use_server=False).use(*validators, on="output")


def validate_user_input(message: str) -> str | None:
    try:
        jailbreak_result = _jailbreak_guard().validate(message)
        if not jailbreak_result.validation_passed:
            return _INPUT_BLOCKED
        explicit_shopping_request = _SHOPPING_INTENT.search(message) and _PRODUCT_TERM.search(message)
        if len(message.split()) >= _TOPIC_VALIDATION_MIN_WORDS and not explicit_shopping_request:
            topic_result = _topic_guard().validate(message)
            if not topic_result.validation_passed:
                return _INPUT_BLOCKED
    except Exception:
        return _INPUT_BLOCKED
    return None


def _validate_assistant_output(answer: str) -> str:
    try:
        result = _output_guard().validate(answer)
        if not result.validation_passed:
            return _OUTPUT_BLOCKED
    except Exception:
        return _OUTPUT_BLOCKED
    return answer


def enforce_response_guardrails(answer: str, tool_results: list[str]) -> str:
    answer = answer.replace("₹", "₱")
    if _CART_CHANGE_CLAIM.search(answer) and not any(
        _SUCCESSFUL_CART_TOOL.search(str(result)) for result in tool_results
    ):
        answer = "I couldn't verify that cart update, so I haven't changed anything."
    return _validate_assistant_output(answer)