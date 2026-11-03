import os
import re
from langchain.agents import create_agent
from langchain_openai import ChatOpenAI
from sqlalchemy import select
from .models import Cart, CartItem, Product
from .guardrails import enforce_response_guardrails, validate_user_input
from .llm_cache import configure_llm_cache
from .shopper_tools import build_tools

SYSTEM_PROMPT = """You are a helpful personal shopping assistant.
Use tools for all catalog facts, prices, stock, comparisons, and cart changes.
Never invent products, prices, availability, ratings, or policy terms.
Ask a clarifying question if a product request is too vague to search well.
Before adding/removing items, make sure the user's intent and quantity are clear.
For requests to replace or swap a cart item, inspect the cart and catalog as needed, then use replace_in_cart rather than separate add/remove calls. If the user specifies a replacement quantity, pass it; otherwise preserve the current quantity.
Never claim a cart change succeeded unless the tool confirms it.
For document questions, use ask_product_documents and cite the source filename in your answer.
When comparing, explain trade-offs and do not declare a universal winner; recommend based on the user's stated needs.
All catalog prices are in Philippine pesos. Always prefix every monetary amount with ₱. Never use ₹, INR, or any other currency symbol or code. Preserve the numeric price exactly as returned by the tools. Be concise and friendly.
"""

def _replacement_context(message: str, history: list | None) -> str | None:
    direct_request = re.match(r"^\s*(?:yes\b[\s,!.]*)?(?:please\s+)?(?:replace|swap)\b", message, re.I)
    if direct_request:
        return message
    if not re.match(r"^\s*(?:yes|yeah|yep|please do|do it|confirmed)\b", message, re.I):
        return None
    if not history or history[-1].get("role") != "assistant":
        return None
    previous_answer = history[-1].get("content", "")
    if re.search(r"\b(replace|swap)\b", previous_answer, re.I) and re.search(
        r"\?|confirm", previous_answer, re.I
    ):
        return f"{previous_answer}\n{message}"
    return None


def _mentioned_products(text: str, products: list[Product]) -> list[Product]:
    lowered = text.casefold()
    return [product for product in products if
            product.name.casefold() in lowered or
            (product.brand and product.brand.casefold() in lowered)]


def _resolve_replacement(db, user_id: str, text: str, tools: list) -> str | None:
    match = re.search(
        r"\b(?:replace|swap)\s+(?:all\s+)?(?P<source>.+?)\s+(?:with|for)\s+(?P<target>.+?)(?:[?.!,\n]|$)",
        text,
        re.I,
    )
    if not match:
        return None
    products = db.scalars(select(Product)).all()
    old_matches = _mentioned_products(match.group("source"), products)
    new_matches = _mentioned_products(match.group("target"), products)
    if len(old_matches) != 1 or len(new_matches) != 1 or old_matches[0].id == new_matches[0].id:
        return None

    cart = db.scalar(select(Cart).where(Cart.user_id == user_id, Cart.status == "active"))
    if not cart:
        return "There is no active cart to update."
    cart_items = db.scalars(select(CartItem).where(CartItem.cart_id == cart.id)).all()
    old_product = old_matches[0]
    if not any(item.product_id == old_product.id for item in cart_items):
        return f"{old_product.name} is not in the cart, so nothing was changed."
    quantity_match = re.search(r"\b(\d{1,3})\b", match.group("target"))
    arguments = {"old_product_id": old_product.id, "new_product_id": new_matches[0].id}
    if quantity_match:
        arguments["quantity"] = int(quantity_match.group(1))
    replace_tool = next(tool for tool in tools if tool.name == "replace_in_cart")
    return replace_tool.invoke(arguments)


def _resolve_add_request(db, message: str, tools: list) -> str | None:
    if not re.match(r"^\s*(?:please\s+)?add\b", message, re.I):
        return None
    lowered_message = message.casefold()
    products = db.scalars(select(Product)).all()
    matches = [product for product in products if product.name.casefold() in lowered_message]
    if len(matches) != 1:
        return None

    quantity_match = re.match(r"^\s*(?:please\s+)?add\s+(\d{1,3})\b", message, re.I)
    quantity = int(quantity_match.group(1)) if quantity_match else 1
    add_tool = next(tool for tool in tools if tool.name == "add_to_cart")
    return add_tool.invoke({"product_id": matches[0].id, "quantity": quantity})


def run_agent(db, user_id: str, message: str, history: list | None = None):
    blocked_message = validate_user_input(message)
    if blocked_message:
        return blocked_message

    tools = build_tools(db, user_id)
    add_result = _resolve_add_request(db, message, tools)
    if add_result is not None:
        return enforce_response_guardrails(add_result, [add_result])

    replacement_text = _replacement_context(message, history)
    if replacement_text:
        result = _resolve_replacement(db, user_id, replacement_text, tools)
        if result is not None:
            return enforce_response_guardrails(result, [result])
        tools = [tool for tool in tools if tool.name not in {
            "add_to_cart", "remove_from_cart", "replace_in_cart"
        }]

    configure_llm_cache()
    model = ChatOpenAI(model=os.getenv("OPENAI_MODEL", "gpt-4o-mini"), temperature=0)
    system_prompt = SYSTEM_PROMPT
    if replacement_text:
        system_prompt += "\nA replacement request is active but could not be resolved unambiguously. Ask a clarifying question and do not change the cart."
    agent = create_agent(model=model, tools=tools, system_prompt=system_prompt)
    messages = (history or []) + [{"role": "user", "content": message}]
    result = agent.invoke({"messages": messages})
    result_messages = result["messages"]
    answer = result_messages[-1].content
    tool_results = [item.content for item in result_messages if getattr(item, "type", None) == "tool"]
    return enforce_response_guardrails(answer, tool_results)
