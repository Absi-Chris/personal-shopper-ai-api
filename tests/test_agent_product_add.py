import unittest
from unittest.mock import patch

from sqlalchemy import create_engine, select
from sqlalchemy.orm import Session

from app.database import Base
from app.models import Cart, CartItem, Product
from app.agent import run_agent


class ExplicitProductAddTests(unittest.TestCase):
    def setUp(self):
        self.engine = create_engine("sqlite:///:memory:")
        Base.metadata.create_all(self.engine)
        self.db = Session(self.engine)
        self.db.add_all([
            Product(
                id=1, sku="HP-001", name="JBL Tune 760NC", brand="JBL",
                category="Headphones", price=2799, stock=15,
            ),
            Product(
                id=5, sku="HP-004", name="JBL Tune 520BT", brand="JBL",
                category="Headphones", price=2499, stock=18,
            ),
            Cart(user_id="test-user"),
        ])
        self.db.commit()

    def tearDown(self):
        self.db.close()
        self.engine.dispose()

    def test_explicit_product_name_adds_exact_matching_product(self):
        with patch("app.agent.validate_user_input", return_value=None):
            with patch("app.agent.enforce_response_guardrails", side_effect=lambda answer, _: answer):
                answer = run_agent(self.db, "test-user", "add JBL Tune 520BT to cart")

        cart = self.db.scalar(select(Cart).where(Cart.user_id == "test-user"))
        items = self.db.scalars(select(CartItem).where(CartItem.cart_id == cart.id)).all()

        self.assertIn("JBL Tune 520BT", answer)
        self.assertEqual([(item.product_id, item.quantity) for item in items], [(5, 1)])

    def test_explicit_add_supports_requested_quantity(self):
        with patch("app.agent.validate_user_input", return_value=None):
            with patch("app.agent.enforce_response_guardrails", side_effect=lambda answer, _: answer):
                run_agent(self.db, "test-user", "add 2 JBL Tune 520BT to cart")

        cart = self.db.scalar(select(Cart).where(Cart.user_id == "test-user"))
        item = self.db.scalar(select(CartItem).where(
            CartItem.cart_id == cart.id, CartItem.product_id == 5
        ))
        self.assertIsNotNone(item)
        self.assertEqual(item.quantity, 2)

    def test_explicit_remove_all_items_in_cart_clears_cart(self):
        cart = self.db.scalar(select(Cart).where(Cart.user_id == "test-user"))
        self.db.add_all([
            CartItem(cart_id=cart.id, product_id=1, quantity=1),
            CartItem(cart_id=cart.id, product_id=5, quantity=2),
        ])
        self.db.commit()

        with patch("app.agent.validate_user_input", return_value=None):
            with patch("app.agent.enforce_response_guardrails", side_effect=lambda answer, _: answer):
                answer = run_agent(self.db, "test-user", "remove all items in the cart")

        items = self.db.scalars(select(CartItem).where(CartItem.cart_id == cart.id)).all()
        self.assertIn("Removed all items", answer)
        self.assertEqual(items, [])

    def test_yes_confirms_remove_all_cart_request(self):
        cart = self.db.scalar(select(Cart).where(Cart.user_id == "test-user"))
        self.db.add(CartItem(cart_id=cart.id, product_id=5, quantity=1))
        self.db.commit()

        history = [{
            "role": "assistant",
            "content": "Would you like to proceed with removing all items from the cart?",
        }]
        with patch("app.agent.validate_user_input", return_value=None):
            with patch("app.agent.enforce_response_guardrails", side_effect=lambda answer, _: answer):
                answer = run_agent(self.db, "test-user", "yes", history=history)

        items = self.db.scalars(select(CartItem).where(CartItem.cart_id == cart.id)).all()
        self.assertIn("Removed all items", answer)
        self.assertEqual(items, [])

    def test_unrelated_yes_does_not_clear_cart(self):
        cart = self.db.scalar(select(Cart).where(Cart.user_id == "test-user"))
        self.db.add(CartItem(cart_id=cart.id, product_id=5, quantity=1))
        self.db.commit()

        history = [{"role": "assistant", "content": "Would you like product recommendations?"}]
        with patch("app.agent.validate_user_input", return_value=None):
            with patch("app.agent.configure_llm_cache"):
                with patch("app.agent.ChatOpenAI"):
                    with patch("app.agent.create_agent") as create_agent:
                        create_agent.return_value.invoke.return_value = {
                            "messages": [type("Message", (), {"content": "Sure", "type": "ai"})()]
                        }
                        with patch("app.agent.enforce_response_guardrails", side_effect=lambda answer, _: answer):
                            run_agent(self.db, "test-user", "yes", history=history)

        item = self.db.scalar(select(CartItem).where(
            CartItem.cart_id == cart.id, CartItem.product_id == 5
        ))
        self.assertIsNotNone(item)

    def test_yes_confirms_product_replacement_without_llm_fallback(self):
        cart = self.db.scalar(select(Cart).where(Cart.user_id == "test-user"))
        self.db.add(CartItem(cart_id=cart.id, product_id=5, quantity=1))
        self.db.commit()
        history = [{
            "role": "assistant",
            "content": (
                "To replace the JBL Tune 520BT in your cart with 1 JBL Tune 760NC, "
                "please confirm if you would like to proceed with this replacement. "
                "The new item will be priced at ₱2,799. Would you like to continue?"
            ),
        }]

        with patch("app.agent.validate_user_input", return_value=None):
            with patch("app.agent.enforce_response_guardrails", side_effect=lambda answer, _: answer):
                with patch("app.agent.create_agent") as create_agent:
                    answer = run_agent(self.db, "test-user", "yes", history=history)

        items = self.db.scalars(select(CartItem).where(CartItem.cart_id == cart.id)).all()
        self.assertIn("swapped JBL Tune 520BT", answer)
        self.assertEqual([(item.product_id, item.quantity) for item in items], [(1, 1)])
        create_agent.assert_not_called()

    def test_yes_confirms_replacing_wording_and_uses_original_quantity(self):
        cart = self.db.scalar(select(Cart).where(Cart.user_id == "test-user"))
        self.db.add(CartItem(cart_id=cart.id, product_id=5, quantity=2))
        self.db.commit()
        history = [
            {"role": "user", "content": "replace item in the cart with 1 JBL Tune 760NC"},
            {
                "role": "assistant",
                "content": (
                    "Would you like to proceed with replacing the JBL Tune 520BT "
                    "with the JBL Tune 760NC? Please confirm and I'll make the change."
                ),
            },
        ]

        with patch("app.agent.validate_user_input", return_value=None):
            with patch("app.agent.enforce_response_guardrails", side_effect=lambda answer, _: answer):
                with patch("app.agent.create_agent") as create_agent:
                    answer = run_agent(self.db, "test-user", "yes", history=history)

        items = self.db.scalars(select(CartItem).where(CartItem.cart_id == cart.id)).all()
        self.assertIn("swapped JBL Tune 520BT", answer)
        self.assertEqual([(item.product_id, item.quantity) for item in items], [(1, 1)])
        create_agent.assert_not_called()

    def test_proceed_with_quantity_confirms_pending_replacement(self):
        cart = self.db.scalar(select(Cart).where(Cart.user_id == "test-user"))
        self.db.add(CartItem(cart_id=cart.id, product_id=5, quantity=2))
        self.db.commit()
        history = [
            {"role": "user", "content": "replace item in the cart with JBL Tune 760NC"},
            {
                "role": "assistant",
                "content": (
                    "Would you like to proceed with replacing the JBL Tune 520BT "
                    "with the JBL Tune 760NC? If so, please confirm the quantity."
                ),
            },
        ]

        with patch("app.agent.validate_user_input", return_value=None):
            with patch("app.agent.enforce_response_guardrails", side_effect=lambda answer, _: answer):
                with patch("app.agent.create_agent") as create_agent:
                    answer = run_agent(
                        self.db, "test-user", "proceed with 1 qty", history=history
                    )

        items = self.db.scalars(select(CartItem).where(CartItem.cart_id == cart.id)).all()
        self.assertIn("swapped JBL Tune 520BT", answer)
        self.assertEqual([(item.product_id, item.quantity) for item in items], [(1, 1)])
        create_agent.assert_not_called()


if __name__ == "__main__":
    unittest.main()