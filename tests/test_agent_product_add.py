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


if __name__ == "__main__":
    unittest.main()