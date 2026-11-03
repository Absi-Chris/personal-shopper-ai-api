import json
from langchain.tools import tool
from sqlalchemy import select, or_
from sqlalchemy.orm import Session
from .models import Product, Cart, CartItem
from .rag import retrieve_context

def build_tools(db: Session, user_id: str):
    def get_cart():
        cart = db.scalar(select(Cart).where(Cart.user_id == user_id, Cart.status == "active"))
        if not cart:
            cart = Cart(user_id=user_id)
            db.add(cart); db.commit(); db.refresh(cart)
        return cart

    @tool
    def search_products(query: str, max_price: float | None = None, category: str | None = None) -> str:
        """Search the product catalog by keyword, optional maximum price, and category."""
        stmt = select(Product)
        if query.strip():
            term = f"%{query.strip()}%"
            stmt = stmt.where(or_(Product.name.ilike(term), Product.brand.ilike(term),
                                  Product.description.ilike(term), Product.category.ilike(term)))
        if max_price is not None:
            stmt = stmt.where(Product.price <= max_price)
        if category:
            stmt = stmt.where(Product.category.ilike(f"%{category}%"))
        rows = db.scalars(stmt.limit(20)).all()
        return json.dumps([{"id":p.id,"sku":p.sku,"name":p.name,"brand":p.brand,
            "category":p.category,"price":p.price,"stock":p.stock,"rating":p.rating,
            "description":p.description,"attributes":json.loads(p.attributes_json or "{}")} for p in rows])

    @tool
    def compare_products(product_ids: list[int]) -> str:
        """Compare two or more products by price, rating, stock, description and attributes."""
        rows = db.scalars(select(Product).where(Product.id.in_(product_ids))).all()
        return json.dumps([{"id":p.id,"name":p.name,"brand":p.brand,"price":p.price,
            "rating":p.rating,"stock":p.stock,"description":p.description,
            "attributes":json.loads(p.attributes_json or "{}")} for p in rows])

    @tool
    def add_to_cart(product_id: int, quantity: int = 1) -> str:
        """Add a product to the current user's cart. Check stock before adding."""
        if quantity < 1 or quantity > 100:
            return "Quantity must be between 1 and 100."
        product = db.get(Product, product_id)
        if not product:
            return "Product not found."
        cart = get_cart()
        item = db.scalar(select(CartItem).where(CartItem.cart_id == cart.id, CartItem.product_id == product_id))
        current_qty = item.quantity if item else 0
        if product.stock < current_qty + quantity:
            return f"Insufficient stock. Available: {product.stock}; already in cart: {current_qty}."
        if item:
            item.quantity += quantity
        else:
            db.add(CartItem(cart_id=cart.id, product_id=product_id, quantity=quantity))
        db.commit()
        return f"Added {quantity} x {product.name} to cart."

    @tool
    def remove_from_cart(product_id: int) -> str:
        """Remove a product completely from the current user's cart."""
        cart = get_cart()
        item = db.scalar(select(CartItem).where(CartItem.cart_id == cart.id, CartItem.product_id == product_id))
        if not item:
            return "That product is not in the cart."
        db.delete(item); db.commit()
        return "Product removed from cart."

    @tool
    def replace_in_cart(old_product_id: int, new_product_id: int, quantity: int | None = None) -> str:
        """Replace a cart product with another; keep its quantity unless a new quantity is specified."""
        if old_product_id == new_product_id:
            return "The replacement product is already in the cart."
        cart = get_cart()
        old_item = db.scalar(select(CartItem).where(
            CartItem.cart_id == cart.id, CartItem.product_id == old_product_id
        ))
        if not old_item:
            return "The product to replace is not in the cart."
        old_product = db.get(Product, old_product_id)
        new_product = db.get(Product, new_product_id)
        if not old_product:
            return "The product to replace was not found."
        if not new_product:
            return "The replacement product was not found."
        replacement_quantity = old_item.quantity if quantity is None else quantity
        if replacement_quantity < 1 or replacement_quantity > 100:
            return "Quantity must be between 1 and 100."
        existing_new_item = db.scalar(select(CartItem).where(
            CartItem.cart_id == cart.id, CartItem.product_id == new_product_id
        ))
        resulting_quantity = replacement_quantity + (existing_new_item.quantity if existing_new_item else 0)
        if new_product.stock < resulting_quantity:
            return f"Insufficient stock for {new_product.name}. Available: {new_product.stock}."
        if existing_new_item:
            existing_new_item.quantity = resulting_quantity
        else:
            db.add(CartItem(cart_id=cart.id, product_id=new_product_id, quantity=replacement_quantity))
        db.delete(old_item)
        db.commit()
        return f"Done, I swapped {old_product.name} for {replacement_quantity} x {new_product.name} in your cart."

    @tool
    def view_cart() -> str:
        """Show the current user's cart, quantities, line totals and grand total."""
        cart = get_cart()
        items = db.scalars(select(CartItem).where(CartItem.cart_id == cart.id)).all()
        result = [{"product_id":i.product_id,"name":i.product.name,"quantity":i.quantity,
                   "unit_price":i.product.price,"line_total":round(i.quantity*i.product.price,2)} for i in items]
        return json.dumps({"items":result,"total":round(sum(x["line_total"] for x in result),2)})

    @tool
    def ask_product_documents(question: str) -> str:
        """Retrieve answers from uploaded product manuals, warranty policies and shopping documents."""
        return retrieve_context(question)

    return [search_products, compare_products, add_to_cart, remove_from_cart, replace_in_cart,
            view_cart, ask_product_documents]
