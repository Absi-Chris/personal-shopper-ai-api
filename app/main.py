import json, os
from pathlib import Path
from uuid import uuid4
from fastapi import FastAPI, Depends, HTTPException, UploadFile, File, Header
from sqlalchemy.orm import Session
from sqlalchemy import select
from dotenv import load_dotenv
from pypdf import PdfReader
from .database import Base, engine, get_db
from .models import Product, Cart, CartItem, ChatMessage
from .schemas import ChatRequest, ProductCreate, CartAdd
from .agent import run_agent
from .rag import delete_document_vectors, ingest_text

load_dotenv()
DOCUMENTS_DIR = Path(os.getenv("DOCUMENTS_DIR", "./documents"))
Base.metadata.create_all(bind=engine)
app = FastAPI(title="Personal Shopper AI API", version="1.0.0")

def current_user(x_user_id: str | None = Header(default="demo-user")):
    # Demo identity only. Replace with verified JWT/OAuth identity before production.
    return x_user_id or "demo-user"

@app.get("/health")
def health():
    return {"status":"ok"}

@app.post("/products")
def create_product(payload: ProductCreate, db: Session = Depends(get_db)):
    if db.scalar(select(Product).where(Product.sku == payload.sku)):
        raise HTTPException(409, "SKU already exists")
    p = Product(**payload.model_dump(exclude={"attributes"}),
                attributes_json=json.dumps(payload.attributes))
    db.add(p); db.commit(); db.refresh(p)
    return {"id":p.id,"sku":p.sku,"name":p.name}

@app.get("/products")
def list_products(q: str = "", category: str | None = None, max_price: float | None = None,
                  db: Session = Depends(get_db)):
    stmt = select(Product)
    if q:
        from sqlalchemy import or_
        like = f"%{q}%"
        stmt = stmt.where(or_(Product.name.ilike(like), Product.brand.ilike(like),
                              Product.description.ilike(like)))
    if category: stmt = stmt.where(Product.category.ilike(f"%{category}%"))
    if max_price is not None: stmt = stmt.where(Product.price <= max_price)
    return [{"id":p.id,"sku":p.sku,"name":p.name,"brand":p.brand,"category":p.category,
             "price":p.price,"stock":p.stock,"rating":p.rating} for p in db.scalars(stmt.limit(100)).all()]

@app.post("/cart/items")
def api_add_cart_item(payload: CartAdd, user_id: str = Depends(current_user), db: Session = Depends(get_db)):
    product = db.get(Product, payload.product_id)
    if not product: raise HTTPException(404, "Product not found")
    cart = db.scalar(select(Cart).where(Cart.user_id == user_id, Cart.status == "active"))
    if not cart:
        cart = Cart(user_id=user_id); db.add(cart); db.commit(); db.refresh(cart)
    item = db.scalar(select(CartItem).where(CartItem.cart_id == cart.id, CartItem.product_id == product.id))
    qty = (item.quantity if item else 0) + payload.quantity
    if qty > product.stock: raise HTTPException(409, f"Insufficient stock; available {product.stock}")
    if item: item.quantity = qty
    else: db.add(CartItem(cart_id=cart.id, product_id=product.id, quantity=payload.quantity))
    db.commit()
    return {"message":"Added to cart","cart_id":cart.id}

@app.get("/cart")
def get_cart(user_id: str = Depends(current_user), db: Session = Depends(get_db)):
    cart = db.scalar(select(Cart).where(Cart.user_id == user_id, Cart.status == "active"))
    if not cart: return {"items":[],"total":0}
    items = db.scalars(select(CartItem).where(CartItem.cart_id == cart.id)).all()
    lines = [{"product_id":i.product_id,"name":i.product.name,"quantity":i.quantity,
              "unit_price":i.product.price,"line_total":round(i.quantity*i.product.price,2)} for i in items]
    return {"items":lines,"total":round(sum(x["line_total"] for x in lines),2)}

@app.delete("/cart/items/{product_id}")
def delete_cart_item(product_id: int, user_id: str = Depends(current_user), db: Session = Depends(get_db)):
    cart = db.scalar(select(Cart).where(Cart.user_id == user_id, Cart.status == "active"))
    if not cart: raise HTTPException(404, "Cart not found")
    item = db.scalar(select(CartItem).where(CartItem.cart_id == cart.id, CartItem.product_id == product_id))
    if not item: raise HTTPException(404, "Cart item not found")
    db.delete(item); db.commit()
    return {"message":"Removed from cart"}

@app.post("/chat")
def chat(payload: ChatRequest, user_id: str = Depends(current_user), db: Session = Depends(get_db)):
    session_id = payload.session_id or str(uuid4())
    try:
        history_rows = db.scalars(
            select(ChatMessage).where(
                ChatMessage.session_id == session_id,
                ChatMessage.user_id == user_id
            ).order_by(ChatMessage.created_at.desc()).limit(12)
        ).all()
        history = [{"role": row.role, "content": row.content} for row in reversed(history_rows)]
        answer = run_agent(db, user_id, payload.message, history=history)
        db.add(ChatMessage(session_id=session_id, user_id=user_id,
                           role="user", content=payload.message))
        db.add(ChatMessage(session_id=session_id, user_id=user_id,
                           role="assistant", content=str(answer)))
        db.commit()
        return {"session_id":session_id,"answer":answer}
    except Exception as exc:
        db.rollback()
        raise HTTPException(500, f"Assistant error: {type(exc).__name__}")

@app.get("/documents")
def list_documents():
    if not DOCUMENTS_DIR.is_dir():
        return {"documents": []}
    documents = []
    for path in sorted(DOCUMENTS_DIR.iterdir(), key=lambda item: item.name.casefold()):
        if path.is_file() and path.suffix.lower() in {".txt", ".md", ".pdf"}:
            stored_as, _, original_name = path.name.partition("_")
            documents.append({
                "stored_as": path.name,
                "filename": original_name or path.name,
                "size_bytes": path.stat().st_size,
            })
    return {"documents": documents}

@app.delete("/documents/{stored_as}")
def delete_document(stored_as: str):
    if Path(stored_as).name != stored_as or stored_as in {".", ".."}:
        raise HTTPException(400, "Invalid document name")
    stored_path = DOCUMENTS_DIR / stored_as
    if not stored_path.is_file() or stored_path.suffix.lower() not in {".txt", ".md", ".pdf"}:
        raise HTTPException(404, "Document not found")
    try:
        delete_document_vectors(stored_as)
    except Exception as exc:
        raise HTTPException(500, f"Could not remove document from search index: {type(exc).__name__}")
    stored_path.unlink()
    return {"message": "Document deleted", "stored_as": stored_as}

@app.post("/documents")
async def upload_document(file: UploadFile = File(...)):
    suffix = Path(file.filename or "").suffix.lower()
    if suffix not in {".txt", ".md", ".pdf"}:
        raise HTTPException(400, "Only TXT, MD and PDF files are supported")
    raw = await file.read()
    if len(raw) > 10 * 1024 * 1024:
        raise HTTPException(413, "Maximum upload size is 10 MB")
    try:
        if suffix == ".pdf":
            import io
            text = "\n".join(page.extract_text() or "" for page in PdfReader(io.BytesIO(raw)).pages)
        else:
            text = raw.decode("utf-8")
        if not text.strip(): raise ValueError("No extractable text")
        safe_name = Path((file.filename or "upload").replace("\\", "/")).name
        stored_path = DOCUMENTS_DIR / f"{uuid4().hex}_{safe_name}"
        DOCUMENTS_DIR.mkdir(parents=True, exist_ok=True)
        stored_path.write_bytes(raw)
        try:
            count = ingest_text(text, stored_path.name)
        except Exception:
            stored_path.unlink(missing_ok=True)
            raise
        return {"filename":file.filename,"stored_as":stored_path.name,"chunks_indexed":count}
    except Exception as exc:
        raise HTTPException(400, f"Could not process document: {type(exc).__name__}")
