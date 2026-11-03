from app.database import Base, engine, SessionLocal
from app.models import Product
import json

Base.metadata.create_all(bind=engine)
products = [
    dict(sku="HP-001",name="JBL Tune 760NC",brand="JBL",category="Headphones",
         description="Wireless over-ear headphones with active noise cancelling and long battery life.",
         price=2799,stock=15,rating=4.5,attributes={"connectivity":"Bluetooth","noise_cancelling":True,"battery_hours":35}),
    dict(sku="HP-002",name="Sony WH-CH720N",brand="Sony",category="Headphones",
         description="Lightweight wireless noise cancelling headphones with multipoint connection.",
         price=4999,stock=8,rating=4.7,attributes={"connectivity":"Bluetooth","noise_cancelling":True,"battery_hours":35}),
    dict(sku="HP-003",name="Anker Soundcore Q20i",brand="Anker",category="Headphones",
         description="Budget hybrid active noise cancelling headphones with customizable EQ.",
         price=1999,stock=20,rating=4.4,attributes={"connectivity":"Bluetooth","noise_cancelling":True,"battery_hours":40}),
    dict(sku="LT-001",name="Lenovo IdeaPad Slim 5",brand="Lenovo",category="Laptops",
         description="Everyday laptop suitable for productivity and software development.",
         price=42999,stock=5,rating=4.6,attributes={"ram_gb":16,"storage_gb":512,"cpu":"Ryzen 7"}),
    dict(sku="HP-004",name="JBL Tune 520BT",brand="JBL",category="Headphones",
         description="Lightweight on-ear wireless headphones for everyday listening.",
         price=2499,stock=18,rating=4.3,attributes={"connectivity":"Bluetooth","noise_cancelling":False,"battery_hours":57}),
    dict(sku="HP-005",name="Soundcore Space One",brand="Anker",category="Headphones",
         description="Over-ear wireless headphones with adaptive noise cancelling and app-based EQ.",
         price=4499,stock=12,rating=4.5,attributes={"connectivity":"Bluetooth","noise_cancelling":True,"battery_hours":40}),
    dict(sku="LT-002",name="ASUS Vivobook 16",brand="ASUS",category="Laptops",
         description="Large-screen laptop for school, office work and everyday multitasking.",
         price=38999,stock=7,rating=4.4,attributes={"ram_gb":16,"storage_gb":512,"cpu":"Intel Core i5"}),
    dict(sku="LT-003",name="Acer Aspire 3",brand="Acer",category="Laptops",
         description="Affordable everyday laptop for browsing, documents and online classes.",
         price=27999,stock=9,rating=4.2,attributes={"ram_gb":8,"storage_gb":512,"cpu":"Ryzen 5"}),
    dict(sku="PH-001",name="Samsung Galaxy A55",brand="Samsung",category="Smartphones",
         description="Midrange Android smartphone with a large display and multi-camera system.",
         price=23999,stock=10,rating=4.5,attributes={"storage_gb":256,"connectivity":"5G","operating_system":"Android"}),
    dict(sku="PH-002",name="Apple iPhone 15",brand="Apple",category="Smartphones",
         description="Apple smartphone with a dual-camera system and USB-C connectivity.",
         price=45990,stock=6,rating=4.7,attributes={"storage_gb":128,"connectivity":"5G","operating_system":"iOS"}),
    dict(sku="MN-001",name="LG UltraGear 24GN60R",brand="LG",category="Monitors",
         description="24-inch Full HD gaming monitor with a high refresh rate.",
         price=8999,stock=8,rating=4.6,attributes={"size_inches":24,"resolution":"1920x1080","refresh_rate_hz":144}),
    dict(sku="MN-002",name="Dell S2722QC",brand="Dell",category="Monitors",
         description="27-inch 4K monitor with USB-C connectivity for work and entertainment.",
         price=24999,stock=4,rating=4.5,attributes={"size_inches":27,"resolution":"3840x2160","connectivity":"USB-C"}),
    dict(sku="KB-001",name="Logitech Pebble Keys 2 K380s",brand="Logitech",category="Keyboards",
         description="Compact wireless keyboard that can pair with multiple devices.",
         price=2295,stock=14,rating=4.4,attributes={"connectivity":"Bluetooth","layout":"Compact","device_switching":True}),
    dict(sku="WT-001",name="Garmin Forerunner 165",brand="Garmin",category="Smartwatches",
         description="GPS running smartwatch with workout tracking and a color display.",
         price=15990,stock=5,rating=4.6,attributes={"gps":True,"water_resistance":"5 ATM","display":"AMOLED"}),
]
with SessionLocal() as db:
    for item in products:
        if not db.query(Product).filter_by(sku=item["sku"]).first():
            attrs = item.pop("attributes")
            db.add(Product(**item, attributes_json=json.dumps(attrs)))
    db.commit()
print("Seeded sample catalog.")
