from fastapi import FastAPI
from pydantic import BaseModel

#  FastAPI instance
app = FastAPI(title="Item Price Checker")

#  Pydantic model for input validation
class Item(BaseModel):
    name: str
    price: float
    quantity: int

#  Root endpoint
@app.get("/")
def root():
    return {"Xabar": "Welcome to Item Price Checker API!"}

#  POST endpoint to calculate total price
@app.post("/calculate/")
def calculate_total(item: Item):
    total = item.price * item.quantity
    return {
        "item_name": item.name,
        "quantity": item.quantity,
        "unit_price": item.price,
        "total_price": total,
        "message": f"Total price for {item.quantity} {item.name}(s) is {total}"
    }
