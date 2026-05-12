from fastapi import FastAPI

app = FastAPI()

# @app.get() creates a route
# A route means that when that route is entered, run this function
@app.get("/")
def home():
    return {"message": "Cat or Not Cat API is running"}

@app.get("/health")
def health():
    return {"status": "ok"}

