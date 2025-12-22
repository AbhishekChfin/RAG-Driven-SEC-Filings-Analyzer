from fastapi import FastAPI
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer
import uvicorn

app = FastAPI()

# Load model once at startup
model = SentenceTransformer('all-mpnet-base-v2')

class EmbedRequest(BaseModel):
    text: str

@app.post("/embed")
async def embed(request: EmbedRequest):
    if not request.text.strip():
        return {"error": "Empty text"}
    embedding = model.encode(request.text).tolist()
    return {"embedding": embedding}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=5001)
