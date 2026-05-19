from fastapi import FastAPI
from dotenv import load_dotenv
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from langchain_openai import ChatOpenAI
import os

load_dotenv(override=True)

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

api_key = os.getenv("OPENAI_API_KEY")
print(api_key)

llm = ChatOpenAI(
    api_key=api_key,
    model="gpt-3.5-turbo",
    temperature=0.7
)

class ChatRequest(BaseModel):
    message: str

@app.get("/")
def home():
    return {"message": "Laptop AI API is running"}

@app.post("/chat")
def chat(req: ChatRequest):
    user_input = req.message.lower()

    laptop_keywords = [
        "laptop", "macbook", "notebook", "ultrabook",
        "dell", "hp", "lenovo", "asus", "acer",
        "msi", "razer", "surface", "thinkpad",
        "ram", "ssd", "cpu", "gpu", "processor",
        "battery", "display", "screen", "gaming",
        "coding", "student", "business"
    ]

    if not any(word in user_input for word in laptop_keywords):
        return {
            "response": "I can only help with laptop-related questions."
        }

    prompt = f"""
You are LaptopAI, a premium laptop recommendation assistant.

Respond ONLY in clean markdown format.

VERY IMPORTANT FORMATTING RULES:
- Use proper markdown headings.
- Add blank lines between sections.
- Never compress everything into one paragraph.
- Never put multiple headings on the same line.
- Every heading MUST start on a new line.
- Use bullet points for specs.
- Keep responses visually clean and easy to read.

Example format:

# Best Gaming Laptop

## ASUS ROG Zephyrus G14

### Specs
- CPU: Ryzen 9
- GPU: RTX 4080
- RAM: 32GB
- Storage: 1TB SSD

### Pros
- Excellent gaming performance
- Portable design

### Cons
- Expensive

### Price
$2500-$3000

Now answer the user's question.

User Question:
{req.message}
"""

    response = llm.invoke(prompt).content

    return {"response": response}