from urllib import response
from fastapi import FastAPI
from serpapi import GoogleSearch
from dotenv import load_dotenv
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from openai import OpenAI
from fastapi.responses import StreamingResponse
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
client = OpenAI(api_key=api_key)

llm = ChatOpenAI(
    api_key=api_key,
    model="gpt-3.5-turbo",
    temperature=0.7
)
def search_laptops(query):
    params = {
    "engine": "google_shopping",
    "q": query,
    "location": "United States",
    "hl": "en",
    "gl": "us",
    "api_key": os.getenv("SERPAPI_API_KEY")
    }
    search = GoogleSearch(params)
    results = search.get_dict()
    print(results)

    shopping_results = results.get("shopping_results", [])

    laptops = []

    for item in shopping_results[:5]:
        laptops.append({
            "title": item.get("title"),
            "price": item.get("price"),
            "rating": item.get("rating"),
            "link": item.get("product_link"),
            "thumbnail": item.get("thumbnail")
        })

    return laptops

def generate_stream(prompt):

    stream = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[
            {
                "role": "user",
                "content": prompt
            }
        ],
        stream=True
    )

    for chunk in stream:
        content = chunk.choices[0].delta.content

        if content:
            yield content

class ChatRequest(BaseModel):
    message: str

@app.get("/")
def home():
    return {"message": "Laptop AI API is running"}

@app.post("/chat")
def chat(req: ChatRequest):
    user_input = req.message.lower()
    retrieved_laptops = search_laptops(user_input)

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
    context = "\n\n".join([
    f"""
     ### {l['title']}

     - Price: {l['price']}
     - Rating: {l['rating']}

     <a href="{l['link']}" target="_blank">Buy Here</a>
    """
    for l in retrieved_laptops
   ])
    if not context:
        context = "No live laptop data found."


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
- Always provide clickable markdown links.

Provide:
- best recommendations
- concise explanations
- clickable markdown links

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

Use the retrieved laptop data below while answering.

Retrieved Laptop Data:
{context}

{req.message}
"""

    response = llm.invoke(prompt).content

    return {
    "response": response,
    "products": retrieved_laptops
}

@app.post("/stream-chat")
def stream_chat(req: ChatRequest):

    prompt = f"""
    You are LaptopAI.

    User Question:
    {req.message}
    """

    return StreamingResponse(
        generate_stream(prompt),
        media_type="text/plain"
    )