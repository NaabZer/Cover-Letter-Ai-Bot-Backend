from dotenv import load_dotenv
from google import genai
import os

load_dotenv()
gemeni_api_key = os.getenv("GEMENI_API_KEY")

client = genai.Client(api_key=gemeni_api_key)

response = client.models.generate_content_stream(
    model="gemini-2.0-flash", contents="Explain how AI works"
)
for chunk in response:
    print(chunk.text, end="")
