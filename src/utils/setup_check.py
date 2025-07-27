from dotenv import load_dotenv
import llama_index.core as lic
from llama_index.llms.openai import OpenAI
import os


load_dotenv()
print(f"llama_index version: {lic.__version__}")

api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    print("OPENAI_API_KEY not found in .env file")
    exit(1)

llm = OpenAI(api_key=api_key, model="gpt-4o")
print(f"LLM Response:{llm.complete('What is the capital of France?').text}")