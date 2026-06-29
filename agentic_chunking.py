from langchain_ollama import OllamaLLM
from dotenv import load_dotenv

load_dotenv()

# Try tinyllama first, or upgrade to llama3.2:3b for better results
llm = OllamaLLM(model="tinyllama")

tesla_text = """Tesla's Q3 Results
Tesla reported record revenue of $25.2B in Q3 2024.
The company exceeded analyst expectations by 15%.
Revenue growth was driven by strong vehicle deliveries.

Model Y Performance  
The Model Y became the best-selling vehicle globally, with 350,000 units sold.
Customer satisfaction ratings reached an all-time high of 96%.
Model Y now represents 60% of Tesla's total vehicle sales.

Production Challenges
Supply chain issues caused a 12% increase in production costs.
Tesla is working to diversify its supplier base.
New manufacturing techniques are being implemented to reduce costs."""

# SIMPLER prompt — small models need this!
prompt = f"""Split this text into chunks. Put <<<SPLIT>>> between each chunk.

Text:
{tesla_text}

Output with <<<SPLIT>>> markers:"""

print("🤖 Asking AI to chunk the text...")
response = llm.invoke(prompt)
marked_text = response

chunks = marked_text.split("<<<SPLIT>>>")

clean_chunks = []
for chunk in chunks:
    cleaned = chunk.strip()
    if cleaned and len(cleaned) > 10:  # Filter out junk
        clean_chunks.append(cleaned)

print(f"\n🎯 AGENTIC CHUNKING RESULTS ({len(clean_chunks)} chunks):")
print("=" * 50)

for i, chunk in enumerate(clean_chunks, 1):
    print(f"Chunk {i}: ({len(chunk)} chars)")
    print(f'"{chunk}"')
    print()