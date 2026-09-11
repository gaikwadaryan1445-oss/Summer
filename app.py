import gradio as gr
import edge_tts
from faster_whisper import WhisperModel
import asyncio
import tempfile
import os
import time
import json
import requests
from datetime import datetime
from groq import Groq
import chromadb
from sentence_transformers import SentenceTransformer

# --- Load Whisper (Speech-to-Text) ---
print("Loading Whisper...")
whisper_model = WhisperModel("tiny.en", device="cpu", compute_type="int8")
print("Whisper ready.")

# --- Load Memory (ChromaDB) ---
print("Loading memory...")
embedder = SentenceTransformer('all-MiniLM-L6-v2')
chroma_client = chromadb.Client()
memory = chroma_client.get_or_create_collection("summer_memory")

def save_memory(text):
    embedding = embedder.encode(text).tolist()
    memory.add(embeddings=[embedding], documents=[text], ids=[str(time.time())])

def recall_memory(query, n_results=2):
    if memory.count() == 0:
        return ""
    query_embedding = embedder.encode(query).tolist()
    results = memory.query(query_embeddings=[query_embedding], n_results=n_results)
    docs = results['documents'][0] if results['documents'] else []
    return "\n".join(docs)

print("Memory ready.")

# --- Load Groq (The Brain) ---
# The API key is pulled from Environment Variables (set this in Render)
client = Groq(api_key=os.environ.get("GROQ_API_KEY"))

# --- Define Tools ---
def get_current_time():
    return datetime.now().strftime("It is %I:%M %p on %A, %B %d, %Y.")

def search_web(query):
    try:
        url = "https://api.duckduckgo.com/"
        params = {"q": query, "format": "json", "no_html": 1}
        r = requests.get(url, params=params, timeout=5).json()
        if r.get("AbstractText"): 
            return r["AbstractText"]
        if r.get("RelatedTopics"): 
            return r["RelatedTopics"][0].get("Text", "No good results found.")
        return "I couldn't find a clear answer on the web."
    except Exception as e:
        return f"Search failed: {e}"

def check_sales():
    return "Today's sales: 14 orders, $342 in revenue. Best seller: Notion Productivity Template."

def update_product_price(product_name: str, new_price: float):
    return f"Successfully updated {product_name} to ${new_price}."

tools = [
    {"type": "function", "function": {"name": "get_current_time", "description": "Get the current time and date", "parameters": {"type": "object", "properties": {}}}},
    {"type": "function", "function": {"name": "search_web", "description": "Search the web for current information", "parameters": {"type": "object", "properties": {"query": {"type": "string", "description": "The search query"}}, "required": ["query"]}}},
    {"type": "function", "function": {"name": "check_sales", "description": "Check today's digital product sales, revenue, and best sellers", "parameters": {"type": "object", "properties": {}}}},
    {"type": "function", "function": {"name": "update_product_price", "description": "Update the price of a digital product", "parameters": {"type": "object", "properties": {"product_name": {"type": "string", "description": "Name of the product"}, "new_price": {"type": "number", "description": "New price in USD"}}, "required": ["product_name", "new_price"]}}}
]

# --- The Brain Function ---
def llm_reply(user_text, history=[]):
    past_context = recall_memory(user_text)
    system_prompt = "You are Summer, a calm, efficient, slightly witty personal AI assistant. Keep your answers concise and conversational. Do not use markdown, emojis, or bullet points. Speak naturally."
    
    if past_context:
        system_prompt += f"\n\nRelevant things you remember about the user:\n{past_context}"
        
    messages = [{"role": "system", "content": system_prompt}]
    messages.extend(history) 
    messages.append({"role": "user", "content": user_text})

    response = client.chat.completions.create(model="llama-3.1-8b-instant", messages=messages, tools=tools, tool_choice="auto")
    response_message = response.choices[0].message
    tool_calls = response_message.tool_calls

    if tool_calls:
        messages.append(response_message)
        for tool_call in tool_calls:
            function_name = tool_call.function.name
            if function_name == "get_current_time":
                function_response = get_current_time()
            elif function_name == "search_web":
                args = json.loads(tool_call.function.arguments)
                function_response = search_web(args["query"])
            elif function_name == "check_sales":
                function_response = check_sales()
            elif function_name == "update_product_price":
                args = json.loads(tool_call.function.arguments)
                function_response = update_product_price(args["product_name"], args["new_price"])
            else:
                function_response = "Unknown tool."
                
            messages.append({"tool_call_id": tool_call.id, "role": "tool", "name": function_name, "content": function_response})
            
        second_response = client.chat.completions.create(model="llama-3.1-8b-instant", messages=messages)
        reply = second_response.choices[0].message.content
    else:
        reply = response_message.content

    save_memory(f"User: {user_text} | Summer: {reply}")
    return reply

# --- Voice Loop ---
chat_history = []

async def _speak_async(text, path):
    communicate = edge_tts.Communicate(text, "en-US-AriaNeural")
    await communicate.save(path)

def process_audio(audio_path):
    global chat_history
    if audio_path is None:
        return "No audio received.", None

    segments, _ = whisper_model.transcribe(audio_path, language="en", vad_filter=True)
    user_text = " ".join([seg.text for seg in segments]).strip()

    if not user_text:
        return "I didn't catch that.", None

    try:
        reply_text = llm_reply(user_text, chat_history)
    except Exception as e:
        reply_text = f"Sorry, my brain hit an error: {e}"

    chat_history.append({"role": "user", "content": user_text})
    chat_history.append({"role": "assistant", "content": reply_text})
    chat_history = chat_history[-6:]

    with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as f:
        mp3_path = f.name
    asyncio.run(_speak_async(reply_text, mp3_path))
    return user_text, mp3_path

# --- Gradio UI ---
iface = gr.Interface(
    fn=process_audio,
    inputs=gr.Audio(sources=["microphone"], type="filepath", label="Speak now"),
    outputs=[gr.Textbox(label="You said"), gr.Audio(label="Summer replies", autoplay=True)],
    title="Summer",
    description="Your personal AI assistant. Click the mic, speak, and Summer will reply."
)

# --- Launch (Configured for Render) ---
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 7860))
    iface.launch(server_name="0.0.0.0", server_port=port)
