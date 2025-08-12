# import faiss
# import pickle
# from flask import Flask, request, jsonify
# from sentence_transformers import SentenceTransformer
# import torch
# from transformers import pipeline

# # --- Configuration (remains the same) ---
# INDEX_FILE = "faiss_index.bin"
# CHUNKS_FILE = "text_chunks.pkl"
# RETRIEVER_MODEL = 'all-MiniLM-L6-v2'
# GENERATOR_MODEL = "microsoft/Phi-3-mini-4k-instruct"

# # --- 1. Initialize Flask and Models ---
# app = Flask(__name__)
# print("Loading retriever model...")
# retriever = SentenceTransformer(RETRIEVER_MODEL)

# print("Loading knowledge base...")
# index = faiss.read_index(INDEX_FILE)
# with open(CHUNKS_FILE, 'rb') as f:
#     chunks = pickle.load(f)

# print("Loading generator model (this may take a few moments)...")
# generator = pipeline(
#     "text-generation",
#     model=GENERATOR_MODEL,
#     model_kwargs={"torch_dtype": "float32"},
#     trust_remote_code=True,
# )
# print("All models loaded. Ready to answer questions.")


# # --- 2. Create the API Endpoint ---
# @app.route('/ask', methods=['POST'])
# def ask_question():
#     data = request.get_json()
#     if not data or 'question' not in data or 'model' not in data:
#         return jsonify({"error": "Request must include 'question' and 'model'."}), 400

#     question = data['question']
#     target_model = data['model']
#     print(f"Received question for model '{target_model}': {question}")

#     # --- STEP A: HYDE - Generate a hypothetical document ---
#     print("Generating hypothetical document for search...")
#     hyde_prompt = (
#         f"You are a helpful assistant for first responders. "
#         f"Generate a concise, factual paragraph that answers the following question about the {target_model} "
#         f"as if it were from the vehicle's Emergency Response Guide.\n\n"
#         f"Question: {question}"
#     )
#     # Keep this generation very short and fast; disable cache to avoid DynamicCache errors
#     hyde_output = generator(
#         hyde_prompt,
#         max_new_tokens=100,
#         do_sample=False,
#         use_cache=False,
#         return_full_text=False,
#     )
#     hypothetical_doc = hyde_output[0]["generated_text"].strip()
#     print(f"HyDE Doc: {hypothetical_doc}")

#     # --- STEP B: RETRIEVE - Search using the HyDE document ---
#     print("Searching with HyDE document...")
#     # We now encode the HYPOTHETICAL document, not the original question
#     search_embedding = retriever.encode([hypothetical_doc]).astype('float32')
    
#     k_search = 10
#     distances, indices = index.search(search_embedding, k_search)

#     # Filter by model and gather context
#     context_text = ""
#     sources = set()
#     for i in indices[0]:
#         retrieved_chunk = chunks[i]
#         if retrieved_chunk['model'] == target_model:
#             context_text += retrieved_chunk['text'] + "\n\n"
#             sources.add(retrieved_chunk['source'])
    
#     if not context_text:
#         return jsonify({"answer": "I could not find any relevant information for that model, even with an expanded search. Please try rephrasing your question.", "sources": []})

#     # --- STEP C: SYNTHESIZE - Generate the final answer ---
#     print("Generating final answer...")
#     system_prompt = """You are an expert assistant for first responders. Your task is to answer the user's original question based *ONLY* on the provided text from the vehicle's Emergency Response Guide. Be clear, concise, and prioritize immediate safety actions. If the information isn't in the provided text, say "The provided guide does not contain that information." Do not make anything up. Structure your answer with clear steps if possible."""

#     final_prompt = (
#         f"{system_prompt}\n\n"
#         f"Original Question: {question}\n\n"
#         f"Emergency Guide Text:\n{context_text}\n\n"
#         f"Answer:"
#     )
    
#     final_output = generator(
#         final_prompt,
#         max_new_tokens=500,
#         do_sample=False,
#         use_cache=False,
#         return_full_text=False,
#     )
#     final_answer = final_output[0]["generated_text"].strip()

#     return jsonify({
#         "answer": final_answer,
#         "sources": list(sources)
#     })

# if __name__ == '__main__':
#     app.run(host='0.0.0.0', port=5000)


from flask import Flask

app = Flask(__name__)

@app.route('/')
def hello():
    return "Hello, World! The test app is running."

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)