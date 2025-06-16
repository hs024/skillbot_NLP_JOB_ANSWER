# from sentence_transformers import SentenceTransformer, util
# from .data import questions, answers

# # Load model only once
# model = SentenceTransformer("all-MiniLM-L6-v2")

# # Encode the predefined questions
# question_embeddings = model.encode(questions, convert_to_tensor=True)

# def get_answer(user_query):
#     # Encode user query
#     query_embedding = model.encode(user_query, convert_to_tensor=True)
    
#     # Compute similarity scores
#     similarities = util.pytorch_cos_sim(query_embedding, question_embeddings)[0]
    
#     # Get the best match index
#     best_match_idx = int(similarities.argmax())
    
#     return answers[best_match_idx]

from transformers import pipeline

# Load QA model
qa_pipeline = pipeline("question-answering", model="deepset/roberta-base-squad2")

# Load context
with open("chatbot/context.txt", "r", encoding="utf-8") as f:
    context = f.read()

def get_answer(user_query):
    result = qa_pipeline({
        "question": user_query,
        "context": context
    })

    return result["answer"]
