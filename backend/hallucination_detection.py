from sentence_transformers import SentenceTransformer
import numpy as np
import pickle


# ============================================
# LOAD EMBEDDING MODEL
# ============================================

print("Loading embedding model...")

embedding_model = SentenceTransformer(
    "all-MiniLM-L6-v2"
)

print("Embedding model loaded!")
print()


# ============================================
# LOAD TRAINED XGBOOST MODEL
# ============================================

print("Loading XGBoost model...")

with open("xgboost_model.pkl", "rb") as file:
    xgboost_model = pickle.load(file)

print("XGBoost model loaded!")
print()


# ============================================
# FUNCTION TO DETECT HALLUCINATION
# ============================================

def detect_hallucination(
    question,
    answer,
    top_chunks
):

    print("Checking answer for hallucination...")

    # ----------------------------------------
    # CREATE EMBEDDINGS
    # ----------------------------------------

    question_embedding = embedding_model.encode(
        question
    )

    answer_embedding = embedding_model.encode(
        answer
    )


    # ----------------------------------------
    # FIND BEST EVIDENCE SIMILARITY
    # ----------------------------------------

    evidence_similarities = []

    for chunk in top_chunks:

        chunk_embedding = embedding_model.encode(
            chunk
        )

        similarity = np.dot(
            answer_embedding,
            chunk_embedding
        ) / (
            np.linalg.norm(answer_embedding)
            *
            np.linalg.norm(chunk_embedding)
        )

        evidence_similarities.append(
            similarity
        )


    # Take the highest similarity
    best_answer_evidence_similarity = max(
        evidence_similarities
    )


    # ----------------------------------------
    # QUESTION-EVIDENCE SIMILARITY
    # ----------------------------------------

    question_evidence_similarities = []

    for chunk in top_chunks:

        chunk_embedding = embedding_model.encode(
            chunk
        )

        similarity = np.dot(
            question_embedding,
            chunk_embedding
        ) / (
            np.linalg.norm(question_embedding)
            *
            np.linalg.norm(chunk_embedding)
        )

        question_evidence_similarities.append(
            similarity
        )


    # Take the highest similarity
    best_question_evidence_similarity = max(
        question_evidence_similarities
    )


    # ----------------------------------------
    # ANSWER LENGTH
    # ----------------------------------------

    answer_length = len(
        answer.split()
    )


    # ----------------------------------------
    # CREATE FEATURES
    # ----------------------------------------

    features = [[
        best_answer_evidence_similarity,
        best_question_evidence_similarity,
        answer_length
    ]]


    # ----------------------------------------
    # ASK XGBOOST FOR PREDICTION
    # ----------------------------------------

    prediction = xgboost_model.predict(
        features
    )


    # ----------------------------------------
    # DISPLAY RESULT
    # ----------------------------------------

    print()
    print("============================================")
    print("HALLUCINATION DETECTION RESULT")
    print("============================================")

    print(
        "Answer-Evidence Similarity:",
        best_answer_evidence_similarity
    )

    print(
        "Question-Evidence Similarity:",
        best_question_evidence_similarity
    )

    print(
        "Answer Length:",
        answer_length
    )

    print()


    if prediction[0] == 1:

        print("RESULT: SUPPORTED")

    else:

        print("RESULT: HALLUCINATED")


    print("============================================")


    return prediction[0]