from sentence_transformers import SentenceTransformer
import numpy as np

model = SentenceTransformer("all-MiniLM-L6-v2")

sentence1 = "Alexander Graham Bell invented the telephone."

sentence2 = "The telephone was invented by Alexander Graham Bell."

embedding1 = model.encode(sentence1)
embedding2 = model.encode(sentence2)

similarity = np.dot(embedding1, embedding2) / (
    np.linalg.norm(embedding1) * np.linalg.norm(embedding2)
)

print("Similarity:", similarity)