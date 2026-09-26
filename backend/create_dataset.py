from sentence_transformers import SentenceTransformer
import numpy as np
import csv

print("Loading embedding model...")
model = SentenceTransformer("all-MiniLM-L6-v2")
print("Embedding model loaded!")
print()


# ============================================================
# 50 QUESTION + EVIDENCE PAIRS
#
# Each pair creates:
#   1 supported answer
#   1 hallucinated answer
#
# Therefore:
#   50 supported + 50 hallucinated = 100 examples
#
# label = 1 -> supported
# label = 0 -> hallucinated
# ============================================================

examples = [

    {
        "question": "What is machine learning?",
        "evidence": "Machine Learning is a branch of Artificial Intelligence that enables computers to learn from data and make predictions or decisions without being explicitly programmed for every possible situation.",
        "answer": "Machine learning is a branch of AI that learns from data to make predictions or decisions.",
        "label": 1
    },

    {
        "question": "What is machine learning?",
        "evidence": "Machine Learning is a branch of Artificial Intelligence that enables computers to learn from data and make predictions or decisions without being explicitly programmed for every possible situation.",
        "answer": "Machine learning is a programming language used to build websites.",
        "label": 0
    },

    {
        "question": "What does machine learning identify in data?",
        "evidence": "The computer analyzes data, identifies patterns, and uses those patterns to produce useful results.",
        "answer": "Machine learning identifies patterns in data.",
        "label": 1
    },

    {
        "question": "What does machine learning identify in data?",
        "evidence": "The computer analyzes data, identifies patterns, and uses those patterns to produce useful results.",
        "answer": "Machine learning identifies the physical location of every computer in a network.",
        "label": 0
    },

    {
        "question": "What is the basic idea behind machine learning?",
        "evidence": "The basic idea behind machine learning is simple: give a computer enough useful data, allow it to find patterns in that data, and use those patterns to make predictions about new data.",
        "answer": "The basic idea is to learn patterns from useful data and use them to make predictions about new data.",
        "label": 1
    },

    {
        "question": "What is the basic idea behind machine learning?",
        "evidence": "The basic idea behind machine learning is simple: give a computer enough useful data, allow it to find patterns in that data, and use those patterns to make predictions about new data.",
        "answer": "The basic idea is to manually write a separate rule for every possible future situation.",
        "label": 0
    },

    {
        "question": "Where is machine learning used?",
        "evidence": "Machine learning is used in smartphones, search engines, social media, online shopping, banking, healthcare, transportation, entertainment, and many other fields.",
        "answer": "Machine learning is used in areas such as smartphones, banking, healthcare, transportation, and entertainment.",
        "label": 1
    },

    {
        "question": "Where is machine learning used?",
        "evidence": "Machine learning is used in smartphones, search engines, social media, online shopping, banking, healthcare, transportation, entertainment, and many other fields.",
        "answer": "Machine learning is used only in scientific laboratories.",
        "label": 0
    },

    {
        "question": "What are some applications of machine learning?",
        "evidence": "Applications such as voice assistants, movie recommendations, spam filters, facial recognition, and fraud detection rely heavily on machine learning techniques.",
        "answer": "Examples include voice assistants, movie recommendations, spam filters, facial recognition, and fraud detection.",
        "label": 1
    },

    {
        "question": "What are some applications of machine learning?",
        "evidence": "Applications such as voice assistants, movie recommendations, spam filters, facial recognition, and fraud detection rely heavily on machine learning techniques.",
        "answer": "Examples include weather control, teleportation, and time travel.",
        "label": 0
    },

    {
        "question": "What is the first step in a machine learning system?",
        "evidence": "The first step is collecting data. Data is the foundation of a machine learning model.",
        "answer": "The first step is collecting data.",
        "label": 1
    },

    {
        "question": "What is the first step in a machine learning system?",
        "evidence": "The first step is collecting data. Data is the foundation of a machine learning model.",
        "answer": "The first step is always deploying the model before collecting data.",
        "label": 0
    },

    {
        "question": "What forms can machine learning data take?",
        "evidence": "Depending on the problem, the data could consist of images, text, numbers, audio, videos, or other forms of information.",
        "answer": "Machine learning data can include images, text, numbers, audio, and videos.",
        "label": 1
    },

    {
        "question": "What forms can machine learning data take?",
        "evidence": "Depending on the problem, the data could consist of images, text, numbers, audio, videos, or other forms of information.",
        "answer": "Machine learning can only use numerical data.",
        "label": 0
    },

    {
        "question": "Why is data preprocessing used?",
        "evidence": "Data preprocessing helps make the information suitable for training a model.",
        "answer": "Data preprocessing prepares information so it is suitable for training a model.",
        "label": 1
    },

    {
        "question": "Why is data preprocessing used?",
        "evidence": "Data preprocessing helps make the information suitable for training a model.",
        "answer": "Data preprocessing is used to permanently remove the trained model.",
        "label": 0
    },

    {
        "question": "What problems can real-world data contain?",
        "evidence": "Real-world data can contain missing values, incorrect information, duplicate records, or irrelevant details.",
        "answer": "Real-world data can contain missing values, incorrect information, duplicates, and irrelevant details.",
        "label": 1
    },

    {
        "question": "What problems can real-world data contain?",
        "evidence": "Real-world data can contain missing values, incorrect information, duplicate records, or irrelevant details.",
        "answer": "Real-world data is always perfect and never contains incorrect information.",
        "label": 0
    },

    {
        "question": "What happens during model training?",
        "evidence": "During training, a machine learning algorithm examines the available data and attempts to identify patterns or relationships.",
        "answer": "During training, the algorithm examines data and tries to identify patterns or relationships.",
        "label": 1
    },

    {
        "question": "What happens during model training?",
        "evidence": "During training, a machine learning algorithm examines the available data and attempts to identify patterns or relationships.",
        "answer": "During training, the model deletes all patterns from the data.",
        "label": 0
    },

    {
        "question": "Why is a model tested on unseen data?",
        "evidence": "Once training is complete, the model can be tested using data that it has not seen before. This helps determine whether the model has actually learned useful patterns or has simply memorized the training examples.",
        "answer": "Testing on unseen data helps determine whether the model learned useful patterns rather than simply memorizing training examples.",
        "label": 1
    },

    {
        "question": "Why is a model tested on unseen data?",
        "evidence": "Once training is complete, the model can be tested using data that it has not seen before.",
        "answer": "Unseen data is used only to make the computer run faster.",
        "label": 0
    },

    {
        "question": "What is the simplified machine learning process?",
        "evidence": "A simplified machine learning process can therefore be represented as: Data → Preprocessing → Training → Testing → Prediction.",
        "answer": "The process is Data, Preprocessing, Training, Testing, and Prediction.",
        "label": 1
    },

    {
        "question": "What is the simplified machine learning process?",
        "evidence": "A simplified machine learning process can therefore be represented as: Data → Preprocessing → Training → Testing → Prediction.",
        "answer": "The process is Prediction, Gaming, Shutdown, and Deleting Data.",
        "label": 0
    },

    {
        "question": "What is supervised learning?",
        "evidence": "Supervised learning is a type of machine learning in which the model is trained using labeled data. This means that the correct answer is already provided during training.",
        "answer": "Supervised learning trains a model using labeled data with the correct answer provided.",
        "label": 1
    },

    {
        "question": "What is supervised learning?",
        "evidence": "Supervised learning is a type of machine learning in which the model is trained using labeled data.",
        "answer": "Supervised learning always works without labels.",
        "label": 0
    },

    {
        "question": "What are two common forms of supervised learning?",
        "evidence": "Supervised learning is commonly divided into classification and regression.",
        "answer": "The two common forms are classification and regression.",
        "label": 1
    },

    {
        "question": "What are two common forms of supervised learning?",
        "evidence": "Supervised learning is commonly divided into classification and regression.",
        "answer": "The two forms are clustering and robotics.",
        "label": 0
    },

    {
        "question": "What is classification used for?",
        "evidence": "Classification is used when the output belongs to categories, such as spam or not spam.",
        "answer": "Classification is used when the output belongs to categories.",
        "label": 1
    },

    {
        "question": "What is classification used for?",
        "evidence": "Classification is used when the output belongs to categories, such as spam or not spam.",
        "answer": "Classification is used only to predict continuous house prices.",
        "label": 0
    },

    {
        "question": "What is regression used for?",
        "evidence": "Regression is used when the output is a numerical value, such as predicting the price of a house.",
        "answer": "Regression is used to predict numerical values such as house prices.",
        "label": 1
    },

    {
        "question": "What is regression used for?",
        "evidence": "Regression is used when the output is a numerical value, such as predicting the price of a house.",
        "answer": "Regression is used only to place customers into categories.",
        "label": 0
    },

    {
        "question": "What is unsupervised learning?",
        "evidence": "Unsupervised learning works with data that does not have predefined labels. Instead, the algorithm attempts to discover hidden patterns or groups within the data.",
        "answer": "Unsupervised learning uses data without predefined labels to discover patterns or groups.",
        "label": 1
    },

    {
        "question": "What is unsupervised learning?",
        "evidence": "Unsupervised learning works with data that does not have predefined labels.",
        "answer": "Unsupervised learning requires every example to have a predefined label.",
        "label": 0
    },

    {
        "question": "What is clustering?",
        "evidence": "One common technique is clustering, where similar data points are placed into groups.",
        "answer": "Clustering places similar data points into groups.",
        "label": 1
    },

    {
        "question": "What is clustering?",
        "evidence": "One common technique is clustering, where similar data points are placed into groups.",
        "answer": "Clustering means deleting similar data points from a dataset.",
        "label": 0
    },

    {
        "question": "What is unsupervised learning useful for?",
        "evidence": "Unsupervised learning is useful for customer segmentation, pattern discovery, anomaly detection, and data exploration.",
        "answer": "It is useful for customer segmentation, pattern discovery, anomaly detection, and data exploration.",
        "label": 1
    },

    {
        "question": "What is unsupervised learning useful for?",
        "evidence": "Unsupervised learning is useful for customer segmentation, pattern discovery, anomaly detection, and data exploration.",
        "answer": "It is useful only for predicting house prices.",
        "label": 0
    },

    {
        "question": "What is reinforcement learning based on?",
        "evidence": "Reinforcement learning is based on the idea of learning through interaction and feedback.",
        "answer": "Reinforcement learning is based on learning through interaction and feedback.",
        "label": 1
    },

    {
        "question": "What is reinforcement learning based on?",
        "evidence": "Reinforcement learning is based on the idea of learning through interaction and feedback.",
        "answer": "Reinforcement learning is based entirely on fixed rules that never change.",
        "label": 0
    },

    {
        "question": "What does an agent receive in reinforcement learning?",
        "evidence": "An agent interacts with an environment and receives rewards or penalties depending on its actions.",
        "answer": "An agent receives rewards or penalties depending on its actions.",
        "label": 1
    },

    {
        "question": "What does an agent receive in reinforcement learning?",
        "evidence": "An agent interacts with an environment and receives rewards or penalties depending on its actions.",
        "answer": "An agent receives only salary payments for its actions.",
        "label": 0
    },


    {
        "question": "What is data quality?",
        "evidence": "One major challenge is data quality. A model trained on inaccurate or biased data may produce inaccurate or biased predictions.",
        "answer": "Data quality is important because inaccurate or biased data can lead to inaccurate or biased predictions.",
        "label": 1
    },

    {
        "question": "What is data quality?",
        "evidence": "One major challenge is data quality.",
        "answer": "Data quality means that every computer must have a high-quality processor.",
        "label": 0
    },

    {
        "question": "What is a machine learning model?",
        "evidence": "During training, a machine learning algorithm examines the available data and attempts to identify patterns or relationships.",
        "answer": "A machine learning model learns patterns or relationships from data during training.",
        "label": 1
    },

    {
        "question": "What is a machine learning model?",
        "evidence": "During training, a machine learning algorithm examines the available data and attempts to identify patterns or relationships.",
        "answer": "A machine learning model is a physical robot that collects data.",
        "label": 0
    },

    {
        "question": "What is anomaly detection?",
        "evidence": "Unsupervised learning is useful for customer segmentation, pattern discovery, anomaly detection, and data exploration.",
        "answer": "Anomaly detection is one application of unsupervised learning.",
        "label": 1
    },

    {
        "question": "What is anomaly detection?",
        "evidence": "Unsupervised learning is useful for customer segmentation, pattern discovery, anomaly detection, and data exploration.",
        "answer": "Anomaly detection is a method for increasing the brightness of computer screens.",
        "label": 0
    },

    {
        "question": "What is customer segmentation?",
        "evidence": "Unsupervised learning is useful for customer segmentation, pattern discovery, anomaly detection, and data exploration.",
        "answer": "Customer segmentation is an application of unsupervised learning.",
        "label": 1
    },

    {
        "question": "What is customer segmentation?",
        "evidence": "Unsupervised learning is useful for customer segmentation, pattern discovery, anomaly detection, and data exploration.",
        "answer": "Customer segmentation means repairing computer hardware.",
        "label": 0
    },

    {
        "question": "What does a supervised learning model learn from?",
        "evidence": "Supervised learning is a type of machine learning in which the model is trained using labeled data.",
        "answer": "A supervised learning model learns from labeled data.",
        "label": 1
    },

    {
        "question": "What does a supervised learning model learn from?",
        "evidence": "Supervised learning is a type of machine learning in which the model is trained using labeled data.",
        "answer": "A supervised learning model learns only from unlabeled data.",
        "label": 0
    },

    {
        "question": "What is prediction in machine learning?",
        "evidence": "A simplified machine learning process can therefore be represented as: Data → Preprocessing → Training → Testing → Prediction.",
        "answer": "Prediction is a step that follows training and testing in the simplified machine learning process.",
        "label": 1
    },

    {
        "question": "What is prediction in machine learning?",
        "evidence": "A simplified machine learning process can therefore be represented as: Data → Preprocessing → Training → Testing → Prediction.",
        "answer": "Prediction happens before any data is collected.",
        "label": 0
    },

    {
        "question": "What is fraud detection?",
        "evidence": "In banking and finance, machine learning can detect unusual transactions and identify potential fraud.",
        "answer": "Machine learning can help detect unusual transactions and potential fraud.",
        "label": 1
    },

    {
        "question": "What is fraud detection?",
        "evidence": "In banking and finance, machine learning can detect unusual transactions and identify potential fraud.",
        "answer": "Machine learning guarantees that no financial fraud can ever occur.",
        "label": 0
    },



    {
        "question": "What is Linear Regression used for?",
        "evidence": "Linear Regression is a relatively simple algorithm used to predict numerical values.",
        "answer": "Linear Regression is used to predict numerical values.",
        "label": 1
    },

    {
        "question": "What is Linear Regression used for?",
        "evidence": "Linear Regression is a relatively simple algorithm used to predict numerical values.",
        "answer": "Linear Regression is used only to classify spam.",
        "label": 0
    },

    {
        "question": "What is Logistic Regression commonly used for?",
        "evidence": "Logistic Regression is commonly used for classification problems.",
        "answer": "Logistic Regression is commonly used for classification problems.",
        "label": 1
    },

    {
        "question": "What is Logistic Regression commonly used for?",
        "evidence": "Logistic Regression is commonly used for classification problems.",
        "answer": "Logistic Regression is used only for predicting continuous house prices.",
        "label": 0
    },

    {
        "question": "How do Decision Trees make decisions?",
        "evidence": "Decision Trees make decisions using a structure similar to a flowchart. At each stage, the algorithm asks a question about the data and follows a particular path based on the answer.",
        "answer": "Decision Trees ask questions about data and follow paths based on the answers, like a flowchart.",
        "label": 1
    },

    {
        "question": "How do Decision Trees make decisions?",
        "evidence": "Decision Trees make decisions using a structure similar to a flowchart.",
        "answer": "Decision Trees make decisions by randomly ignoring every feature.",
        "label": 0
    },

    {
        "question": "What do Random Forests combine?",
        "evidence": "Random Forests combine multiple decision trees to improve prediction performance and reduce the weaknesses of an individual tree.",
        "answer": "Random Forests combine multiple decision trees.",
        "label": 1
    },

    {
        "question": "What do Random Forests combine?",
        "evidence": "Random Forests combine multiple decision trees to improve prediction performance.",
        "answer": "Random Forests combine multiple databases instead of decision trees.",
        "label": 0
    },

    {
        "question": "What do Support Vector Machines attempt to find?",
        "evidence": "Support Vector Machines are algorithms that attempt to find boundaries that effectively separate different categories of data.",
        "answer": "SVMs attempt to find boundaries that separate different categories of data.",
        "label": 1
    },

    {
        "question": "What do Support Vector Machines attempt to find?",
        "evidence": "Support Vector Machines are algorithms that attempt to find boundaries that effectively separate different categories of data.",
        "answer": "SVMs attempt to find the physical boundaries of computer buildings.",
        "label": 0
    },

    {
        "question": "What are Neural Networks particularly powerful for?",
        "evidence": "Neural Networks are particularly powerful for complex tasks involving images, speech, and natural language.",
        "answer": "Neural Networks are powerful for complex image, speech, and natural-language tasks.",
        "label": 1
    },

    {
        "question": "What are Neural Networks particularly powerful for?",
        "evidence": "Neural Networks are particularly powerful for complex tasks involving images, speech, and natural language.",
        "answer": "Neural Networks are powerful only for simple arithmetic by hand.",
        "label": 0
    },

    {
        "question": "What is Artificial Intelligence?",
        "evidence": "Artificial Intelligence is the broader field concerned with creating systems capable of performing tasks that normally require human intelligence.",
        "answer": "AI is the broader field of creating systems capable of tasks that normally require human intelligence.",
        "label": 1
    },

    {
        "question": "What is Artificial Intelligence?",
        "evidence": "Artificial Intelligence is the broader field concerned with creating systems capable of performing tasks that normally require human intelligence.",
        "answer": "AI is a database format used only for storing images.",
        "label": 0
    },

    {
        "question": "How are machine learning and artificial intelligence related?",
        "evidence": "Machine Learning is one approach used to achieve AI.",
        "answer": "Machine learning is one approach used to achieve AI.",
        "label": 1
    },

    {
        "question": "How are machine learning and artificial intelligence related?",
        "evidence": "Machine Learning is one approach used to achieve AI.",
        "answer": "AI and machine learning are completely unrelated fields.",
        "label": 0
    },

    {
        "question": "What is deep learning?",
        "evidence": "Deep Learning is a specialized area of machine learning that uses multi-layer neural networks.",
        "answer": "Deep learning is a specialized area of machine learning that uses multi-layer neural networks.",
        "label": 1
    },

    {
        "question": "What is deep learning?",
        "evidence": "Deep Learning is a specialized area of machine learning that uses multi-layer neural networks.",
        "answer": "Deep learning is a database technique that never uses neural networks.",
        "label": 0
    },

    {
        "question": "Where has deep learning played a major role?",
        "evidence": "Deep learning has played a major role in recent advances in computer vision, speech recognition, and generative AI.",
        "answer": "It has played a major role in computer vision, speech recognition, and generative AI.",
        "label": 1
    },

    {
        "question": "Where has deep learning played a major role?",
        "evidence": "Deep learning has played a major role in recent advances in computer vision, speech recognition, and generative AI.",
        "answer": "It has played a major role only in mechanical engine repair.",
        "label": 0
    },

    {
        "question": "How can machine learning be used in healthcare?",
        "evidence": "In healthcare, machine learning can help analyze medical images, identify patterns in patient data, and assist researchers in discovering potential treatments.",
        "answer": "It can analyze medical images, identify patterns in patient data, and assist treatment research.",
        "label": 1
    },

    {
        "question": "How can machine learning be used in healthcare?",
        "evidence": "In healthcare, machine learning can help analyze medical images and identify patterns in patient data.",
        "answer": "It can replace every medical professional and make all medical decisions without oversight.",
        "label": 0
    },

    {
        "question": "How can machine learning help in banking?",
        "evidence": "In banking and finance, machine learning can detect unusual transactions and identify potential fraud.",
        "answer": "It can detect unusual transactions and identify potential fraud.",
        "label": 1
    },

    {
        "question": "How can machine learning help in banking?",
        "evidence": "In banking and finance, machine learning can detect unusual transactions and identify potential fraud.",
        "answer": "It can guarantee that every financial transaction is profitable.",
        "label": 0
    },

    {
        "question": "How do e-commerce recommendation systems work?",
        "evidence": "In e-commerce, recommendation systems analyze users' previous activities to suggest products they may be interested in.",
        "answer": "They analyze users' previous activities to suggest products of interest.",
        "label": 1
    },

    {
        "question": "How do e-commerce recommendation systems work?",
        "evidence": "In e-commerce, recommendation systems analyze users' previous activities to suggest products they may be interested in.",
        "answer": "They recommend exactly the same products to every user regardless of previous activity.",
        "label": 0
    },

    {
        "question": "How is machine learning used in transportation?",
        "evidence": "In transportation, machine learning is used for route optimization, traffic prediction, and the development of autonomous driving technologies.",
        "answer": "It is used for route optimization, traffic prediction, and autonomous-driving technologies.",
        "label": 1
    },

    {
        "question": "How is machine learning used in transportation?",
        "evidence": "In transportation, machine learning is used for route optimization, traffic prediction, and the development of autonomous driving technologies.",
        "answer": "It is used only to paint roads and manufacture traffic lights.",
        "label": 0
    },

    {
        "question": "What can modern language systems do?",
        "evidence": "Modern systems can use machine learning to translate languages, summarize documents, answer questions, and generate text.",
        "answer": "They can translate languages, summarize documents, answer questions, and generate text.",
        "label": 1
    },

    {
        "question": "What can modern language systems do?",
        "evidence": "Modern systems can use machine learning to translate languages, summarize documents, answer questions, and generate text.",
        "answer": "They can only store files and cannot work with language.",
        "label": 0
    },

    {
        "question": "What is one major advantage of machine learning?",
        "evidence": "One major advantage is its ability to process large amounts of data much faster than humans can manually analyze it.",
        "answer": "A major advantage is that it can process large amounts of data much faster than humans can manually analyze it.",
        "label": 1
    },

    {
        "question": "What is one major advantage of machine learning?",
        "evidence": "One major advantage is its ability to process large amounts of data much faster than humans can manually analyze it.",
        "answer": "A major advantage is that machine learning never needs data.",
        "label": 0
    },

    {
        "question": "Can machine learning automate repetitive tasks?",
        "evidence": "Machine learning systems can also automate repetitive tasks. For example, an email service can automatically classify thousands of messages as spam or legitimate.",
        "answer": "Yes. Machine learning can automate repetitive tasks such as classifying large numbers of emails.",
        "label": 1
    },

    {
        "question": "Can machine learning automate repetitive tasks?",
        "evidence": "Machine learning systems can also automate repetitive tasks.",
        "answer": "No. Machine learning cannot automate repetitive tasks.",
        "label": 0
    },

    {
        "question": "What is a major challenge of machine learning?",
        "evidence": "One major challenge is data quality. A model trained on inaccurate or biased data may produce inaccurate or biased predictions.",
        "answer": "A major challenge is data quality because poor data can lead to poor predictions.",
        "label": 1
    },

    {
        "question": "What is a major challenge of machine learning?",
        "evidence": "One major challenge is data quality.",
        "answer": "The only challenge is the color of the computer screen.",
        "label": 0
    },

    {
        "question": "What is overfitting?",
        "evidence": "Overfitting occurs when a model learns the training data too closely and performs poorly on new, unseen data.",
        "answer": "Overfitting occurs when a model learns training data too closely and performs poorly on unseen data.",
        "label": 1
    },

    {
        "question": "What is overfitting?",
        "evidence": "Overfitting occurs when a model learns the training data too closely and performs poorly on new, unseen data.",
        "answer": "Overfitting means a model performs perfectly on every new dataset.",
        "label": 0
    },

    {
        "question": "What is the black-box problem?",
        "evidence": "Some advanced machine learning models can be difficult to interpret. This is sometimes called the black-box problem because it may be difficult to understand exactly why a model produced a particular prediction.",
        "answer": "The black-box problem refers to difficulty understanding exactly why a model produced a prediction.",
        "label": 1
    },

    {
        "question": "What is the black-box problem?",
        "evidence": "Some advanced machine learning models can be difficult to interpret.",
        "answer": "The black-box problem means the computer physically has to be placed inside a black box.",
        "label": 0
    },

    {
        "question": "Can machine learning require significant computational resources?",
        "evidence": "Machine learning can also require significant computational resources, especially when training large deep learning models.",
        "answer": "Yes. Machine learning can require significant computational resources, especially for large deep learning models.",
        "label": 1
    },

    {
        "question": "Can machine learning require significant computational resources?",
        "evidence": "Machine learning can require significant computational resources, especially when training large deep learning models.",
        "answer": "Large deep learning models require no computational resources.",
        "label": 0
    },

    {
        "question": "What can generative AI systems create?",
        "evidence": "Generative AI systems are demonstrating how machine learning can be used to create text, images, audio, video, and software.",
        "answer": "Generative AI can create text, images, audio, video, and software.",
        "label": 1
    },

    {
        "question": "What can generative AI systems create?",
        "evidence": "Generative AI systems are demonstrating how machine learning can be used to create text, images, audio, video, and software.",
        "answer": "Generative AI can only create numerical spreadsheets and cannot create text or images.",
        "label": 0
    }

]


# ============================================================
# CREATE FEATURES
# ============================================================

dataset = []

for number, example in enumerate(examples, start=1):

    print("Processing example", number, "of", len(examples))

    question = example["question"]
    evidence = example["evidence"]
    answer = example["answer"]
    label = example["label"]

    # Convert question into an embedding
    question_embedding = model.encode(question)

    # Convert evidence into an embedding
    evidence_embedding = model.encode(evidence)

    # Convert answer into an embedding
    answer_embedding = model.encode(answer)


    # --------------------------------------------------------
    # FEATURE 1
    # Answer-Evidence Similarity
    # --------------------------------------------------------

    answer_evidence_similarity = np.dot(
        answer_embedding,
        evidence_embedding
    ) / (
        np.linalg.norm(answer_embedding)
        *
        np.linalg.norm(evidence_embedding)
    )


    # --------------------------------------------------------
    # FEATURE 2
    # Question-Evidence Similarity
    # --------------------------------------------------------

    question_evidence_similarity = np.dot(
        question_embedding,
        evidence_embedding
    ) / (
        np.linalg.norm(question_embedding)
        *
        np.linalg.norm(evidence_embedding)
    )


    # --------------------------------------------------------
    # FEATURE 3
    # Answer Length
    # --------------------------------------------------------

    answer_length = len(answer.split())


    # Add the features and label to our dataset
    dataset.append([
        answer_evidence_similarity,
        question_evidence_similarity,
        answer_length,
        label
    ])


# ============================================================
# SAVE DATASET
# ============================================================

with open(
    "hallucination_dataset.csv",
    "w",
    newline=""
) as file:

    writer = csv.writer(file)

    # Column names
    writer.writerow([
        "answer_evidence_similarity",
        "question_evidence_similarity",
        "answer_length",
        "label"
    ])


    # Write every example
    for row in dataset:
        writer.writerow(row)


# ============================================================
# FINISHED
# ============================================================

print()
print("============================================")
print("100-EXAMPLE DATASET CREATED")
print("============================================")

print("Total examples:", len(dataset))

print(
    "Supported examples:",
    sum(row[3] == 1 for row in dataset)
)

print(
    "Hallucinated examples:",
    sum(row[3] == 0 for row in dataset)
)

print()
print("Saved as:")
print("hallucination_dataset.csv")

print("============================================")