# Install necessary libraries (uncomment if not installed)
# !pip install bertopic umap-learn hdbscan sentence-transformers scikit-learn pandas matplotlib

import pandas as pd
from umap import UMAP
from hdbscan import HDBSCAN
from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import CountVectorizer
from bertopic import BERTopic
from bertopic.representation import KeyBERTInspired
from bertopic.vectorizers import ClassTfidfTransformer
import matplotlib.pyplot as plt

# Sample Dataset (Replace with real data)
documents = [
    # Existing AI-related documents
    "Artificial Intelligence is transforming industries.",
    "Machine Learning is a subset of AI and involves data-driven predictions.",
    "Deep Learning is used in image recognition tasks.",
    "Financial markets rely on AI for fraud detection and automation.",
    "Chatbots and virtual assistants use Generative AI for better interactions.",
    "Cloud computing enables scalable AI applications.",
    "Cybersecurity benefits from AI-powered threat detection.",

    # Additional 30 Non-AI Documents

    # Finance & Economy
    "Stock markets fluctuate based on economic and political events.",
    "Cryptocurrency is gaining popularity as an alternative investment.",
    "Central banks regulate inflation through monetary policies.",
    "Personal finance management helps individuals save and invest wisely.",
    "Real estate investment provides long-term financial stability.",
    "Interest rates impact the cost of borrowing and economic growth.",
    "Credit scores determine an individual’s loan eligibility and interest rates.",
    "The rise of digital banking is transforming traditional financial institutions.",
    
    # Healthcare & Medicine
    "Vaccinations have eradicated many deadly diseases over the years.",
    "Telemedicine is making healthcare more accessible to remote areas.",
    "Mental health awareness is increasing globally due to rising stress levels.",
    "Medical research continues to find new treatments for chronic diseases.",
    "Healthy eating habits and exercise contribute to a longer lifespan.",
    "The pharmaceutical industry plays a crucial role in developing new drugs.",
    "Wearable technology helps monitor heart rate and fitness levels.",

    # Cybersecurity & Technology
    "Phishing attacks are a major threat to personal and corporate data.",
    "Cloud security is essential for businesses storing sensitive information.",
    "Encryption ensures secure communication over the internet.",
    "Data breaches can lead to significant financial and reputational losses.",
    "The Internet of Things (IoT) connects smart devices for automation.",
    "Blockchain technology enhances transparency and security in transactions.",
    "Quantum computing has the potential to revolutionize data processing.",
    
    # Business & Economy
    "E-commerce growth has accelerated due to the rise of online shopping.",
    "Remote work is becoming the new norm in many industries.",
    "Supply chain disruptions impact global trade and production.",
    "Sustainable business practices are crucial for long-term success.",
    "Customer service excellence drives brand loyalty and retention.",
    "Corporate social responsibility enhances a company's public image.",
    "Market competition fosters innovation and better products for consumers."
]

# Step 1 - Load Pretrained Sentence Transformer Model for Embeddings
embedding_model = SentenceTransformer("all-MiniLM-L6-v2")

# Step 2 - Reduce Dimensionality with UMAP
umap_model = UMAP(n_neighbors=15, n_components=5, min_dist=0.0, metric='cosine', low_memory=False)

# Step 3 - Cluster Reduced Embeddings with HDBSCAN
hdbscan_model = HDBSCAN(min_cluster_size=2, metric='euclidean', cluster_selection_method='eom', prediction_data=True)

# Step 4 - Tokenize Topics with CountVectorizer
vectorizer_model = CountVectorizer(stop_words="english")

# Step 5 - Extract Topic Words using Class-based TF-IDF (c-TF-IDF)
ctfidf_model = ClassTfidfTransformer()

# Step 6 - Fine-Tune Topic Representations with KeyBERT
representation_model = KeyBERTInspired()

# Step 7 - Create BERTopic Model
topic_model = BERTopic(
    embedding_model=embedding_model,      # Extract sentence embeddings
    umap_model=umap_model,                # Reduce dimensionality
    hdbscan_model=hdbscan_model,          # Cluster topics
    vectorizer_model=vectorizer_model,    # Tokenize text
    ctfidf_model=ctfidf_model,            # Extract topic words
    representation_model=representation_model  # Fine-tune topic representations
)

# Step 8 - Train the BERTopic Model on Documents
topics, probs = topic_model.fit_transform(documents)

# Step 9 - Display Top Topics
print("Top Topics Identified:")
print(topic_model.get_topic_info())

# Step 10 - Visualize Topics and Save as an Image
fig1 = topic_model.visualize_barchart(top_n_topics=5)
fig1.write_image("barchart_topics.png")  # Save the figure

# Step 11 - Visualize Topic Reduction and Save as an Image
fig2 = topic_model.visualize_topics()
fig2.write_image("topic_reduction.png")  # Save the figure

# Step 12 - Find Representative Words for a Specific Topic (Example: Topic 0)
print("\nTop Words in Topic 0:")
print(topic_model.get_topic(0))
