from transformers import AutoTokenizer, BartForConditionalGeneration

# Load DistilBART model and tokenizer
model_name = "sshleifer/distilbart-cnn-12-6"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = BartForConditionalGeneration.from_pretrained(model_name)

# Article to summarize
ARTICLE_TO_SUMMARIZE = (
    "Artificial Intelligence (AI) is revolutionizing various industries by automating tasks, "
    "enhancing decision-making, and improving efficiency. In healthcare, AI-powered diagnostics help detect diseases such as cancer at early stages, improving patient outcomes. AI-driven predictive analytics in finance assists in fraud detection and algorithmic trading, reducing financial risks. Similarly, "
    "in the retail industry, AI is used for personalized recommendations, optimizing customer experiences and increasing sales.One of the most transformative applications of AI is in autonomous systems, such as self-driving cars and robotic process automation. These technologies reduce human intervention, minimizing "
    "errors and improving operational efficiency. Additionally, AI plays a crucial role in cybersecurity by identifying potential threats in real-time, helping organizations prevent cyberattacks."
    "Despite its benefits, AI poses ethical challenges, including data privacy concerns and biases in decision-making models. Ensuring transparency and fairness in AI systems is crucial for gaining public trust and maximizing its potential benefits. As AI continues to evolve, organizations must strike a balance between innovation and responsible AI deployment."
)

# Tokenize the input text
inputs = tokenizer(ARTICLE_TO_SUMMARIZE, max_length=1024, truncation=True, return_tensors="pt")

# Generate Summary
summary_ids = model.generate(
    inputs["input_ids"], 
    num_beams=4,          # More beams for better quality
    min_length=30,        # Avoid very short summaries
    max_length=100,       # Prevent excessive truncation
    length_penalty=2.0,   # Encourage longer summaries
    early_stopping=True
)

# Decode and print the summary
summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True, clean_up_tokenization_spaces=True)
print("Summary:", summary)
