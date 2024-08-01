from transformers import AutoModelForSequenceClassification, AutoTokenizer, AutoModelForCausalLM, AutoModel, pipeline
from torch.optim import AdamW

model = AutoModelForSequenceClassification.from_pretrained("distilbert-base-uncased")


classifier = pipeline("text-classification",model='bhadresh-savani/distilbert-base-uncased-emotion', return_all_scores=True)
tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")