from transformers import pipeline

# Replace this with your own checkpoint
model_checkpoint = "replacing-this-with-your-checkpoint"
translator = pipeline("translation", model=model_checkpoint)

# Translate a sentence
text = "Put your own sentence here to translate"

translation = translator(text, max_length=40)
print(translation)