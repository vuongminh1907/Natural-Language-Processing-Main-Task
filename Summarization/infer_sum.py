from transformers import pipeline

hub_model_id = "put_your_model_id_here"
summarizer = pipeline("summarization", model=hub_model_id)

#read the text from the file
text_dir = "path_to_your_text_file"
with open(text_dir, "r") as file:
    text = file.read()

#summarize the text
summary = summarizer(text)[0]['summary_text']
print(summary)