from flask import Flask, render_template, request
from transformers import pipeline
import os

# Make a flask app
app = Flask(__name__)

# Load the summarizer model using pipeline
summarizer = pipeline("summarization", model="sshleifer/distilbart-cnn-12-6")

# Home route – both GET and POST
@app.route("/", methods=["GET", "POST"])
def index():
    summary = ""
    if request.method == "POST":
        text = request.form["text"]  # Take input text from user
        if len(text.split()) > 50:  # If more than 50 words, then only do summary
            summary_text = summarizer(text, max_length=130, min_length=30, do_sample=False)
            summary = summary_text[0]['summary_text']  # Get only the summary text
        else:
            summary = "Text too short for summarization. Please enter at least 50 words."  # Show this if text is small
    return render_template("index.html", summary=summary)  # Send summary to index.html

# Run the app

if __name__ == "__main__":
    app.run(debug=True) # debug mode helps to see errors while testing
