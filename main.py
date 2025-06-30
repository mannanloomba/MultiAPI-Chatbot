import os
import json
from flask import Flask, render_template, request, jsonify
import google.generativeai as genai
import cohere
from dotenv import load_dotenv
from datetime import datetime
from better_profanity import profanity
import markdown
import logging
# Set up logging
logging.basicConfig(level=logging.DEBUG)
load_dotenv()
# Configure Gemini API
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))
# Configure Cohere API
co = cohere.Client(os.getenv("COHERE_API_KEY"))
# Initialize Flask app
app = Flask(__name__)
# In-memory cache for frequent queries
response_cache = {}
# Path to store chat history
CHAT_HISTORY_FILE = "chat_history.json"
def load_chat_history():
    """Load chat history from JSON file."""
    try:
        if os.path.exists(CHAT_HISTORY_FILE):
            with open(CHAT_HISTORY_FILE, 'r') as f:
                return json.load(f)
        return []
    except Exception as e:
        logging.error(f"Error loading chat history: {e}")
        return []
def save_chat_history(history):
    """Save chat history to JSON file."""
    try:
        with open(CHAT_HISTORY_FILE, 'w') as f:
            json.dump(history, f, indent=2)
    except Exception as e:
        logging.error(f"Error saving chat history: {e}")
def filter_text(text):
    """Apply text filters: profanity and length check."""
    if not text or len(text.strip()) == 0:
        return None, "Error: Empty message."
    if len(text) > 1000:
        return None, "Error: Message too long. Keep under 1000 characters."
    filtered_text = profanity.censor(text)
    return filtered_text, None
def is_complex_query(text):
    """Determine if query requires Gemini (complex) or Cohere (simple)."""
    keywords = ["explain", "detailed", "in-depth", "how does", "why", "analyze"]
    return len(text.split()) > 10 or any(keyword in text.lower() for keyword in keywords)

def chat_with_cohere(prompt):
    """Get a quick response from Cohere API."""
    try:
        response = co.generate(
    model="command",  
    prompt=prompt,
    max_tokens=200,
    temperature=0.7
)
        text = response.generations[0].text.strip()
        # Check response quality (e.g., too short)
        if len(text) < 10:
            logging.debug("Cohere response too short, falling back to Gemini")
            return None
        return markdown.markdown(text)
    except Exception as e:
        logging.error(f"Cohere error: {e}")
        return None

@app.route('/')
def home():
    """Render the main chat page."""
    return render_template('index.html')

@app.route('/chat', methods=['POST'])
def chat():
    """Handle chat messages using Cohere for speed and Gemini for accuracy."""
    try:
        user_input = request.json.get("message")
        if not user_input:
            return jsonify({"response": "Please enter a message."}), 400
        filtered_input, error = filter_text(user_input)
        if error:
            return jsonify({"response": error}), 400
        # Check cache
        cache_key = filtered_input
        if cache_key in response_cache:
            logging.debug("Cache hit")
            return jsonify({"response": response_cache[cache_key], "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")})
        # Decide which API to use
        response = None
        if is_complex_query(filtered_input):
            logging.debug("Using Gemini for complex query")
            
        else:
            logging.debug("Using Cohere for simple query")
            response = chat_with_cohere(filtered_input)
            # Fallback to Gemini if Cohere fails or response is poor
            if not response:
                logging.debug("Cohere failed or poor response, using Gemini")
                
        if not response:
            return jsonify({"response": "Error: Unable to generate response."}), 500
        # Cache response
        response_cache[cache_key] = response
        # Save to history
        history = load_chat_history()
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        history.append({
            "user": filtered_input,
            "bot": response,
            "timestamp": timestamp
        })
        save_chat_history(history)
        return jsonify({"response": response, "timestamp": timestamp})
    except Exception as e:
        logging.error(f"Chat error: {e}")
        return jsonify({"response": f"Server error: {e}"}), 500
@app.route('/history', methods=['GET'])
def get_history():
    """Return chat history."""
    try:
        history = load_chat_history()
        for entry in history:
            entry["bot"] = markdown.markdown(entry["bot"])
        return jsonify({"history": history})
    except Exception as e:
        logging.error(f"History error: {e}")
        return jsonify({"error": f"Failed to load history: {e}"}), 500
@app.route('/clear_history', methods=['POST'])
def clear_history():
    """Clear chat history by resetting the JSON file."""
    try:
        save_chat_history([])
        return jsonify({"success": True})
    except Exception as e:
        logging.error(f"Clear history error: {e}")
        return jsonify({"error": f"Failed to clear history: {e}"}), 500
if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)
