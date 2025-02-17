# app.py
from flask import Flask, render_template, request, jsonify
import pickle

app = Flask(__name__)

# Load the saved model and vectorizer
with open('model.pkl', 'rb') as file:
    vectorizer, model = pickle.load(file)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        # Get input from the form
        email_text = request.form['email_text']
        
        # Transform the input using the saved TF-IDF Vectorizer
        email_vec = vectorizer.transform([email_text])
        
        # Predict using the saved model
        prediction = model.predict(email_vec)
        
        # Return result
        result = "Spam" if prediction[0] == 1 else "Not Spam"
        return jsonify({'prediction': result})

if __name__ == '__main__':
    app.run()