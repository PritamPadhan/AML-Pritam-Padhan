from flask import Flask, request, jsonify
import joblib
from score import score

# Create Flask app
app = Flask(__name__)
model = joblib.load("best_model.joblib")  # This is your full pipeline (vectorizer + classifier)

# Optional: Print a cool Spamurai banner when the app starts
print(r"""
   ____                                      
  / ___| __ _ _ __ ___   __ _ _ __ ___   ___ 
 | |    / _` | '_ ` _ \ / _` | '_ ` _ \ / _ \
 | |___| (_| | | | | | | (_| | | | | | |  __/
  \____|\__,_|_| |_| |_|\__,_|_| |_| |_|\___| 
                Spamurai: Slice the spam ⚔️
""")

# Define the /score endpoint
@app.route("/score", methods=["POST"])
def score_endpoint():
    data = request.get_json()

    text = data.get("text", "")
    threshold = float(data.get("threshold", 0.5))

    prediction, propensity = score(text, model, threshold)

    return jsonify({
        "app": "Spamurai",
        "prediction": int(prediction),    # 1 or 0
        "propensity": float(propensity)   # probability
    })

# Run the app
if __name__ == "__main__":
    app.run(debug=False, port=5000)
