from flask import Flask, render_template, request
from model import predict_note

app = Flask(__name__)

# Homepage
@app.route('/', methods=['GET', 'POST'])
def index():
    prediction = None
    if request.method == 'POST':

        # Get file from user
        file = request.files['file']

        # Pass file to prediction function
        prediction = predict_note(file)

    return render_template("index.html", prediction=prediction)

if __name__ == '__main__':
    app.run(debug=True)
