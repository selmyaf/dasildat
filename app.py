from flask import Flask, request, redirect, url_for, render_template_string, send_file
import pandas as pd
import joblib
import os
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

app = Flask(__name__)

HTML_TEMPLATE = '''
<!doctype html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Bank Marketing Deposit Prediction</title>
    <style>
        body {
            background-color: #f0f8ff;
            font-family: Arial, sans-serif;
            display: flex;
            justify-content: center;
            align-items: center;
            height: 100vh;
            margin: 0;
        }
        h1 {
            color: #1e90ff;
        }
        .container {
            text-align: center;
            padding: 20px;
            background-color: #fff;
            border: 2px solid #1e90ff;
            border-radius: 10px;
            box-shadow: 0 4px 8px rgba(0, 0, 0, 0.1);
        }
        .upload-form {
            margin-top: 20px;
        }
        .upload-form input[type="file"] {
            display: block;
            margin: 10px auto;
            padding: 5px;
            border: 1px solid #ddd;
            border-radius: 5px;
        }
        .upload-form input[type="submit"] {
            background-color: #1e90ff;
            color: white;
            border: none;
            padding: 10px 20px;
            border-radius: 5px;
            cursor: pointer;
            font-size: 16px;
        }
        .upload-form input[type="submit"]:hover {
            background-color: #4682b4;
        }
        .success-message {
            color: green;
            margin-top: 20px;
            font-size: 18px;
        }
        .links {
            margin-top: 10px;
        }
        .links a {
            color: #1e90ff;
            text-decoration: none;
            margin: 0 10px;
            font-size: 16px;
        }
        .links a:hover {
            text-decoration: underline;
        }
    </style>
</head>
<body>
    <div class="container">
        <h1>Bank Marketing Deposit Prediction</h1>
        <img src="{{ url_for('static', filename='gambar.png') }}" alt="Banner" style="width: 30%; height: auto; border-radius: 10px; margin-bottom: 20px;">
        <div class="upload-form">
            <form action="/prediksi" method="post" enctype="multipart/form-data">
                <input type="file" name="file" required>
                <input type="submit" value="Upload">
            </form>
        </div>
        {% if success %}
        <div class="success-message">
            <p>Upload berhasil!</p>
            <div class="links">
                <a href="{{ uploaded_file_url }}" target="_blank">Link ke dataset yang diupload</a>
                <a href="{{ download_url }}" id="download-link">Download hasil prediksi</a>
            </div>
        </div>
        <script>
            // Automatically trigger the download
            document.getElementById("download-link").click();
        </script>
        {% endif %}
    </div>
</body>
</html>
'''

@app.route('/')
def index():
    return render_template_string(HTML_TEMPLATE, success=False)

@app.route('/prediksi', methods=['POST'])
def prediksi():
    if 'file' not in request.files:
        return redirect(request.url)
    file = request.files['file']
    if file.filename == '':
        return redirect(request.url)
    if file:
        filename = file.filename
        file_path = os.path.join('uploads', filename)
        os.makedirs('uploads', exist_ok=True)
        file.save(file_path)
        output_path = predict(file_path)

        # Generate URLs for the uploaded file and the prediction result download link
        uploaded_file_url = url_for('static', filename=filename)
        download_url = url_for('download_file', filename=os.path.basename(output_path))

        return render_template_string(HTML_TEMPLATE, 
                                      success=True, 
                                      uploaded_file_url=uploaded_file_url, 
                                      download_url=download_url)

@app.route('/download/<filename>')
def download_file(filename):
    file_path = os.path.join('static', filename)
    return send_file(file_path, as_attachment=True)

def predict(file_path):
    # Membaca dataset
    df = pd.read_csv(file_path)

    # Preprocessing data
    df = df.drop_duplicates()
    df_encoded = pd.get_dummies(df, drop_first=True)
    target = 'deposit_yes'
    features = df_encoded.drop(columns=[target])
    X_train, X_test, y_train, y_test = train_test_split(features, df_encoded[target], test_size=0.2, random_state=42)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Load the trained model
    loaded_model = joblib.load('svm_model.joblib')

    # Prediction
    predictions = loaded_model.predict(X_test_scaled)

    predictions_binary = [1 if pred else 0 for pred in predictions]
    y_test_binary = [1 if val else 0 for val in y_test]

    output_df = pd.DataFrame({
        'nilai_asli': y_test_binary,
        'prediksi': predictions_binary
    })

    output_path = os.path.join('static', 'hasil_prediksi_svm.csv')
    output_df.to_csv(output_path, index=False)
    return output_path

if __name__ == '__main__':
    app.run(debug=True)
