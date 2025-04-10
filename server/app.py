from flask import Flask, request, jsonify, send_file  ,render_template
from flask_cors import CORS
from werkzeug.utils import secure_filename
import os
import shutil
from collections import defaultdict
from PIL import Image
# import tensorflow as tf
import numpy as np
import io

import os
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from flask import Flask, render_template, request, jsonify
from werkzeug.utils import secure_filename
# from tensorflow.keras.preprocessing import image
from flask import send_file, send_from_directory
app = Flask(__name__,template_folder="../pages", static_folder=os.path.join(os.path.pardir, "static"), static_url_path='/static')

CORS(app)
UPLOAD_FOLDER = "uploads"
MODEL_PATH = "model"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(MODEL_PATH, exist_ok=True)
model = None
class_names = []


EDA_FOLDER = os.path.join('static', 'eda_charts')
os.makedirs(EDA_FOLDER, exist_ok=True)


@app.route("/")
def index():
    return render_template("home.html")



from flask import request, jsonify
import os
from werkzeug.utils import secure_filename

UPLOAD_FOLDER = 'uploads'

@app.route("/upload", methods=["POST"])
def upload_dataset():
    os.makedirs(UPLOAD_FOLDER, exist_ok=True)

    uploaded_files = request.files.getlist("images")
    if not uploaded_files:
        return jsonify({"message": "No files received"}), 400

    for file in uploaded_files:
        if file.filename:
            parts = file.filename.split('/')
            if len(parts) == 2:
                class_name, filename = parts
            else:
                class_name, filename = 'Unknown', file.filename

            save_dir = os.path.join(UPLOAD_FOLDER, secure_filename(class_name))
            os.makedirs(save_dir, exist_ok=True)
            file.save(os.path.join(save_dir, secure_filename(filename)))

    return jsonify({"message": "Dataset uploaded successfully."})


def generate_eda_charts(df):
    chart_urls = []

    for f in os.listdir(EDA_FOLDER):
        if f.endswith('.png'):
            os.remove(os.path.join(EDA_FOLDER, f))

    # Histogram for numeric columns
    numeric_cols = df.select_dtypes(include='number').columns
    for col in numeric_cols:
        plt.figure(figsize=(6, 4))
        sns.histplot(df[col].dropna(), kde=True, bins=30, color='skyblue')
        plt.title(f"Histogram of {col}")
        plt.tight_layout()
        chart_path = os.path.join(EDA_FOLDER, f"hist_{col}.png")
        plt.savefig(chart_path)
        plt.close()
        chart_urls.append(f"/static/eda_charts/{os.path.basename(chart_path)}")

    # Correlation heatmap
    if len(numeric_cols) > 1:
        plt.figure(figsize=(8, 6))
        corr = df[numeric_cols].corr()
        sns.heatmap(corr, annot=True, cmap='coolwarm', fmt=".2f")
        plt.title("Correlation Heatmap")
        plt.tight_layout()
        heatmap_path = os.path.join(EDA_FOLDER, "correlation_heatmap.png")
        plt.savefig(heatmap_path)
        plt.close()
        chart_urls.append(f"/static/eda_charts/{os.path.basename(heatmap_path)}")

    # Missing value heatmap
    if df.isnull().sum().sum() > 0:
        plt.figure(figsize=(10, 5))
        sns.heatmap(df.isnull(), cbar=False, cmap="viridis")
        plt.title("Missing Values Heatmap")
        plt.tight_layout()
        nullmap_path = os.path.join(EDA_FOLDER, "missing_values.png")
        plt.savefig(nullmap_path)
        plt.close()
        chart_urls.append(f"/static/eda_charts/{os.path.basename(nullmap_path)}")

    return chart_urls


@app.route("/eda", methods=["GET", "POST"])
def eda():
    if request.method == 'GET':
        return render_template("eda.html")

    if request.method == 'POST':
        if 'dataset' not in request.files:
            return "No file part", 400

        file = request.files['dataset']

        if file.filename == '':
            return "No selected file", 400

        # Save uploaded file
        file_path = os.path.join(EDA_FOLDER, secure_filename(file.filename))
        file.save(file_path)

        # Load dataset
        try:
            if file.filename.endswith('.csv'):
                df = pd.read_csv(file_path)
            elif file.filename.endswith(('.xls', '.xlsx')):
                df = pd.read_excel(file_path)
            else:
                return "Unsupported file format", 400
        except Exception as e:
            return f"Error reading file: {e}", 500

        # Generate EDA charts
        chart_urls = generate_eda_charts(df)

        return render_template("eda.html", charts=chart_urls)

@app.route("/train", methods=["POST", "GET"])
def train():
    if request.method == 'GET':
        print("GET request received")
        return render_template("train.html")

    elif request.method == 'POST':
        try:
            UPLOAD_FOLDER = "uploads"
            img_height, img_width = 180, 180

            # Load data
            train_ds = tf.keras.utils.image_dataset_from_directory(
                UPLOAD_FOLDER,
                image_size=(img_height, img_width),
                batch_size=32
            )

            class_names = train_ds.class_names
            print("Detected classes:", class_names)

            # Build model
            model = tf.keras.Sequential([
                tf.keras.layers.Rescaling(1./255, input_shape=(img_height, img_width, 3)),
                tf.keras.layers.Conv2D(16, 3, activation='relu'),
                tf.keras.layers.MaxPooling2D(),
                tf.keras.layers.Conv2D(32, 3, activation='relu'),
                tf.keras.layers.MaxPooling2D(),
                tf.keras.layers.Conv2D(64, 3, activation='relu'),
                tf.keras.layers.MaxPooling2D(),
                tf.keras.layers.Flatten(),
                tf.keras.layers.Dense(128, activation='relu'),
                tf.keras.layers.Dense(len(class_names))
            ])

            model.compile(
                optimizer='adam',
                loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                metrics=['accuracy']
            )

            history = model.fit(train_ds, epochs=5)

            # Save model
            model_path = os.path.join("static", "image_model.h5")
            model.save(model_path)

            return jsonify({
                "success": True,
                "message": "Model trained and saved successfully.",
                "model_path": model_path
            })

        except Exception as e:
            print(f"Error during training: {e}")
            return jsonify({"success": False, "message": str(e)})

    # def train():
    # # Define your dataset folder (same as UPLOAD_FOLDER)
    #     UPLOAD_FOLDER = "uploads"
    #     img_height, img_width = 180, 180

    # # Load training data
    #     train_ds = tf.keras.utils.image_dataset_from_directory(
    #         UPLOAD_FOLDER,
    #         image_size=(img_height, img_width),
    #         batch_size=32
    #     )
    #     class_names = train_ds.class_names
    #     print("Detected classes:", class_names)
    # # Build and train model
    #     model = tf.keras.Sequential([
    #         tf.keras.layers.Rescaling(1./255, input_shape=(img_height, img_width, 3)),
    #         tf.keras.layers.Conv2D(16, 3, activation='relu'),
    #         tf.keras.layers.MaxPooling2D(),
    #         tf.keras.layers.Conv2D(32, 3, activation='relu'),
    #         tf.keras.layers.MaxPooling2D(),
    #         tf.keras.layers.Conv2D(64, 3, activation='relu'),
    #         tf.keras.layers.MaxPooling2D(),
    #         tf.keras.layers.Flatten(),
    #         tf.keras.layers.Dense(128, activation='relu'),
    #         tf.keras.layers.Dense(len(class_names))
    #     ])
    #     model.compile(
    #         optimizer='adam',
    #         loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    #         metrics=['accuracy']
    #     )

    #     model.fit(train_ds, epochs=5)

    # # Save the model (optional)
    #     model.save("image_model.h5")
    #     return jsonify({"message": "Model trained and saved successfully."})
    

@app.route("/export", methods=["POST"])
def export_model():
    try:
        model_path = os.path.join("static", "image_model.h5")
        export_dir = os.path.join("static", "tfjs_model")
        os.makedirs(export_dir, exist_ok=True)

        model = tf.keras.models.load_model(model_path)

        # Convert and save to TFJS format
        import tensorflowjs as tfjs
        tfjs.converters.save_keras_model(model, export_dir)

        # Zip the folder
        zip_path = os.path.join("static", "tfjs_model.zip")
        shutil.make_archive(zip_path.replace(".zip", ""), 'zip', export_dir)

        return jsonify({
            "success": True,
            "message": "Model exported successfully.",
            "download_url": "/download/tfjs_model.zip"
        })

    except Exception as e:
        return jsonify({"success": False, "message": str(e)})

@app.route("/download/<filename>")
def download_model(filename):
    return send_from_directory("static", filename, as_attachment=True)

if __name__ == "__main__":
    app.run(debug=True)
