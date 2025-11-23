import os
import io
import numpy as np
import tensorflow as tf
from flask import Flask, render_template, request, jsonify, send_file, url_for
from werkzeug.utils import secure_filename
from PIL import Image
import cv2
from pathlib import Path

# Keras imports for model reconstruction
from tensorflow.keras.optimizers import Adamax
from tensorflow.keras.metrics import Precision, Recall
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Flatten, Dropout, Dense

# ReportLab for PDF generation
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image as RLImage, Table, TableStyle
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib import colors
from datetime import datetime

# --- CONFIG ---
app = Flask(__name__)
IMG_SIZE = (128, 128)
UPLOAD_FOLDER = Path('static/uploads')
MODEL_PATH = Path('brain_tumor_xception.h5')
CLASSES = ['glioma', 'meningioma', 'notumor', 'pituitary']

UPLOAD_FOLDER.mkdir(parents=True, exist_ok=True)
app.config['UPLOAD_FOLDER'] = str(UPLOAD_FOLDER)

# --- MODEL RECONSTRUCTION ---
def create_initial_model(img_shape, num_classes, lr=0.001):
    base_model = tf.keras.applications.Xception(
        include_top=False, weights='imagenet', input_shape=img_shape, pooling='max'
    )
    model = Sequential([
        base_model,
        Flatten(),
        Dropout(0.3),
        Dense(128, activation='relu'),
        Dropout(0.25),
        Dense(num_classes, activation='softmax')
    ])
    model.compile(optimizer=Adamax(learning_rate=lr),
                  loss='categorical_crossentropy',
                  metrics=['accuracy', Precision(), Recall()])
    return model

def load_weights_and_recompile(model_path, img_size, num_classes):
    img_shape = (img_size[0], img_size[1], 3)
    model = create_initial_model(img_shape, num_classes)
    model.load_weights(str(model_path))
    return model

# --- LOAD MODEL ---
model = None
grad_model = None
try:
    model = load_weights_and_recompile(MODEL_PATH, IMG_SIZE, len(CLASSES))
    print("Model loaded successfully")
    LAST_CONV_LAYER_NAME = 'block14_sepconv2_act'
    xception_base = model.get_layer('xception')
    last_conv_layer = xception_base.get_layer(LAST_CONV_LAYER_NAME)
    feature_extractor = Model(inputs=xception_base.input, outputs=last_conv_layer.output)
    input_tensor = model.inputs[0]
    conv_output = feature_extractor(input_tensor)
    prediction_output = model(input_tensor)
    grad_model = Model(inputs=input_tensor, outputs=[conv_output, prediction_output])
    print(f"Grad-CAM model ready ({LAST_CONV_LAYER_NAME})")
except Exception as e:
    print(f"Error loading model or Grad-CAM: {e}")

# --- GRAD-CAM ---
def make_gradcam_heatmap(img_array, grad_model, pred_index=None):
    img_array = tf.convert_to_tensor(img_array, dtype=tf.float32)
    with tf.GradientTape() as tape:
        tape.watch(img_array)
        conv_outputs, preds = grad_model(img_array)
        if pred_index is None:
            pred_index = tf.argmax(preds[0])
        loss = preds[:, pred_index]
    grads = tape.gradient(loss, conv_outputs)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    conv_outputs = conv_outputs[0]
    heatmap = tf.zeros(conv_outputs.shape[0:2], dtype=tf.float32)
    for i in range(int(pooled_grads.shape[-1])):
        heatmap += pooled_grads[i] * conv_outputs[:, :, i]
    heatmap = tf.maximum(heatmap, 0)
    max_val = tf.reduce_max(heatmap)
    if max_val == 0:
        return None
    heatmap /= max_val
    return heatmap.numpy()

def generate_gradcam_overlay(img_path, grad_model, out_path):
    try:
        img = Image.open(str(img_path)).convert('RGB').resize(IMG_SIZE)
        img_arr = np.asarray(img, dtype=np.float32) / 255.0
        inp = np.expand_dims(img_arr, axis=0)
        preds = grad_model(inp)[1].numpy()[0]
        pred_idx = int(np.argmax(preds))
        heatmap = make_gradcam_heatmap(inp, grad_model, pred_index=pred_idx)
        if heatmap is None:
            img.save(str(out_path))
            return str(out_path)
        heatmap_resized = cv2.resize((heatmap*255).astype('uint8'), IMG_SIZE, interpolation=cv2.INTER_LINEAR)
        heatmap_color = cv2.applyColorMap(heatmap_resized, cv2.COLORMAP_JET)
        orig_bgr = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
        overlay = cv2.addWeighted(orig_bgr, 0.55, heatmap_color, 0.45, 0)
        cv2.imwrite(str(out_path), overlay)
        return str(out_path)
    except Exception as e:
        print(f"Grad-CAM failed for {img_path}: {e}")
        img = Image.open(str(img_path)).convert('RGB').resize(IMG_SIZE)
        img.save(str(out_path))
        return str(out_path)

# --- PREDICTION ---
def predict_image(img_path):
    img = Image.open(str(img_path)).convert('RGB').resize(IMG_SIZE)
    img_array = np.asarray(img, dtype=np.float32)/255.0
    input_tensor = np.expand_dims(img_array, axis=0)
    probs = model.predict(input_tensor)[0]
    pred_idx = np.argmax(probs)
    pred_label = CLASSES[pred_idx]
    confidence = float(probs[pred_idx])*100
    all_probs = {CLASSES[i]: float(probs[i])*100 for i in range(len(CLASSES))}
    return pred_label, confidence, all_probs

# --- PDF REPORT ---
def generate_pdf_report(data):
    buffer = io.BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=letter)
    styles = getSampleStyleSheet()
    story = []
    story.append(Paragraph("Brain Tumor Classification Report", styles['Title']))
    story.append(Spacer(1, 12))
    patient_data = [
        ['Date of Report:', datetime.now().strftime("%Y-%m-%d %H:%M:%S")],
        ['Patient Name:', data['patient_info']['Name'] or 'N/A'],
        ['Age:', data['patient_info']['Age'] or 'N/A'],
        ['Gender:', data['patient_info']['Gender'] or 'N/A']
    ]
    t = Table(patient_data, colWidths=[150, 300])
    t.setStyle(TableStyle([('BACKGROUND', (0,0), (-1,0), colors.lightgrey),
                           ('GRID', (0,0), (-1,-1), 0.5, colors.black)]))
    story.append(Paragraph("Patient Information", styles['Heading2']))
    story.append(t)
    story.append(Spacer(1,12))
    story.append(Paragraph("Prediction Summary", styles['Heading2']))
    story.append(Paragraph(f"<b>Predicted Diagnosis:</b> <font color='{'red' if data['prediction']['predicted_label'] != 'normal' else 'green'}'>{data['prediction']['predicted_label']}</font>", styles['Normal']))
    story.append(Paragraph(f"<b>Confidence:</b> {data['prediction']['confidence']:.2f}%", styles['Normal']))
    story.append(Spacer(1,12))
    prob_data = [['Tumor Type', 'Probability (%)']]
    sorted_probs = sorted(data['prediction']['all_probs'].items(), key=lambda item: item[1], reverse=True)
    for label, prob in sorted_probs:
        prob_data.append([label, f"{prob:.2f}%"])
    t_prob = Table(prob_data, colWidths=[225,225])
    t_prob.setStyle(TableStyle([('BACKGROUND',(0,0),(-1,0),colors.lightgrey),
                                ('GRID',(0,0),(-1,-1),0.5,colors.black)]))
    story.append(Paragraph("Detailed Probabilities", styles['Heading3']))
    story.append(t_prob)
    story.append(Spacer(1,24))
    story.append(Paragraph("Visual Analysis", styles['Heading2']))
    try:
        img_path = Path(app.root_path) / Path(data['image_url'].lstrip('/'))
        img = RLImage(str(img_path), width=200, height=200)
        story.append(Paragraph("Original MRI Image:", styles['Normal']))
        story.append(img)
    except: pass
    story.append(Spacer(1,12))
    try:
        gradcam_path = Path(app.root_path) / Path(data['gradcam_url'].lstrip('/'))
        gradcam_img = RLImage(str(gradcam_path), width=200, height=200)
        story.append(Paragraph("Grad-CAM Activation Map (Model Focus):", styles['Normal']))
        story.append(gradcam_img)
    except: pass
    story.append(Spacer(1,36))
    story.append(Paragraph("Disclaimer: This report is generated by AI and is for informational purposes only. Not a substitute for professional medical advice.", styles['Italic']))
    doc.build(story)
    buffer.seek(0)
    return buffer

# --- FLASK ROUTES ---
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if not model or not grad_model:
        return jsonify({'error':'AI model not loaded'}),500
    if 'file' not in request.files:
        return jsonify({'error':'No file part'}),400
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error':'No selected file'}),400

    # Safe filename
    filename = secure_filename(file.filename).replace(" ", "_")
    unique_filename = f"{datetime.now().strftime('%Y%m%d%H%M%S')}_{filename}"
    img_path = UPLOAD_FOLDER / unique_filename
    file.save(str(img_path))

    try:
        pred_label, confidence, all_probs = predict_image(img_path)
        gradcam_filename = f"gradcam_{unique_filename}"
        gradcam_path = UPLOAD_FOLDER / gradcam_filename
        generate_gradcam_overlay(img_path, grad_model, gradcam_path)
        image_url = url_for('static', filename=f'uploads/{unique_filename}')
        gradcam_url = url_for('static', filename=f'uploads/{gradcam_filename}')
        return jsonify({
            'predicted_label': pred_label,
            'confidence': round(confidence,2),
            'all_probs': all_probs,
            'image_url': image_url,
            'gradcam_url': gradcam_url,
            'error': None
        })
    except Exception as e:
        if img_path.exists(): img_path.unlink()
        return jsonify({'error': f'Prediction failed: {str(e)}'}),500

@app.route('/download_pdf', methods=['POST'])
def download_pdf():
    data = request.get_json()
    if not data:
        return jsonify({'error':'No data for PDF'}),400
    try:
        pdf_buffer = generate_pdf_report(data)
        return send_file(pdf_buffer, as_attachment=True, download_name='patient_report.pdf', mimetype='application/pdf')
    except Exception as e:
        return jsonify({'error': f'Failed to generate PDF: {str(e)}'}),500

if __name__ == '__main__':
    UPLOAD_FOLDER.mkdir(parents=True, exist_ok=True)
    app.run(debug=True)
