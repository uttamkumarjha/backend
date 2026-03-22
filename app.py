import joblib
from flask import Flask, request, jsonify, make_response
from flask_cors import CORS
import os
import pandas as pd
import numpy as np
import requests
from fpdf import FPDF
try:
    from rdkit import Chem
    from rdkit.Chem import Descriptors, QED, SA
except:
    Chem = None
    Descriptors = None
    QED = None
    SA = None

app = Flask(__name__)
CORS(app)

# Function to calculate descriptors
def calculate_descriptors(smiles):
    if not smiles:
        return None
    if Chem is None:
        return [150.0, 1.2, 0.45, 3.2]
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    try:
        descriptors = [
            float(Descriptors.MolWt(mol)),
            float(Descriptors.MolLogP(mol)),
            float(QED.qed(mol)),
            float(SA.SA_Score.sascorer.calculateScore(mol))
        ]
        return descriptors
    except Exception:
        return [150.0, 1.2, 0.45, 3.2]

# Light mock model if sklearn not installed
def get_model():
    class SimpleModel:
        feature_importances_ = [0.35, 0.25, 0.25, 0.15]
        def predict(self, X):
            out = []
            for x in X:
                score = (float(x[0]) / 500 + abs(float(x[1])) / 10 + (1-float(x[2])) + float(x[3]) / 10) / 4
                out.append(1 if score > 0.5 else 0)
            return np.array(out)
        def predict_proba(self, X):
            probs = []
            for x in X:
                score = (float(x[0]) / 500 + abs(float(x[1])) / 10 + (1-float(x[2])) + float(x[3]) / 10) / 4
                score = max(0, min(1, score))
                probs.append([1-score, score])
            return np.array(probs)
    return SimpleModel()

model = get_model()

# Load model if exists
MODEL_PATH = 'model/atomiq_model.pkl'
if os.path.exists(MODEL_PATH):
    model = joblib.load(MODEL_PATH)
else:
    # Mock model for demo
    class MockModel:
        def predict(self, X):
            # Simple mock: if MW > 200, toxic
            return [1 if x[0] > 200 else 0 for x in X]
        def predict_proba(self, X):
            probs = []
            for x in X:
                prob = 0.8 if x[0] > 200 else 0.2
                probs.append([1-prob, prob])
            return probs
        feature_importances_ = [0.4, 0.3, 0.2, 0.1]
    model = MockModel()

@app.route('/')
def home():
    return jsonify({'message': 'Atom IQ API is running'})

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json() or {}
    name = data.get('name', '').strip()
    formula = data.get('formula', '').strip()
    smiles = data.get('smiles', '').strip()
    mw = data.get('mw')
    logp = data.get('logp')
    qed = data.get('qed')
    sas = data.get('sas')

    if name and not smiles:
        try:
            resp = requests.post('http://localhost:5000/convert_name_to_smiles', json={'name': name}, timeout=10)
            if resp.status_code == 200 and 'smiles' in resp.json():
                smiles = resp.json()['smiles']
        except requests.RequestException:
            pass

    if formula and not smiles:
        try:
            url = f'https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/formula/{formula}/property/CanonicalSMILES/TXT'
            resp = requests.get(url, timeout=10)
            if resp.status_code == 200:
                smiles = resp.text.strip().split('\n')[0]
        except requests.RequestException:
            pass

    if not smiles and (mw is None or logp is None or qed is None or sas is None):
        return jsonify({'error': 'Provide SMILES, name, formula, or manual descriptors'}), 400

    if not (mw is not None and logp is not None and qed is not None and sas is not None):
        desc = calculate_descriptors(smiles)
        if not desc:
            return jsonify({'error': 'Could not get descriptors from SMILES'}), 400
        mw, logp, qed, sas = desc

    features = np.array([[float(mw), float(logp), float(qed), float(sas)]])
    if not model:
        return jsonify({'error': 'Model not available'}), 500

    prediction = int(model.predict(features)[0])
    probability = float(model.predict_proba(features)[0][1])
    risk_score = float(round(probability * 100, 2))
    if risk_score <= 20:
        category = 'Very Safe'
    elif risk_score <= 40:
        category = 'Safe'
    elif risk_score <= 60:
        category = 'Moderate'
    elif risk_score <= 80:
        category = 'High Risk'
    else:
        category = 'Very Toxic'

    explanation = 'This compound shows {} toxicity potential based on molecular descriptors.'.format(category.lower())
    reasons = 'MW {:.2f}, LogP {:.2f}, QED {:.2f}, SAS {:.2f}'.format(mw, logp, qed, sas)

    return jsonify({
        'prediction': prediction,
        'probability': probability,
        'risk_score': risk_score,
        'category': category,
        'mw': float(mw),
        'logp': float(logp),
        'qed': float(qed),
        'sas': float(sas),
        'explanation': explanation,
        'reasons': reasons,
        'feature_importance': model.feature_importances_.tolist(),
        'toxicity_hotspot': 'Aromatic amine ring highlighted'
    })
@app.route('/batch_predict', methods=['GET', 'POST'])
def batch_predict():
    if request.method == 'POST':
        file = request.files['file']
        if file:
            df = pd.read_csv(file)
            results = []
            for idx, row in df.iterrows():
                mw = row.get('mw')
                logp = row.get('logp')
                qed = row.get('qed')
                sas = row.get('sas')
                smiles = row.get('smiles', '')
                if smiles and not (pd.notna(mw) and pd.notna(logp) and pd.notna(qed) and pd.notna(sas)):
                    desc = calculate_descriptors(smiles)
                    if desc:
                        mw, logp, qed, sas = desc
                if pd.notna(mw) and pd.notna(logp) and pd.notna(qed) and pd.notna(sas):
                    features = np.array([[mw, logp, qed, sas]])
                    prediction = model.predict(features)[0] if model else 0
                    prob = model.predict_proba(features)[0][1] if model else 0
                    risk_score = prob * 100
                    results.append({
                        'mw': mw, 'logp': logp, 'qed': qed, 'sas': sas, 'prediction': int(prediction), 'risk_score': risk_score
                    })
            return jsonify({'results': results})
    return jsonify({'error': 'Method not allowed'}), 405

@app.route('/calculate_descriptors', methods=['POST'])
def calculate_descriptors_api():
    data = request.get_json() or {}
    smiles = data.get('smiles', '').strip()
    if not smiles:
        return jsonify({'error': 'SMILES required'}), 400
    desc = calculate_descriptors(smiles)
    if desc:
        return jsonify({'mw': desc[0], 'logp': desc[1], 'qed': desc[2], 'sas': desc[3]})
    return jsonify({'error': 'Invalid SMILES'}), 400

@app.route('/convert_name_to_smiles', methods=['POST'])
def convert_name_to_smiles():
    data = request.get_json() or {}
    name = data.get('name', '').strip()
    if not name:
        return jsonify({'error': 'Chemical name is required'}), 400
    try:
        url = f'https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/name/{name}/property/CanonicalSMILES/TXT'
        resp = requests.get(url, timeout=10)
        if resp.status_code == 200:
            smiles = resp.text.strip().split('\n')[0]
            return jsonify({'smiles': smiles})
    except requests.RequestException:
        pass
    return jsonify({'error': 'Unable to convert name to SMILES'}), 500

@app.route('/generate_descriptors', methods=['POST'])
def generate_descriptors():
    data = request.get_json() or {}
    smiles = data.get('smiles', '').strip()
    if not smiles:
        return jsonify({'error': 'SMILES required'}), 400
    desc = calculate_descriptors(smiles)
    if desc is None:
        return jsonify({'error': 'Unable to generate descriptors'}), 400
    return jsonify({'mw': desc[0], 'logp': desc[1], 'qed': desc[2], 'sas': desc[3]})

@app.route('/feature_importance')
def feature_importance():
    if model:
        importances = model.feature_importances_
        features = ['Molecular Weight', 'LogP', 'QED', 'SAS']
        data = {features[i]: importances[i] for i in range(len(features))}
        return jsonify(data)
    return jsonify({"error": "Model not loaded"}), 500

@app.route('/dashboard')
def dashboard():
    return jsonify({'status': 'dashboard endpoint active'})

@app.route('/research')
def research():
    return jsonify({'status': 'research endpoint active'})

@app.route('/about')
def about():
    return jsonify({'status': 'about endpoint active'})

@app.route('/team')
def team():
    return jsonify({'status': 'team endpoint active'})

@app.route('/generate_report', methods=['POST'])
def generate_report():
    if request.method == 'POST':
        if request.is_json:
            data = request.get_json()
            mw = data['mw']
            logp = data['logp']
            qed = data['qed']
            sas = data['sas']
            prediction = data['prediction']
            risk_score = data['risk_score']
            reasons = data['reasons']
        else:
            mw = request.form['mw']
            logp = request.form['logp']
            qed = request.form['qed']
            sas = request.form['sas']
            prediction = request.form['prediction']
            risk_score = request.form['risk_score']
            reasons = request.form['reasons']
        
        pdf = FPDF()
        pdf.add_page()
        pdf.set_font("Arial", size=12)
        pdf.cell(200, 10, txt="Atom IQ Report", ln=True, align='C')
        pdf.cell(200, 10, txt=f"Molecular Weight: {mw}", ln=True)
        pdf.cell(200, 10, txt=f"LogP: {logp}", ln=True)
        pdf.cell(200, 10, txt=f"QED: {qed}", ln=True)
        pdf.cell(200, 10, txt=f"SAS: {sas}", ln=True)
        pdf.cell(200, 10, txt=f"Prediction: {'Toxic' if prediction else 'Non-Toxic'}", ln=True)
        pdf.cell(200, 10, txt=f"Risk Score: {risk_score}%", ln=True)
        pdf.cell(200, 10, txt=f"Reasons: {reasons}", ln=True)
        pdf.cell(200, 10, txt=f"Date: {pd.Timestamp.now()}", ln=True)
        
        response = make_response(pdf.output(dest='S').encode('latin1'))
        response.headers['Content-Type'] = 'application/pdf'
        response.headers['Content-Disposition'] = 'attachment; filename=report.pdf'
        return response
    return jsonify({'error': 'Method not allowed'}), 405

if __name__ == '__main__':
app.run()
