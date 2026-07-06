from flask import Flask, render_template, request, redirect, url_for, session, flash
import pandas as pd
import numpy as np
import pickle
import os
from datetime import datetime
from werkzeug.security import generate_password_hash, check_password_hash

app = Flask(__name__)
app.secret_key = 'your_secret_key_here'  # Change this to a secure random key

# Create necessary directories
os.makedirs('model', exist_ok=True)
os.makedirs('history', exist_ok=True)
os.makedirs('static/css', exist_ok=True)
os.makedirs('static/images', exist_ok=True)
os.makedirs('screenshots', exist_ok=True)

# Demo user credentials (In production, use a database)
USERS = {
    'navya': generate_password_hash('navya1410')
}

# Load or create model
def load_model():
    model_path = 'model/hdi_model.pkl'
    encoder_path = 'model/label_encoder.pkl'
    
    if os.path.exists(model_path) and os.path.exists(encoder_path):
        with open(model_path, 'rb') as f:
            model = pickle.load(f)
        with open(encoder_path, 'rb') as f:
            encoder = pickle.load(f)
        return model, encoder
    else:
        return create_sample_model()

def create_sample_model():
    """Create a simple rule-based model for demonstration"""
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.preprocessing import LabelEncoder
    
    # Sample training data
    data = {
        'life_expectancy': [70, 80, 65, 75, 82, 60, 55, 78, 85, 50],
        'expected_schooling': [12, 16, 10, 14, 18, 8, 6, 15, 20, 5],
        'mean_schooling': [10, 13, 8, 11, 15, 6, 4, 12, 16, 3],
        'gni_per_capita': [15000, 45000, 8000, 25000, 55000, 5000, 3000, 35000, 65000, 2000],
        'hdi_tier': ['High', 'Very High', 'Medium', 'High', 'Very High', 'Medium', 'Low', 'High', 'Very High', 'Low']
    }
    
    df = pd.DataFrame(data)
    X = df[['life_expectancy', 'expected_schooling', 'mean_schooling', 'gni_per_capita']]
    y = df['hdi_tier']
    
    encoder = LabelEncoder()
    y_encoded = encoder.fit_transform(y)
    
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X, y_encoded)
    
    # Save model
    with open('model/hdi_model.pkl', 'wb') as f:
        pickle.dump(model, f)
    with open('model/label_encoder.pkl', 'wb') as f:
        pickle.dump(encoder, f)
    
    return model, encoder

# Initialize model
model, label_encoder = load_model()

# Initialize prediction history
def init_history():
    history_file = 'history/prediction_history.csv'
    if not os.path.exists(history_file):
        df = pd.DataFrame(columns=['timestamp', 'username', 'life_expectancy', 
                                   'expected_schooling', 'mean_schooling', 
                                   'gni_per_capita', 'predicted_tier'])
        df.to_csv(history_file, index=False)

init_history()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        
        if username in USERS and check_password_hash(USERS[username], password):
            session['username'] = username
            flash('Login successful!', 'success')
            return redirect(url_for('dashboard'))
        else:
            flash('Invalid credentials. Try username: harshitha, password: harshi1410', 'danger')
    
    return render_template('login.html')

@app.route('/logout')
def logout():
    session.clear()
    flash('You have been logged out.', 'info')
    return redirect(url_for('index'))

@app.route('/dashboard')
def dashboard():
    if 'username' not in session:
        return redirect(url_for('login'))
    return render_template('dashboard.html', username=session['username'])

@app.route('/predict', methods=['GET', 'POST'])
def predict():
    if 'username' not in session:
        return redirect(url_for('login'))
    
    if request.method == 'POST':
        try:
            life_expectancy = float(request.form['life_expectancy'])
            expected_schooling = float(request.form['expected_schooling'])
            mean_schooling = float(request.form['mean_schooling'])
            gni_per_capita = float(request.form['gni_per_capita'])
            
            # Validate inputs
            if not (0 <= life_expectancy <= 100):
                flash('Life Expectancy must be between 0 and 100', 'danger')
                return render_template('predict.html')
            
            if not (0 <= expected_schooling <= 25):
                flash('Expected Schooling must be between 0 and 25', 'danger')
                return render_template('predict.html')
            
            if not (0 <= mean_schooling <= 20):
                flash('Mean Schooling must be between 0 and 20', 'danger')
                return render_template('predict.html')
            
            if gni_per_capita < 0:
                flash('GNI per Capita must be positive', 'danger')
                return render_template('predict.html')
            
            # Make prediction
            input_data = np.array([[life_expectancy, expected_schooling, mean_schooling, gni_per_capita]])
            prediction_encoded = model.predict(input_data)[0]
            predicted_tier = label_encoder.inverse_transform([prediction_encoded])[0]
            
            # Save to history
            history_df = pd.read_csv('history/prediction_history.csv')
            new_record = {
                'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                'username': session['username'],
                'life_expectancy': life_expectancy,
                'expected_schooling': expected_schooling,
                'mean_schooling': mean_schooling,
                'gni_per_capita': gni_per_capita,
                'predicted_tier': predicted_tier
            }
            history_df = pd.concat([history_df, pd.DataFrame([new_record])], ignore_index=True)
            history_df.to_csv('history/prediction_history.csv', index=False)
            
            return render_template('result.html', 
                                 prediction=predicted_tier,
                                 life_expectancy=life_expectancy,
                                 expected_schooling=expected_schooling,
                                 mean_schooling=mean_schooling,
                                 gni_per_capita=gni_per_capita)
        
        except Exception as e:
            flash(f'Error: {str(e)}', 'danger')
            return render_template('predict.html')
    
    return render_template('predict.html')

@app.route('/history')
def history():
    if 'username' not in session:
        return redirect(url_for('login'))
    
    history_file = 'history/prediction_history.csv'
    if os.path.exists(history_file):
        df = pd.read_csv(history_file)
        user_history = df[df['username'] == session['username']].tail(10)
        return render_template('history.html', history=user_history.to_dict('records'))
    return render_template('history.html', history=[])

@app.route('/about')
def about():
    return render_template('about.html')

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
