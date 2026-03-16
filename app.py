import os
import secrets
import numpy as np
from datetime import datetime

from flask import Flask, render_template, request, redirect, url_for, flash
from flask_sqlalchemy import SQLAlchemy
from flask_login import LoginManager, UserMixin, login_user, login_required, logout_user, current_user
from werkzeug.security import generate_password_hash, check_password_hash
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image

app = Flask(__name__)
app.config['SECRET_KEY'] = secrets.token_hex(16)
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///users.db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
app.config['UPLOAD_FOLDER'] = 'static/uploads'
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

db = SQLAlchemy(app)
login_manager = LoginManager(app)
login_manager.login_view = 'login'


class User(UserMixin, db.Model):
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(100), nullable=False)
    email = db.Column(db.String(100), unique=True, nullable=False)
    username = db.Column(db.String(50), unique=True, nullable=False)
    password_hash = db.Column(db.String(200), nullable=False)
    mobile = db.Column(db.String(20))
    address = db.Column(db.Text)
    predictions = db.relationship('Prediction', backref='user', lazy=True)

    def set_password(self, password):
        self.password_hash = generate_password_hash(password)

    def check_password(self, password):
        return check_password_hash(self.password_hash, password)

class Prediction(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)
    image_path = db.Column(db.String(200), nullable=False)
    disease = db.Column(db.String(100), nullable=False)
    confidence = db.Column(db.Float, nullable=False)
    remedies = db.Column(db.Text, nullable=False)
    timestamp = db.Column(db.DateTime, default=datetime.utcnow)

@login_manager.user_loader
def load_user(user_id):
    return User.query.get(int(user_id))

# Load trained model and class names

MODEL_PATH = "model/Xception_rice_disease.h5"
DATASET_PATH = "Dataset/train"   # folder containing class subfolders

try:
    model = load_model(MODEL_PATH)
    print("Model loaded successfully.")
except Exception as e:
    print(f"Error loading model: {e}")
    model = None

# Get class names (sorted to match model output)
if os.path.exists(DATASET_PATH):
    classes = sorted([d for d in os.listdir(DATASET_PATH) if os.path.isdir(os.path.join(DATASET_PATH, d))])
else:
    # Fallback: define classes manually (must match your training)
    classes = ["Rice_Bacterialblight", "Rice_Blast", "Rice_Brownspot", "Rice_Tungro"]
    print("Warning: Dataset path not found, using fallback class list.")

# Disease information with remedies
disease_info = {
    "Rice_Bacterialblight": {
        "name": "Bacterial Blight",
        "remedy": "Apply copper-based bactericides. Use resistant varieties. Avoid nitrogen overuse."
    },
    "Rice_Blast": {
        "name": "Blast Disease",
        "remedy": "Use fungicides like tricyclazole or carbendazim. Maintain optimal water levels."
    },
    "Rice_Brownspot": {
        "name": "Brown Spot",
        "remedy": "Apply fungicides (mancozeb). Ensure balanced nutrition, especially potassium."
    },
    "Rice_Tungro": {
        "name": "Tungro Disease",
        "remedy": "Control leafhopper vectors with insecticides. Remove infected plants."
    }
}

def predict_disease(img_path):
    """Return (disease_name, confidence, remedy) for the given image."""
    if model is None:
        return "Model not available", 0.0, "Please contact administrator."

    img = image.load_img(img_path, target_size=(224, 224))
    img = image.img_to_array(img) / 255.0
    img = np.expand_dims(img, axis=0)

    pred = model.predict(img)
    class_idx = np.argmax(pred)
    folder_name = classes[class_idx]
    confidence = round(100 * np.max(pred), 2)

    info = disease_info.get(folder_name, {
        "name": "Unknown",
        "remedy": "Consult an agricultural expert."
    })
    disease_name = info["name"]
    remedy = info["remedy"]

    return disease_name, confidence, remedy

@app.route('/')
def index():
    """Home page."""
    return render_template('index.html')

@app.route('/register', methods=['GET', 'POST'])
def register():
    """User registration."""
    if request.method == 'POST':
        name = request.form['name']
        email = request.form['email']
        username = request.form['username']
        password = request.form['password']
        mobile = request.form.get('mobile', '')
        address = request.form.get('address', '')

        # Check if user already exists
        if User.query.filter_by(username=username).first():
            flash('Username already exists', 'danger')
            return redirect(url_for('register'))
        if User.query.filter_by(email=email).first():
            flash('Email already registered', 'danger')
            return redirect(url_for('register'))

        user = User(name=name, email=email, username=username,
                    mobile=mobile, address=address)
        user.set_password(password)
        db.session.add(user)
        db.session.commit()

        flash('Registration successful! Please log in.', 'success')
        return redirect(url_for('login'))

    return render_template('register.html')

@app.route('/login', methods=['GET', 'POST'])
def login():
    """User login."""
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        user = User.query.filter_by(username=username).first()
        if user and user.check_password(password):
            login_user(user)
            flash('Login successful!', 'success')
            return redirect(url_for('dashboard'))
        else:
            flash('Invalid username or password', 'danger')
    return render_template('login.html')

@app.route('/logout')
@login_required
def logout():
    """User logout."""
    logout_user()
    flash('You have been logged out.', 'info')
    return redirect(url_for('index'))

@app.route('/dashboard')
@login_required
def dashboard():
    """User dashboard with recent predictions."""
    recent = Prediction.query.filter_by(user_id=current_user.id)\
                .order_by(Prediction.timestamp.desc()).limit(5).all()
    total = Prediction.query.filter_by(user_id=current_user.id).count()
    return render_template('dashboard.html', recent=recent, total=total)

@app.route('/predict', methods=['GET', 'POST'])
@login_required
def predict():
    """Upload image for disease prediction."""
    if request.method == 'POST':
        if 'image' not in request.files:
            flash('No file part', 'danger')
            return redirect(request.url)
        file = request.files['image']
        if file.filename == '':
            flash('No selected file', 'danger')
            return redirect(request.url)

        # Save file with a unique name
        filename = secrets.token_hex(8) + '_' + file.filename
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)

        # Predict
        disease, confidence, remedy = predict_disease(filepath)

        # Save prediction to database
        pred = Prediction(
            user_id=current_user.id,
            image_path=filepath,
            disease=disease,
            confidence=confidence,
            remedies=remedy
        )
        db.session.add(pred)
        db.session.commit()

        return redirect(url_for('result', pred_id=pred.id))

    return render_template('predict.html')

@app.route('/result/<int:pred_id>')
@login_required
def result(pred_id):
    """Show a single prediction result."""
    pred = Prediction.query.get_or_404(pred_id)
    if pred.user_id != current_user.id:
        flash('Unauthorized access', 'danger')
        return redirect(url_for('dashboard'))
    return render_template('result.html', pred=pred)

@app.route('/history')
@login_required
def history():
    """List all predictions of the current user with pagination."""
    page = request.args.get('page', 1, type=int)
    predictions = Prediction.query.filter_by(user_id=current_user.id)\
                    .order_by(Prediction.timestamp.desc())\
                    .paginate(page=page, per_page=10, error_out=False)
    return render_template('history.html', predictions=predictions)

@app.route('/analysis')
@login_required
def analysis():
    from sqlalchemy import func
    disease_counts = db.session.query(
        Prediction.disease,
        func.count(Prediction.disease)
    ).filter_by(user_id=current_user.id) \
     .group_by(Prediction.disease).all()

    labels = [item[0] for item in disease_counts]
    data = [item[1] for item in disease_counts]
    disease_data = list(zip(labels, data))   # list of (disease, count)
    total = sum(data)

    return render_template(
        'analysis.html',
        labels=labels,
        data=data,
        disease_data=disease_data,
        total=total
    )

with app.app_context():
    db.create_all()
    print("Database tables created/verified.")

if __name__ == "__main__":
    app.run(debug=True)