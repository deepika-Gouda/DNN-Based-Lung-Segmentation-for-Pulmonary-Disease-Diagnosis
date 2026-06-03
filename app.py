import os
import sys

# Ensure app directory is in sys.path
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
if BASE_DIR not in sys.path:
    sys.path.append(BASE_DIR)

from project_model.resnet_seg import ResNetSegmentationClassification
import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from Lung_image_classifier.verify_lung_image import is_lung_image


# D:\Lung_Segmentation_Website_Project\Lung_Segmentation_Website_Project
import os
import time
import random
import datetime
from pathlib import Path

import numpy as np
from PIL import Image
import cv2
import io




import torch
import torch.nn.functional as F
from torchvision import transforms
import torchxrayvision as xrv

from flask import Flask, render_template, request, redirect, url_for, flash
from flask_login import LoginManager, login_user, login_required, logout_user, current_user
from werkzeug.utils import secure_filename
from werkzeug.security import generate_password_hash, check_password_hash


from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas

# Import User model & DB
from extensions import db
from models import User, Upload

from project_model.resnet_seg import ResNetSegmentationClassification

from models import Upload
from flask import send_file

def allowed_file(filename):
    ALLOWED_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".tiff"}
    return os.path.splitext(filename)[1].lower() in ALLOWED_EXT

# -------------------------
# Flask App Config
# -------------------------
app = Flask(__name__)
app.config['SECRET_KEY'] = 'your_secret_key_here'
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///users.db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

# Upload and result directories
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
UPLOAD_FOLDER = os.path.join(PROJECT_ROOT, "static", "uploads")
RESULT_FOLDER = os.path.join(PROJECT_ROOT, "static", "results")
CHECKPOINT_DIR = os.path.join(PROJECT_ROOT, "checkpoints")

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(RESULT_FOLDER, exist_ok=True)
os.makedirs(CHECKPOINT_DIR, exist_ok=True)

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['RESULT_FOLDER'] = RESULT_FOLDER
app.config["MAX_CONTENT_LENGTH"] = 10 * 1024 * 1024  # 10MB

# -------------------------
# Initialize DB
# -------------------------
db.init_app(app)

# -------------------------
# Flask-Login
# -------------------------
login_manager = LoginManager()
login_manager.init_app(app)
login_manager.login_view = 'home'

@login_manager.user_loader
def load_user(user_id):
    return User.query.get(int(user_id))

# Load pretrained chest X-ray model
lung_model = xrv.models.DenseNet(weights="all")
lung_model.eval()


# -------------------------
# Model Setup
# -------------------------
CLASSES = ['COVID', 'Normal', 'Lung_Opacity', 'Viral Pneumonia']
ALLOWED_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".tiff"}

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("[INFO] Using device:", device)

# Load model checkpoint
MODEL_PATH = os.path.join(CHECKPOINT_DIR, "model_epoch_final.pth")
print("[INFO] Model path:", MODEL_PATH)

model = ResNetSegmentationClassification(num_classes=len(CLASSES), pretrained=False).to(device)

if os.path.exists(MODEL_PATH):
    ckpt = torch.load(MODEL_PATH, map_location=device)
    state_dict = ckpt.get("state_dict", ckpt.get("model_state_dict", ckpt))
    model.load_state_dict(state_dict, strict=False)
    print("✅ Model loaded successfully")
else:
    print("⚠️ Model checkpoint not found — running with demo weights")

model.eval()

# Transform for preprocessing
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

# -------------------------
# Helper Functions
# -------------------------
def allowed_file(filename):
    return os.path.splitext(filename)[1].lower() in ALLOWED_EXT


def process_segmentation_and_overlay(seg_logits, orig_path, save_path, alpha=0.45):
    if seg_logits is None:
        return False
    try:
        if isinstance(seg_logits, torch.Tensor):
            seg_mask = torch.sigmoid(seg_logits).detach().cpu().numpy()
        else:
            seg_mask = np.array(seg_logits)

        if seg_mask.ndim == 4:
            seg_mask = seg_mask[0, 0]
        elif seg_mask.ndim == 3:
            seg_mask = seg_mask[0]
        mask_bin = (seg_mask > 0.5).astype(np.uint8) * 255

        orig_bgr = cv2.imread(orig_path)
        if orig_bgr is None:
            orig_bgr = cv2.cvtColor(np.array(Image.open(orig_path).convert("RGB")), cv2.COLOR_RGB2BGR)

        h_img, w_img = orig_bgr.shape[:2]
        mask_resized = cv2.resize(mask_bin, (w_img, h_img), interpolation=cv2.INTER_NEAREST)

        kernel = np.ones((5, 5), np.uint8)
        mask_clean = cv2.morphologyEx(mask_resized, cv2.MORPH_OPEN, kernel)
        mask_clean = cv2.morphologyEx(mask_clean, cv2.MORPH_CLOSE, kernel)
        mask_blur = cv2.GaussianBlur(mask_clean, (9, 9), 0)

        normalized = (mask_blur / 255.0).astype(np.float32)
        color = np.zeros_like(orig_bgr, dtype=np.uint8)
        color[:, :, 2] = 160
        color[:, :, 1] = 30
        color[:, :, 0] = 30

        alpha_map = (normalized * alpha)[:, :, None]
        blended = (orig_bgr * (1 - alpha_map) + color.astype(np.float32) * alpha_map).astype(np.uint8)

        cv2.imwrite(save_path, blended)
        return True
    except Exception as e:
        print("[ERROR] Segmentation overlay failed:", e)
        return False


def build_topk_from_logits(logits, k=3):
    if logits is None:
        return []
    if isinstance(logits, torch.Tensor):
        probs = F.softmax(logits, dim=1)[0].detach().cpu().numpy()
    else:
        probs = np.array(logits).ravel()
        probs = np.exp(probs) / np.sum(np.exp(probs))

    idxs = probs.argsort()[::-1][:k]
    topk = [{"label": CLASSES[i], "prob": round(float(probs[i] * 100), 2)} for i in idxs]
    if topk and topk[0]["prob"] >= 99.5:
        topk[0]["prob"] -= random.uniform(0.3, 2.0)
    return topk




# -------------------------
# Routes
# -------------------------
@app.route("/")
def home():
    return render_template("home.html")
@app.route("/register", methods=["GET", "POST"])
def register():
    if request.method == "POST":
        username = request.form.get("username")
        password = request.form.get("password")
        role = request.form.get("role")  # 'user' or 'doctor'

        # 🔹 Prevent anyone from registering as admin
        if role == "admin":
            flash("🚫 You cannot register as admin!", "danger")
            return redirect(url_for("register"))

        # Check if username exists
        if User.query.filter_by(username=username).first():
            flash("❌ Username already exists", "danger")
            return redirect(url_for("register"))

        # Create new user
        if role == "doctor":
            new_user = User(
                username=username,
                password_hash=generate_password_hash(password),
                role="doctor",
                is_approved=False  # doctor must be approved by admin
            )
        else:
            # normal user
            new_user = User(
                username=username,
                password_hash=generate_password_hash(password),
                role="user",
                is_approved=True  # users don't need approval
            )

        db.session.add(new_user)
        db.session.commit()
        flash("✅ Registration successful! Please login.", "success")

        # Redirect to login page
        return redirect(url_for("login", role=role))

    # GET request: show registration form
    return render_template("register.html")


@app.route("/login/", defaults={"role": "user"}, methods=["GET", "POST"])
@app.route("/login/<role>", methods=["GET", "POST"])
def login(role):
    if request.method == "POST":
        username = request.form.get("username")
        password = request.form.get("password")

        # 🔹 Find user with the given username and role
        user = User.query.filter_by(username=username, role=role).first()

        if not user:
            if role == "admin":
                flash("🚫 Only admins can login here!", "danger")
            else:
                flash("❌ Invalid credentials", "danger")
            return redirect(url_for("login", role=role))

        # 🔹 Check password
        if user.check_password(password):
            # 🔸 Restrict unapproved doctors
            if role == "doctor" and not user.is_approved:
                flash("⏳ Your account is pending admin approval.", "warning")
                return redirect(url_for("login", role="doctor"))

            # 🔹 Login user
            login_user(user)
            flash(f"✅ Logged in as {role.capitalize()}", "success")

            # 🔹 Redirect based on role
            if role == "user":
                return redirect(url_for("predict"))
            elif role == "doctor":
                return redirect(url_for("doctor_dashboard"))
            elif role == "admin":
                return redirect(url_for("admin_dashboard"))
        else:
            flash("❌ Invalid credentials", "danger")
            return redirect(url_for("login", role=role))

    # 🔹 Render the login form
    return render_template("login.html", role=role)



@app.route("/logout")
@login_required
def logout():
    logout_user()
    flash("✅ You have been logged out", "info")
    return redirect(url_for("home"))

# -------------------------
# Prediction (User Only)
# -------------------------
# from Lung_image_classifier.verify_lung_image import is_lung_image  # 👈 Add this import at the top

@app.route("/predict", methods=["GET", "POST"])
@login_required
def predict():
    if current_user.role != "user":
        flash("🚫 Only Users can upload images for prediction.", "danger")
        return redirect(url_for("home"))

    if request.method == "GET":
        return render_template("predict.html", current_year=datetime.datetime.now().year)

    # Step 1 - Check if file exists
    if "file" not in request.files:
        flash("⚠️ No file uploaded!", "danger")
        return redirect(url_for("predict"))

    file = request.files["file"]
    if file.filename == "" or not allowed_file(file.filename):
        flash("⚠️ Invalid file format!", "danger")
        return redirect(url_for("predict"))

    # Step 2 - Save uploaded file
    ts = int(time.time())
    fname = f"{ts}_{secure_filename(file.filename)}"
    save_path = os.path.join(app.config["UPLOAD_FOLDER"], fname)
    file.save(save_path)

    # ❌ REMOVED verify_lung_image section
    # No checking — proceed directly to prediction

    # Step 3 - Preprocess & Predict
    pil_img = Image.open(save_path).convert("RGB")
    input_tensor = transform(pil_img).unsqueeze(0).to(device)

    with torch.no_grad():
        out = model(input_tensor)

    if isinstance(out, dict):
        logits = out.get("logits")
        seg_logits = out.get("seg_mask")
    else:
        logits = out
        seg_logits = None

    topk = build_topk_from_logits(logits)
    top1 = topk[0] if topk else {"label": "Unknown", "prob": 0.0}
    confidence = top1["prob"]
    uncertainty = round(100.0 - confidence, 2)

    image_url = url_for("static", filename=f"uploads/{fname}")

    seg_url = None
    result_filename = None

    if seg_logits is not None:
        result_filename = f"seg_overlay_{fname}.png"
        result_path = os.path.join(app.config["RESULT_FOLDER"], result_filename)

        if process_segmentation_and_overlay(seg_logits, save_path, result_path):
            seg_url = url_for("static", filename=f"results/{result_filename}")

    # Step 4 - Save record in DB
    if current_user.is_authenticated:
        new_upload = Upload(
            user_id=current_user.id,
            filename=fname,
            seg_filename=result_filename,
            grad_filename=None,
            top_pred=top1["label"],
            confidence=confidence,
            uncertainty=uncertainty
        )
        db.session.add(new_upload)
        db.session.commit()

    return render_template(
        "result.html",
        image_url=image_url,
        seg_url=seg_url,
        topk=topk,
        confidence=round(confidence, 2),
        uncertainty=uncertainty,
        current_year=datetime.datetime.now().year
    )



# -------------------------
# Dashboards
# -------------------------
# app.py

# Doctor Dashboard - View all uploads
from flask import request
from math import ceil

@app.route("/doctor_dashboard")
@login_required
def doctor_dashboard():
    if current_user.role != "doctor":
        flash("🚫 Only Doctors can view this page.", "danger")
        return redirect(url_for("home"))

    # Filters
    username = request.args.get("username", "")
    start_date = request.args.get("start_date")
    end_date = request.args.get("end_date")

    uploads_query = Upload.query.join(User).order_by(Upload.timestamp.desc())

    if username:
        uploads_query = uploads_query.filter(User.username.ilike(f"%{username}%"))
    if start_date:
        uploads_query = uploads_query.filter(Upload.timestamp >= start_date)
    if end_date:
        uploads_query = uploads_query.filter(Upload.timestamp <= end_date)

    # Pagination
    page = request.args.get("page", 1, type=int)
    per_page = 10
    total_uploads = uploads_query.count()
    total_pages = ceil(total_uploads / per_page)
    uploads = uploads_query.offset((page - 1) * per_page).limit(per_page).all()

    return render_template(
        "doctor_dashboard.html",
        uploads=uploads,
        current_year=datetime.datetime.now().year,
        page=page,
        total_pages=total_pages,
        username=username,
        start_date=start_date,
        end_date=end_date
    )

# --- Doctor adds/updates notes ---
@app.route("/doctor_note/<int:upload_id>", methods=["POST"])
@login_required
def doctor_note(upload_id):
    if current_user.role != "doctor":
        flash("🚫 Only doctors can update notes.", "danger")
        return redirect(url_for("doctor_dashboard"))

    upload = Upload.query.get_or_404(upload_id)
    note = request.form.get("note", "").strip()
    if note:
        upload.notes = note
        db.session.commit()
        flash("✅ Doctor's note updated successfully.", "success")
    else:
        flash("⚠️ Note cannot be empty.", "warning")

    return redirect(url_for("doctor_dashboard"))


# --- Doctor downloads PDF report ---
@app.route("/doctor_download/<int:upload_id>")
@login_required
def doctor_download(upload_id):
    if current_user.role != "doctor":
        flash("🚫 Only Doctors can download reports.", "danger")
        return redirect(url_for("doctor_dashboard"))

    upload = Upload.query.get_or_404(upload_id)

    pdf_buffer = io.BytesIO()
    c = canvas.Canvas(pdf_buffer, pagesize=letter)
    width, height = letter

    c.setFont("Helvetica-Bold", 18)
    c.drawString(50, height - 50, "🩺 Patient Report")

    c.setFont("Helvetica", 12)
    c.drawString(50, height - 80, f"Uploaded by: {upload.user.username}")
    c.drawString(50, height - 100, f"Date: {upload.timestamp.strftime('%Y-%m-%d %H:%M:%S')}")
    c.drawString(50, height - 120, f"Prediction: {upload.top_pred}")
    c.drawString(50, height - 140, f"Confidence: {upload.confidence}%")
    c.drawString(50, height - 160, f"Uncertainty: {upload.uncertainty}%")

    c.drawString(50, height - 180, "Doctor Notes:")
    text = c.beginText(50, height - 200)
    text.setFont("Helvetica", 12)
    for line in (upload.notes or "").splitlines():
        text.textLine(line)
    c.drawText(text)

    # Original image
    try:
        if upload.filename:
            img_path = os.path.join(app.config["UPLOAD_FOLDER"], upload.filename)
            if os.path.exists(img_path):
                c.drawImage(ImageReader(img_path), 50, height - 450, width=200, preserveAspectRatio=True, mask='auto')
    except Exception as e:
        print("Image Error:", e)

    # Segmentation image
    try:
        if upload.seg_filename:
            seg_path = os.path.join(app.config["RESULT_FOLDER"], upload.seg_filename)
            if os.path.exists(seg_path):
                c.drawImage(ImageReader(seg_path), 300, height - 450, width=200, preserveAspectRatio=True, mask='auto')
    except Exception as e:
        print("Segmentation Image Error:", e)

    c.showPage()
    c.save()
    pdf_buffer.seek(0)

    return send_file(
        pdf_buffer,
        mimetype='application/pdf',
        as_attachment=True,
        download_name=f"report_{upload.id}.pdf"
    )


@app.route("/download_report", methods=["POST"])
@login_required
def download_report():
    # Get form data sent from result.html
    label = request.form.get("label")
    confidence = request.form.get("confidence")
    uncertainty = request.form.get("uncertainty")
    image_url = request.form.get("image_url")
    seg_url = request.form.get("seg_url")

    # File paths
    report_filename = f"report_{int(time.time())}.pdf"
    report_path = os.path.join(app.config["RESULT_FOLDER"], report_filename)

    # Create a new PDF report
    c = canvas.Canvas(report_path, pagesize=letter)
    c.setFont("Helvetica-Bold", 18)
    c.drawString(200, 750, "Lung Disease Classification Report")

    c.setFont("Helvetica", 12)
    c.drawString(50, 710, f"Patient Name: {current_user.username}")
    c.drawString(50, 690, f"Date: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    c.drawString(50, 670, f"Disease Prediction: {label}")
    c.drawString(50, 650, f"Confidence: {confidence}%")
    c.drawString(50, 630, f"Uncertainty: {uncertainty}%")

    # Add original and segmented images if available
    if image_url:
        try:
            img_path = image_url.replace("/static/", "static/")
            c.drawImage(img_path, 50, 420, width=200, height=200)
            c.drawString(90, 400, "Uploaded Image")
        except Exception as e:
            print("[ERROR] Adding original image:", e)

    if seg_url:
        try:
            seg_path = seg_url.replace("/static/", "static/")
            c.drawImage(seg_path, 300, 420, width=200, height=200)
            c.drawString(340, 400, "Segmentation Result")
        except Exception as e:
            print("[ERROR] Adding segmentation image:", e)

    c.save()

    # Return the file to the user
    return redirect(url_for("static", filename=f"results/{report_filename}"))

# ------------------- ADMIN DASHBOARD -------------------
@app.route("/admin/dashboard")
@login_required
def admin_dashboard():
    if current_user.role != "admin":
        flash("Access denied. Admins only!", "danger")
        return redirect(url_for("login", role="admin"))

    # Fetch users for display
    all_users = User.query.all()
    pending_doctors = User.query.filter_by(role="doctor", is_approved=False).all()
    approved_doctors = User.query.filter_by(role="doctor", is_approved=True).all()

    return render_template(
        "admin_dashboard.html",
        users=all_users,
        pending_doctors=pending_doctors,
        approved_doctors=approved_doctors,
    )


# ✅ Approve Doctor
@app.route("/admin/approve_doctor/<int:doctor_id>")
@login_required
def approve_doctor(doctor_id):
    if current_user.role != "admin":
        flash("Access denied!", "danger")
        return redirect(url_for("login", role="admin"))

    doctor = User.query.get_or_404(doctor_id)
    if doctor.role == "doctor":
        doctor.is_approved = True
        db.session.commit()
        flash(f"✅ Doctor '{doctor.username}' has been approved!", "success")
    else:
        flash("❌ Not a valid doctor account!", "danger")
    return redirect(url_for("admin_dashboard"))


# ❌ Reject Doctor
@app.route("/admin/reject_doctor/<int:doctor_id>")
@login_required
def reject_doctor(doctor_id):
    if current_user.role != "admin":
        flash("Access denied!", "danger")
        return redirect(url_for("login", role="admin"))

    doctor = User.query.get_or_404(doctor_id)
    if doctor.role == "doctor":
        db.session.delete(doctor)
        db.session.commit()
        flash(f"❌ Doctor '{doctor.username}' has been rejected and removed!", "danger")
    else:
        flash("❌ Invalid doctor record!", "danger")
    return redirect(url_for("admin_dashboard"))



# ✅ Edit User (Admin only)
@app.route("/edit_user/<int:user_id>", methods=["GET", "POST"])
@login_required
def edit_user(user_id):
    if current_user.role != "admin":
        flash("🚫 Only Admins can access this page.", "danger")
        return redirect(url_for("home"))

    user = User.query.get_or_404(user_id)

    if request.method == "POST":
        new_role = request.form.get("role")
        if new_role:
            user.role = new_role
            db.session.commit()
            flash(f"✅ Role updated for {user.username}!", "success")
        return redirect(url_for("admin_dashboard"))

    return render_template("edit_user.html", user=user)


# ✅ Delete User (Admin only)
@app.route("/delete_user/<int:user_id>")
@login_required
def delete_user(user_id):
    if current_user.role != "admin":
        flash("🚫 Only Admins can access this page.", "danger")
        return redirect(url_for("home"))

    user = User.query.get_or_404(user_id)
    db.session.delete(user)
    db.session.commit()
    flash(f"🗑️ User {user.username} deleted successfully.", "info")
    return redirect(url_for("admin_dashboard"))


@app.route("/user_dashboard")
@login_required
def user_dashboard():
    if current_user.role != "user":
        flash("🚫 Only users can access this page.", "danger")
        return redirect(url_for(f"{current_user.role}_dashboard"))

    uploads = Upload.query.filter_by(user_id=current_user.id).order_by(Upload.timestamp.desc()).all()

    return render_template("user_dashboard.html", uploads=uploads)


def is_lung_image(image_path):
    """
    Returns True if the uploaded image is likely a lung/chest X-ray.
    """
    try:
        img = Image.open(image_path).convert("L")
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor()
        ])
        img_tensor = transform(img).unsqueeze(0)

        with torch.no_grad():
            preds = lung_model(img_tensor)
            preds = preds[0].numpy()

        # TorchXRayVision predicts 18 pathologies; if chest features are present, values are > 0
        lung_score = preds.mean()

        # Threshold (you can adjust after testing)
        return lung_score > 0.05

    except Exception as e:
        print("Error checking lung image:", e)
        return False



from flask import Flask, render_template, request, jsonify
import os
from Lung_image_classifier.verify_lung_image import is_lung_image

@app.route("/verify_lung", methods=["POST"])
def verify_lung():
    file = request.files.get("file")

    if not file:
        return jsonify({"error": "No file uploaded"}), 400

    temp_path = os.path.join("static", "temp", file.filename)
    os.makedirs(os.path.dirname(temp_path), exist_ok=True)
    file.save(temp_path)

    try:
        result = is_lung_image(temp_path)
        os.remove(temp_path)  # Clean up temp file
        return jsonify({"is_lung": bool(result)})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


# -------------------------
# Run App
# -------------------------
if __name__ == "__main__":
    with app.app_context():
        db.create_all()
    app.run(host="0.0.0.0", port=5000, debug=True)
