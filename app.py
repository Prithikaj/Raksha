import os
import time
import threading
import cv2
import subprocess
import pandas as pd

from flask import (
    Flask, render_template, request, redirect, url_for, session, jsonify, Response
)
from werkzeug.security import generate_password_hash, check_password_hash
from pymongo import MongoClient

# Twilio for sending SMS
from twilio.rest import Client

# Optional speech recognition
import speech_recognition as sr

# Geocoding and Reverse Geocoding
from geopy.geocoders import Nominatim
from geopy.exc import GeocoderUnavailable, GeopyError

# Local imports
import config
import route_finder
from ml_model import train_and_update_model

app = Flask(__name__)
app.secret_key = config.SECRET_KEY

# ========== 1) MONGODB SETUP (User Registration) ==========
client = MongoClient(config.MONGO_URI)
db = client["raksha"]
collection = db["safety"]

# ========== 2) TWILIO SETUP ==========
TWILIO_ACCOUNT_SID = config.TWILIO_SID
TWILIO_AUTH_TOKEN = config.TWILIO_AUTH_TOKEN
TWILIO_PHONE_NUMBER = config.TWILIO_PHONE_NUMBER
twilio_client = Client(TWILIO_ACCOUNT_SID, TWILIO_AUTH_TOKEN)

# ========== 3) CRIME DATA FILE (Safe Route Finder) ==========
CRIME_DATA_FOLDER = "crime_data"
CRIME_DATA_FILE = os.path.join(CRIME_DATA_FOLDER, "reported_crimes.csv")
if not os.path.exists(CRIME_DATA_FOLDER):
    os.makedirs(CRIME_DATA_FOLDER)
if not os.path.exists(CRIME_DATA_FILE):
    pd.DataFrame(columns=["location", "incident_type", "date", "verified"]).to_csv(CRIME_DATA_FILE, index=False)

# ========== GLOBAL VARIABLE FOR PANIC MODE ==========
PANIC_STOP_EVENT = None

# ========== HELPER FUNCTIONS ==========

def send_alert_message(latitude=None, longitude=None):
    """
    Sends an SMS alert (via Twilio) to a fixed, verified number.
    If latitude and longitude are provided, attempts to reverse geocode them to get
    a human-readable address and includes it in the message.
    """
    if "user" in session:
        user_data = session["user"]
        location_str = ""
        if latitude is not None and longitude is not None:
            try:
                geolocator = Nominatim(user_agent="safe_route_app", timeout=10)
                # Reverse geocode the coordinates to get an address
                location = geolocator.reverse((latitude, longitude))
                if location and location.address:
                    location_str = f" near {location.address}"
                else:
                    location_str = f" at latitude {latitude}, longitude {longitude}"
            except Exception as e:
                print("Error in reverse geocoding:", e)
                location_str = f" at latitude {latitude}, longitude {longitude}"
        message = f"Emergency Alert! {user_data['name']} might be in danger{location_str}."
        try:
            twilio_client.messages.create(
                body=message,
                from_=TWILIO_PHONE_NUMBER,
                to=""  # Hardcoded verified number for trial accounts
            )
            print("SMS Alert Sent!")
        except Exception as e:
            print("Error sending SMS:", e)

def save_video(stop_event):
    """
    Continuously capture webcam video until stop_event is set.
    Writes frames to evidence/recording.mp4 using the mp4v codec.
    Automatically stops after 10 seconds if not manually stopped.
    """
    os.makedirs("evidence", exist_ok=True)
    # Save as MP4 instead of AVI
    video_path = os.path.join("evidence", "recording.mp4")

    # Open default camera (DirectShow on Windows)
    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    actual_width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
    actual_height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
    print(f"[DEBUG] Camera opened at resolution: {actual_width} x {actual_height}")

    # Use mp4v codec for MP4 output
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(video_path, fourcc, 20.0, (640, 480))

    start_time = time.time()
    max_duration = 10  # seconds

    while not stop_event.is_set():
        ret, frame = cap.read()
        print("[DEBUG] Frame read:", ret)
        if not ret:
            print("[DEBUG] No more frames or camera error.")
            break

        out.write(frame)

        # Auto-stop after 10 seconds if user doesn't manually stop
        if time.time() - start_time > max_duration:
            print("[DEBUG] 10-second recording limit reached, stopping.")
            break

    cap.release()
    out.release()
    print("[DEBUG] Video recording stopped and file released.")

def generate_frames():
    """
    Generator function for streaming webcam frames (MJPEG) to /video_feed.
    """
    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    actual_width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
    actual_height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
    print(f"[DEBUG] generate_frames() camera opened at {actual_width} x {actual_height}")

    while True:
        success, frame = cap.read()
        print("[DEBUG] generate_frames() Frame read:", success)
        if not success:
            print("[DEBUG] generate_frames() No more frames or camera error.")
            break
        ret, buffer = cv2.imencode('.jpg', frame)
        if not ret:
            print("[DEBUG] generate_frames() Failed to encode frame.")
            break
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')
    cap.release()
    print("[DEBUG] generate_frames() camera released.")

# ========== 4) PUBLIC ROUTES ==========

@app.route("/")
def home():
    return render_template("home.html")

@app.route("/about")
def about():
    return "About page (public). You can create about.html if you want."

@app.route("/contact")
def contact():
    return "Contact page (public). You can create contact.html if you want."

# ========== 5) USER AUTH (Login, Register, Logout) ==========

@app.route("/register", methods=["GET", "POST"])
def register():
    if request.method == "POST":
        name = request.form["name"]
        phone_number = request.form["phone_number"]
        emergency_contact_name = request.form["emergency_contact_name"]
        emergency_contact_phone = request.form["emergency_contact_phone"]
        emergency_contact_email = request.form["emergency_contact_email"]
        emergency_contact_relation = request.form["emergency_contact_relation"]

        emergency_contact2_name = request.form["emergency_contact2_name"]
        emergency_contact2_phone = request.form["emergency_contact2_phone"]
        emergency_contact2_email = request.form["emergency_contact2_email"]
        emergency_contact2_relation = request.form["emergency_contact2_relation"]

        username = request.form["username"]
        password = generate_password_hash(request.form["password"])
        trigger_word = request.form["trigger_word"]
        stop_word = request.form["stop_word"]

        user_data = {
            "name": name,
            "phone_number": phone_number,
            "emergency_contact": {
                "name": emergency_contact_name,
                "phone": emergency_contact_phone,
                "email": emergency_contact_email,
                "relation": emergency_contact_relation
            },
            "emergency_contact2": {
                "name": emergency_contact2_name,
                "phone": emergency_contact2_phone,
                "email": emergency_contact2_email,
                "relation": emergency_contact2_relation
            },
            "username": username,
            "password": password,
            "trigger_word": trigger_word,
            "stop_word": stop_word
        }
        collection.insert_one(user_data)
        return redirect(url_for("login"))
    return render_template("register.html")

@app.route("/login", methods=["GET", "POST"])
def login():
    if request.method == "POST":
        username = request.form["username"]
        password = request.form["password"]

        user = collection.find_one({"username": username})
        if not user:
            return "User not found"

        if check_password_hash(user["password"], password):
            session["user"] = {
                "name": user["name"],
                "phone_number": user["phone_number"],
                "username": user["username"],
                "trigger_word": user["trigger_word"],
                "stop_word": user["stop_word"]
            }
            return redirect(url_for("welcome"))
        else:
            return "Invalid password"
    return render_template("login.html")

@app.route("/logout")
def logout():
    session.clear()
    return redirect(url_for("home"))

# ========== 6) PRIVATE ROUTES (Require Login) ==========

@app.route("/welcome")
def welcome():
    if "user" not in session:
        return redirect(url_for("login"))
    return render_template("welcome.html")

@app.route("/panic_mode", methods=["POST"])
def panic_mode():
    global PANIC_STOP_EVENT
    if "user" not in session:
        return redirect(url_for("login"))

    data = request.get_json() or {}
    latitude = data.get("latitude")
    longitude = data.get("longitude")

    if PANIC_STOP_EVENT is None:
        PANIC_STOP_EVENT = threading.Event()
        # Start a background thread to record video
        threading.Thread(target=save_video, args=(PANIC_STOP_EVENT,)).start()
        # Send SMS alert including reverse geocoded location
        send_alert_message(latitude, longitude)
        print(f"Panic mode triggered! lat={latitude}, lon={longitude}")
        return jsonify({"message": "Emergency Alert Sent! Recording Started."})
    else:
        return jsonify({"message": "Panic mode is already active."})

@app.route("/stop_panic", methods=["POST"])
def stop_panic():
    global PANIC_STOP_EVENT
    if "user" not in session:
        return redirect(url_for("login"))

    if PANIC_STOP_EVENT is not None:
        PANIC_STOP_EVENT.set()
        PANIC_STOP_EVENT = None
        print("Panic mode stopped by user.")
        return jsonify({"message": "Panic mode stopped."})
    else:
        return jsonify({"message": "Panic mode is not active."})

@app.route("/video_feed")
def video_feed():
    if "user" not in session:
        return redirect(url_for("login"))
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

# ========== 7) CRIME REPORTING & ROUTE FINDER ==========

@app.route("/report")
def report_crime():
    if "user" not in session:
        return redirect(url_for("login"))
    return render_template("report_crime.html")

@app.route("/submit_report", methods=["POST"])
def submit_report():
    if "user" not in session:
        return redirect(url_for("login"))

    location = request.form["location"]
    incident_type = request.form["incident_type"]
    date = request.form["date"]

    try:
        df = pd.read_csv(CRIME_DATA_FILE)
    except Exception as e:
        print("Error reading CSV:", e)
        df = pd.DataFrame(columns=["location", "incident_type", "date", "verified"])
    
    print("Before appending, DataFrame contents:")
    print(df)

    new_report = pd.DataFrame([{
        "location": location,
        "incident_type": incident_type,
        "date": date,
        "verified": False
    }])
    
    df = pd.concat([df, new_report], ignore_index=True)
    print("After appending, DataFrame contents:")
    print(df)
    
    try:
        df.to_csv(CRIME_DATA_FILE, index=False)
    except Exception as e:
        print("Error writing CSV:", e)
    
    subprocess.run(["python", "ml_model.py"])
    return redirect(url_for("welcome"))

@app.route("/verify_crime/<location>")
def verify_crime(location):
    if "user" not in session:
        return redirect(url_for("login"))

    df = pd.read_csv(CRIME_DATA_FILE)
    if location in df["location"].values:
        df.loc[df["location"] == location, "verified"] = True
        df.to_csv(CRIME_DATA_FILE, index=False)
        return f"✅ Crime at {location} has been verified!"
    else:
        return f"❌ Location '{location}' not found in reported crimes.", 404

# ========== 8) ROUTE FINDER WITH A POST FORM ==========

@app.route("/find_route", methods=["GET", "POST"])
def find_route():
    if "user" not in session:
        return redirect(url_for("login"))

    if request.method == "GET":
        return render_template("route_form.html")

    start = request.form.get("start", "").title()
    end = request.form.get("end", "").title()
    if not start or not end:
        return "Missing start or end location", 400

    geolocator = Nominatim(user_agent="safe_route_app", timeout=10)

    def get_coordinates(place_name):
        try:
            loc = geolocator.geocode(f"{place_name}, Coimbatore, India")
            if loc:
                return (loc.latitude, loc.longitude)
            else:
                return None
        except (GeocoderUnavailable, GeopyError) as e:
            print(f"Geocoding error for '{place_name}': {e}")
            return None
        except Exception as e:
            print(f"Unexpected geocoding error: {e}")
            return None

    start_coords = get_coordinates(start)
    if not start_coords:
        return f"❌ Unable to geocode start location: '{start}'", 400

    end_coords = get_coordinates(end)
    if not end_coords:
        return f"❌ Unable to geocode end location: '{end}'", 400

    df = pd.read_csv(CRIME_DATA_FILE)
    verified_crimes = df[df["verified"] == True]
    crime_locations = []
    for loc in verified_crimes["location"]:
        coords = get_coordinates(loc)
        if coords:
            crime_locations.append(coords)

    safe_path, alternate_path, caution_message = route_finder.find_safest_path(
        start_coords, end_coords, crime_locations
    )
    if safe_path is None:
        return "❌ No route available!", 400

    return render_template(
        "map.html",
        safe_path=safe_path,
        alternate_path=alternate_path,
        caution_message=caution_message,
        crime_locations=crime_locations
    )

# ========== MAIN ==========
if __name__ == "__main__":
    train_and_update_model()
    app.run(debug=True)
