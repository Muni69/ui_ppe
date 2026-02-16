import sqlite3
import numpy as np
import face_recognition
import os
import json
import hashlib
from flask import Flask, render_template, request, redirect, url_for, jsonify, g
from werkzeug.utils import secure_filename
from PIL import Image

app = Flask(__name__)
app.secret_key = os.environ.get("SECRET_KEY", "dev-fallback-key")
UPLOAD_FOLDER = 'static/faces'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

DB_NAME = "safety_system.db"

# RFID rate limiting: same card cannot be used again within this many minutes
RFID_COOLDOWN_MINUTES = 5

# --- SYNC CACHE ---
# Cached /api/sync response to avoid re-serializing on every 2-second poll.
_sync_cache = {"data": None, "dirty": True}


def _invalidate_sync_cache():
    """Mark the sync cache as dirty so the next /api/sync call re-fetches."""
    _sync_cache["dirty"] = True


# --- DATABASE HELPERS ---
def get_db():
    """Get a per-request database connection using Flask's g object."""
    if 'db' not in g:
        g.db = sqlite3.connect(DB_NAME)
        g.db.row_factory = sqlite3.Row
        g.db.execute("PRAGMA journal_mode=WAL")  # Better concurrent read performance
    return g.db


@app.teardown_appcontext
def close_db(exception):
    """Automatically close the DB connection at the end of each request."""
    db = g.pop('db', None)
    if db is not None:
        db.close()


# --- DATABASE SETUP ---
def init_db():
    conn = sqlite3.connect(DB_NAME)
    c = conn.cursor()
    # 1. Logs Table
    c.execute('''CREATE TABLE IF NOT EXISTS logs
                 (
                     id
                     INTEGER
                     PRIMARY
                     KEY
                     AUTOINCREMENT,
                     timestamp
                     TEXT,
                     name
                     TEXT,
                     status
                     TEXT,
                     details
                     TEXT
                 )''')
    # 2. Config Table (Stores required PPE IDs as a JSON list)
    c.execute('''CREATE TABLE IF NOT EXISTS config
                 (
                     key
                     TEXT
                     PRIMARY
                     KEY,
                     value
                     TEXT
                 )''')
    # 3. Users Table (Stores Name and Face Encoding)
    c.execute('''CREATE TABLE IF NOT EXISTS users
                 (
                     id
                     INTEGER
                     PRIMARY
                     KEY
                     AUTOINCREMENT,
                     name
                     TEXT,
                     encoding
                     TEXT,
                     image_path
                     TEXT
                 )''')
    # 4. RFID Cards Table (Stores Card UID linked to a user)
    c.execute('''CREATE TABLE IF NOT EXISTS rfid_cards
    (
        id
        INTEGER
        PRIMARY
        KEY
        AUTOINCREMENT,
        uid
        TEXT
        UNIQUE,
        user_id
        INTEGER,
        added_at
        TEXT
        DEFAULT (
        datetime
                 (
        'now',
        'localtime'
                 )),
        last_used_at TEXT DEFAULT NULL,
        auth_token_hash TEXT DEFAULT NULL,
        FOREIGN KEY
                 (
                     user_id
                 ) REFERENCES users
                 (
                     id
                 ) ON DELETE CASCADE)''')

    # Add columns if upgrading from older schema
    for col in ['last_used_at TEXT DEFAULT NULL', 'auth_token_hash TEXT DEFAULT NULL']:
        try:
            c.execute(f"ALTER TABLE rfid_cards ADD COLUMN {col}")
        except sqlite3.OperationalError:
            pass  # Column already exists

    # Index for fast RFID lookups
    c.execute('''CREATE INDEX IF NOT EXISTS idx_rfid_uid ON rfid_cards(uid)''')
    # Index for log queries (ordered by id DESC)
    c.execute('''CREATE INDEX IF NOT EXISTS idx_logs_id ON logs(id DESC)''')

    # Set default PPE (Vest=16, Helmet=10) if not exists
    c.execute("INSERT OR IGNORE INTO config (key, value) VALUES ('required_ppe', '[10, 16]')")
    conn.commit()
    conn.close()


init_db()


# --- WEB ROUTES (For Admin) ---
@app.route('/')
def dashboard():
    conn = get_db()
    logs = conn.execute("SELECT * FROM logs ORDER BY id DESC LIMIT 50").fetchall()
    config = conn.execute("SELECT value FROM config WHERE key='required_ppe'").fetchone()
    current_ppe = json.loads(config['value']) if config else []
    return render_template('dashboard.html', logs=logs, current_ppe=current_ppe)


@app.route('/faces', methods=['GET', 'POST'])
def manage_faces():
    conn = get_db()

    if request.method == 'POST':
        name = request.form['name']
        file = request.files['image']
        if file and name:
            filename = secure_filename(f"{name}_{file.filename}")
            path = os.path.join(UPLOAD_FOLDER, filename)

            # Save and Resize Image
            try:
                img = Image.open(file)
                # Convert to RGB if necessary (e.g. for PNGs with alpha)
                if img.mode in ('RGBA', 'P'):
                    img = img.convert('RGB')

                # Resize if too large (max 800x800) to save space and sync time
                img.thumbnail((800, 800))
                img.save(path, quality=85, optimize=True)
            except Exception as e:
                return f"Error processing image: {e}", 400

            # Process Face Encoding
            image = face_recognition.load_image_file(path)
            encodings = face_recognition.face_encodings(image)

            if len(encodings) > 0:
                encoding_json = json.dumps(encodings[0].tolist())
                # Store path with forward slashes for consistency
                db_path = path.replace("\\", "/")
                conn.execute("INSERT INTO users (name, encoding, image_path) VALUES (?, ?, ?)",
                             (name, encoding_json, db_path))
                conn.commit()
                _invalidate_sync_cache()
            else:
                return "Error: No face found in photo!", 400

    users = conn.execute("SELECT * FROM users").fetchall()
    return render_template('faces.html', users=users)


@app.route('/update_ppe', methods=['POST'])
def update_ppe():
    selected_ids = request.form.getlist('ppe_ids')
    json_data = json.dumps([int(i) for i in selected_ids])

    conn = get_db()
    conn.execute("UPDATE config SET value = ? WHERE key='required_ppe'", (json_data,))
    conn.commit()
    _invalidate_sync_cache()
    return redirect(url_for('dashboard'))


@app.route('/delete_user/<int:user_id>')
def delete_user(user_id):
    conn = get_db()
    conn.execute("DELETE FROM rfid_cards WHERE user_id=?", (user_id,))
    conn.execute("DELETE FROM users WHERE id=?", (user_id,))
    conn.commit()
    _invalidate_sync_cache()
    return redirect(url_for('manage_faces'))


# ==============================================
# RFID CARD MANAGEMENT ROUTES
# ==============================================

@app.route('/rfid_cards', methods=['GET', 'POST'])
def manage_rfid_cards():
    conn = get_db()
    error = None
    success = None

    if request.method == 'POST':
        uid = request.form.get('uid', '').strip().upper()
        user_id = request.form.get('user_id', '').strip()
        auth_token = request.form.get('auth_token', '').strip().upper()

        if uid and user_id:
            # Hash the auth token if provided (from sector key enrollment)
            token_hash = hashlib.sha256(auth_token.encode()).hexdigest() if auth_token else None

            try:
                conn.execute("INSERT INTO rfid_cards (uid, user_id, auth_token_hash) VALUES (?, ?, ?)",
                             (uid, int(user_id), token_hash))
                conn.commit()
                success = f"Card {uid} assigned successfully" + (" (with sector key auth)" if token_hash else "")
                _invalidate_sync_cache()
            except sqlite3.IntegrityError:
                error = f"Card UID {uid} is already registered"
            except Exception as e:
                error = f"Error: {e}"
        else:
            error = "Please provide both a card UID and select a worker"

    cards = conn.execute("""
                         SELECT rfid_cards.id, rfid_cards.uid, rfid_cards.added_at, users.name as worker_name
                         FROM rfid_cards
                                  LEFT JOIN users ON rfid_cards.user_id = users.id
                         ORDER BY rfid_cards.id DESC
                         """).fetchall()

    users = conn.execute("SELECT id, name FROM users ORDER BY name").fetchall()
    return render_template('rfid_cards.html', cards=cards, users=users, error=error, success=success)


@app.route('/delete_rfid/<int:card_id>')
def delete_rfid(card_id):
    conn = get_db()
    conn.execute("DELETE FROM rfid_cards WHERE id=?", (card_id,))
    conn.commit()
    return redirect(url_for('manage_rfid_cards'))


# --- API ROUTES (For Gate App) ---
@app.route('/api/sync', methods=['GET'])
def api_sync():
    """Sends PPE Rules and All Face Encodings to the Gate. Response is cached."""
    if not _sync_cache["dirty"] and _sync_cache["data"] is not None:
        return _sync_cache["data"]

    conn = get_db()
    config = conn.execute("SELECT value FROM config WHERE key='required_ppe'").fetchone()
    required_ids = json.loads(config['value'])

    users = conn.execute("SELECT name, encoding FROM users").fetchall()
    faces_data = [
        {"name": u['name'], "encoding": json.loads(u['encoding'])}
        for u in users
    ]

    response = jsonify({"required_ppe": required_ids, "faces": faces_data})
    _sync_cache["data"] = response
    _sync_cache["dirty"] = False
    return response


@app.route('/api/rfid_lookup', methods=['GET'])
def api_rfid_lookup():
    """Lookup a card UID with sector key auth + rate-limiting cooldown."""
    uid = request.args.get('uid', '').strip().upper()
    token = request.args.get('token', '').strip().upper()
    if not uid:
        return jsonify({"found": False, "name": "", "error": "No UID provided"}), 400

    conn = get_db()
    result = conn.execute("""
                          SELECT users.name,
                                 rfid_cards.last_used_at,
                                 rfid_cards.id as card_id,
                                 rfid_cards.auth_token_hash
                          FROM rfid_cards
                                   JOIN users ON rfid_cards.user_id = users.id
                          WHERE rfid_cards.uid = ?
                          """, (uid,)).fetchone()

    if not result:
        return jsonify({"found": False, "name": ""})

    # Sector key authentication: verify token hash
    stored_hash = result['auth_token_hash']
    if stored_hash:
        if not token:
            return jsonify({"found": True, "name": result['name'], "auth_failed": True,
                            "error": "Card has no auth token - possible clone"})
        token_hash = hashlib.sha256(token.encode()).hexdigest()
        if token_hash != stored_hash:
            return jsonify({"found": True, "name": result['name'], "auth_failed": True,
                            "error": "Token mismatch - possible clone"})

    # Rate limiting: check cooldown
    if result['last_used_at']:
        from datetime import datetime, timedelta
        try:
            last_used = datetime.strptime(result['last_used_at'], '%Y-%m-%d %H:%M:%S')
            cooldown_end = last_used + timedelta(minutes=RFID_COOLDOWN_MINUTES)
            now = datetime.now()
            if now < cooldown_end:
                remaining = (cooldown_end - now).total_seconds() / 60
                return jsonify({
                    "found": True,
                    "name": result['name'],
                    "cooldown": True,
                    "remaining_minutes": round(remaining, 1)
                })
        except (ValueError, TypeError):
            pass  # Malformed timestamp, allow access

    # Update last_used_at timestamp
    conn.execute(
        "UPDATE rfid_cards SET last_used_at = datetime('now', 'localtime') WHERE id = ?",
        (result['card_id'],)
    )
    conn.commit()

    return jsonify({"found": True, "name": result['name']})


@app.route('/api/rfid_enroll', methods=['POST'])
def api_rfid_enroll():
    """Store the auth token hash for a newly enrolled card."""
    data = request.json
    uid = data.get('uid', '').strip().upper()
    token = data.get('token', '').strip().upper()

    if not uid or not token:
        return jsonify({"error": "Missing uid or token"}), 400

    token_hash = hashlib.sha256(token.encode()).hexdigest()

    conn = get_db()
    result = conn.execute("SELECT id FROM rfid_cards WHERE uid = ?", (uid,)).fetchone()

    if result:
        # Update existing card's token
        conn.execute("UPDATE rfid_cards SET auth_token_hash = ? WHERE id = ?",
                     (token_hash, result['id']))
        conn.commit()
        return jsonify({"status": "updated", "uid": uid})
    else:
        return jsonify({"error": "Card UID not registered. Register it first."}), 404


@app.route('/api/log', methods=['POST'])
def api_log():
    data = request.json
    conn = get_db()
    conn.execute("INSERT INTO logs (timestamp, name, status, details) VALUES (datetime('now', 'localtime'), ?, ?, ?)",
                 (data.get('name'), data.get('status'), data.get('details')))
    conn.commit()
    return "OK", 200


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)