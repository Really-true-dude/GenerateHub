from flask import Flask, flash, redirect, render_template, url_for, request, send_file
from flask_sqlalchemy import SQLAlchemy
from flask_login import UserMixin, login_user, LoginManager, login_required, logout_user, current_user
from flask_bcrypt import Bcrypt
from werkzeug.utils import secure_filename
import os
import shutil
import sys
from datetime import datetime
sys.path.append("sd/")
from run import runSD

#app = Flask(__name__, static_folder = 'output')
app = Flask(__name__)

bcrypt = Bcrypt(app)

basedir = os.path.abspath(os.path.dirname(__file__))
upload_folder = "static/"
allowed_extensions = {'png', 'jpg', 'jpeg'}

app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///' + os.path.join(basedir, 'database.db')
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
app.config['SECRET_KEY'] = "Yuuki"
app.config['UPLOAD_FOLDER'] = upload_folder

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in allowed_extensions

db = SQLAlchemy(app)

login_manager = LoginManager()
login_manager.init_app(app)
login_manager.login_view = "login"

@login_manager.user_loader
def load_user(user_id):
    return User.query.get(int(user_id))

class User(db.Model, UserMixin):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(80), nullable=False, unique=True)
    email = db.Column(db.String(100), nullable=False, unique=True)
    password = db.Column(db.String(50), nullable=False, unique=True)
    registered = db.Column(db.String(30), nullable=False)

    def __repr__(self):
        return f'{self.username}'
    
class Image(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    imageName = db.Column(db.String(40), nullable=False)
    username = db.Column(db.String(80), nullable=False)
    prompt = db.Column(db.String(300), nullable=False)
    negativePrompt = db.Column(db.String(300))
    inferenceSteps = db.Column(db.Integer, nullable=False)
    modelName = db.Column(db.String(100), nullable=False)
    width = db.Column(db.Integer, nullable=False)
    height = db.Column(db.Integer, nullable=False)
    seed = db.Column(db.String(30))
    cfg_scale = db.Column(db.Integer)
    strength = db.Column(db.Numeric(precision=3, scale=2))

    def __repr__(self):
        return f'{self.imageName}'

# def get_avtar_path(user_folder):
#     avatar_pic_path = 'userPic.png'
#     for path in os.listdir(user_folder):
#         if path.split('.')[0] == 'avatar':
#             avatar_pic_path = path
#     return user_folder + avatar_pic_path

@app.route('/')
def start_page():
    return redirect(url_for('login'))

@app.route('/login', methods=['GET', 'POST'])
def login():
    err_msg = ""

    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        user = User.query.filter_by(username=username).first()
        if user:
            if bcrypt.check_password_hash(user.password, password):
                login_user(user)
                return redirect(url_for('home_page'))
            err_msg = "Password does not match!"
        else:
            err_msg = "This user does not exist!"

    return render_template("login.html", error_message = err_msg)

@app.route('/register', methods=['GET', 'POST'])
def register():
    err_msg = ""

    if request.method == 'POST':
        username = request.form['username']
        email = request.form['email']
        user = User.query.filter_by(username=username).first()
        if not user:
            hashed_pass = bcrypt.generate_password_hash(request.form['password'])
            register_date = datetime.now().strftime("%d/%m/%Y")
            user = User(username=username, email=email, password=hashed_pass, registered=register_date)
            db.session.add(user)
            db.session.commit()
            flash("New user created succesfully. You can now login.")
            return redirect(url_for('login'))
        else:
            err_msg = "This user already exists!"

    return render_template("register.html", error_message=err_msg)

@app.route('/account')
@login_required
def my_account():
    username = str(current_user)
    user_folder = 'static/users/' + username
    user_info = User.query.filter_by(username=username).first()
    num_gen_images = len(os.listdir("static/users/" + user_info.username + "/sd"))
    avatar_pic_path = user_folder + '/avatar.jpg' if os.path.isfile(user_folder + '/avatar.jpg') else url_for('static', filename='images/userPic.png')
    return render_template("account.html", user_info=user_info, num_gen_images=num_gen_images, avatar_pic_path=avatar_pic_path)

@app.route('/upload', methods=['POST'])
@login_required
def upload():
    username = str(current_user)
    if request.method == 'POST':
        if 'file' not in request.files:
            return 'No file part!'
        file = request.files['file']
        if file.filename == '':
            return 'No file selected!'

        if file and allowed_file(file.filename):
            # filename = secure_filename(file.filename)
            # filename, file_extension = os.path.splitext(filename)
            filename = "avatar.jpg"
            #filename = "avatar" + file_extension
            file.save(os.path.join(upload_folder + "users/" + username + "/" + filename))
    
    return redirect(url_for('my_account'))

@app.route('/delete_account')
@login_required
def delete_account():
    username = str(current_user)
    user = User.query.filter_by(username=username).first()
    db.session.delete(user)
    db.session.commit()
    shutil.rmtree("static/users/" + username, ignore_errors=True)
    flash("Account deleted successfully.")
    logout_user()
    return redirect(url_for('login'))


@app.route('/logout', methods=['GET', 'POST'])
@login_required
def logout():
    logout_user()
    return redirect(url_for('login'))

@app.route('/home', methods=['GET', 'POST'])
@login_required
def home_page():
    username = str(current_user)
    user_folder = 'static/users/' + username
    images = Image.query.filter_by(username=username).all()
    images_info = {}
    for image in images:
        images_info[image.imageName] = { 
            "Prompt" : image.prompt,
            "Negative_Prompt" : image.negativePrompt,
            "Inference_Steps" : image.inferenceSteps,
            "Width" : image.width,
            "Height" : image.height,
            "Model_Name" : image.modelName
        }
    folder_path = "static/users/" + username
    avatar_pic_path = user_folder + '/avatar.jpg' if os.path.isfile(user_folder + '/avatar.jpg') else url_for('static', filename='images/userPic.png')
    if not os.path.isdir(folder_path):
        os.makedirs(folder_path) # creates recursevily
        os.makedirs(folder_path + "/sd/")
    
    #image_paths = os.listdir(folder_path + "/sd/")

    return render_template("home.html", username=username, images_info=images_info, avatar_pic_path=avatar_pic_path)

@app.route('/download')
def download():
    image_path = request.args['image_path']
    return send_file(image_path, as_attachment=True)


@app.route('/delete')
@login_required
def delete():
    image_name = request.args.get('image_path')
    image_info = Image.query.filter_by(imageName=image_name).first()
    print(image_info.imageName, image_info.prompt)
    db.session.delete(image_info)
    db.session.commit()
    os.remove(image_name)
    flash("Image deleted succesfully.")
    print(f"Image removed: {image_name} | username: {str(current_user)}")

    return redirect(url_for('home_page'))


@app.route('/generate')
@login_required
def generate_page():
    return render_template("generate.html")

@app.route('/generating', methods=['GET', 'POST'])
@login_required
def generating():
    # Generating image
    username = str(current_user)
    save_path = "static/users/" + username + "/sd/"
    model_path = "data/" + request.form["model"] + ".safetensors"
    prompt = request.form["prompt"]
    uncond_prompt = request.form["uncond_prompt"]
    steps = int(request.form["n_inference_steps"])
    width = int(request.form["width"])
    height = int(request.form["height"])
    image_path = runSD(prompt=prompt, uncond_prompt=uncond_prompt, n_inference_steps=steps, DO_UPSCALE=True,
                       WIDTH=width, HEIGHT=height, model_file=model_path, SAVE_PATH=save_path)
    print(f"Generated image: {image_path}")
    # Saving info about the image in db
    image_info = Image(imageName=image_path, username=username, prompt=prompt, negativePrompt=uncond_prompt,
                       inferenceSteps=steps, modelName=os.path.split(model_path)[1], width=width, height=height)
    db.session.add(image_info)
    db.session.commit()

    return render_template("generate.html", image_path=image_path)

if __name__ == "__main__":
    app.run()