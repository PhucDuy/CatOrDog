import os
from flask import Flask, redirect, render_template, request, session, url_for
from flask import send_from_directory
import numpy as np
import tensorflow as tf
from flask_dropzone import Dropzone
from flask_uploads import UploadSet, configure_uploads, IMAGES, patch_request_class

app = Flask(__name__)
dropzone = Dropzone(app)
app.config['SECRET_KEY'] = 'supersecretkeygoeshere'

app.config['DROPZONE_UPLOAD_MULTIPLE'] = False
app.config['DROPZONE_ALLOWED_FILE_CUSTOM'] = True
app.config['DROPZONE_ALLOWED_FILE_TYPE'] = 'image/*'
app.config['DROPZONE_REDIRECT_VIEW'] = 'upload_file'

# Uploads settings
app.config['UPLOADED_PHOTOS_DEST'] = os.getcwd() + '/uploads'
photos = UploadSet('photos', IMAGES)
configure_uploads(app, photos)
patch_request_class(app)  # set maximum file size, default is 16MB


dir_path = os.path.dirname(os.path.realpath(__file__))
UPLOAD_FOLDER = 'uploads'
STATIC_FOLDER = 'static'

# Load model
cnn_model = tf.keras.models.load_model(
    STATIC_FOLDER + '/' + 'xception_model.h5')

IMAGE_SIZE = 299

# Preprocess an image


def preprocess_image(image):
    image = tf.image.decode_jpeg(image, channels=3)
    image = tf.image.resize(image, [IMAGE_SIZE, IMAGE_SIZE])
    image /= 255.0  # normalize to [0,1] range

    return image

# Read the image from path and preprocess


def load_and_preprocess_image(path):
    image = tf.io.read_file(path)
    return preprocess_image(image)

# Predict & classify image


def classify(model, image_path):

    preprocessed_imgage = load_and_preprocess_image(image_path)
    preprocessed_imgage = tf.reshape(
        preprocessed_imgage, (1, IMAGE_SIZE, IMAGE_SIZE, 3))

    prob = cnn_model.predict(preprocessed_imgage).flatten()[0]
    label = "Cat" if prob >= 0.5 else "Dog"
    classified_prob = prob if prob >= 0.5 else 1 - prob

    return label, float(classified_prob)

# home page
@app.route('/', methods=['GET', 'POST'])
def home():
    # set session for image results
    if "file_urls" not in session:
        session['file_urls'] = []
        session['files'] = []
    # list to hold our uploaded image urls
    file_urls = session['file_urls']
    files = session['files']
    # handle image upload from Dropszone
    if request.method == 'POST':
        file_obj = request.files
        for f in file_obj:
            file = request.files.get(f)

            # save the file with to our photos folder
            filename = photos.save(
                file,
                name=file.filename
            )

            # append image urls
            file_urls.append(photos.url(filename))
            files.append(filename)

        session['file_urls'] = file_urls
        session['files'] = files
        print(file_urls)
        return "uploading..."
    # return dropzone template on GET request

    return render_template('home.html')


@app.route('/classify')
def upload_file():
    if "file_urls" not in session or session['file_urls'] == []:
        return redirect(url_for('home'))

    # set the file_urls and remove the session variable
    file_urls = session['file_urls']
    files = session['files']
    session.pop('file_urls', None)
    session.pop('files', None)

    file = files[0]
    print(file)
    file_url = file_urls[0]
    upload_image_path = os.path.join(UPLOAD_FOLDER, file)
    label, prob = classify(cnn_model, upload_image_path)
    prob = round((prob * 100), 2)

    return render_template('classify.html', image_file_name=file_url, label=label, prob=prob)


if __name__ == '__main__':
    app.debug = True
    app.run(debug=True)
    app.debug = True
