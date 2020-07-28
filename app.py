from flask import Flask, render_template, url_for, redirect, request
from model_synthesis import make_simpsons
import os

app = Flask(__name__)
#static_folder = 'static'


@app.route('/')
def home():
    return render_template('index.html')


@app.route('/', methods=['POST'])
def predictonImage():
    if request.method == 'POST':
        cnt = 0
        if(os.path.exists("./home/final_model.hdf5")):
            cnt = 1
        rnd = make_simpsons(cnt)
        path_to_img = "./static/predictedImages/image%d.jpg" % rnd
        return render_template('index.html', path_to_img=path_to_img)
    return render_template('index.html')


if __name__ == '__main__':
    app.run(debug=True)
