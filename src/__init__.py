from msilib.schema import Error
import os
import re
import tensorflow as tf
from flask import Flask, render_template, request
from util import calc_example


def create_app(test_config=None):
    new_model = tf.keras.models.load_model(os.path.dirname(__file__) + '/saved_model/ms_model')
    app = Flask(__name__, instance_relative_config=True)
    app.config.from_mapping(
        SECRET_KEY='dev',
        DATABASE=os.path.join(app.instance_path, 'flaskr.sqlite'),
    )

    if test_config is None:
        app.config.from_pyfile('config.py', silent=True)
    else:
        app.config.from_mapping(test_config)

    try:
        os.makedirs(app.instance_path)
    except OSError:
        pass

    @app.route('/')
    def home():
        return render_template('index.html')
    
    # a simple page that says hello
    @app.route('/hello')
    def hello():
        return 'Hello, World!'
    
    @app.route('/', methods=['POST'])
    def submit():
        text = request.form['expression']
        r = text
        r = r.replace(' ', '')
        r = re.sub(r"</?\[\d+>!@#$%^&*(){}'", "", r)
        if re.search('[a-zA-Z]', r):
            return render_template('index.html', output='Error!')
        print(os.path.dirname(__file__))   
        f = ''     
        try:
            f = calc_example(new_model, r)
        except Error as err:
            print(err)
            return render_template('index.html', output='Error!')
        return render_template('index.html', output=f)



    return app

if __name__=='__main__':
    create_app().run()