from asyncio.windows_events import NULL
from importlib.resources import path
from flask import Flask, render_template, request
from keras.models import load_model
from keras.preprocessing import image
import tensorflow as tf

app = Flask(__name__)

dic = {0 : 'nofire', 1 : 'fire'}

des = {
	"nofire": {
		"deskripsi"  : ""
	},
	"fire":{
		"deskripsi"  : ""
	}
}

model = load_model('./Model Terbaik Layer 1-350.h5')

model.make_predict_function()

path = dic

def predict_label(path):
	i = tf.keras.utils.load_img(path, target_size=(224,224))
	i = tf.keras.utils.img_to_array(i)
	i = i.reshape(1,224,224,3)
	p = model.predict(i)
	return dic[p.argmax()]

# routes
@app.route("/", methods=['GET', 'POST'])
def main():
	return render_template("index.html")

@app.route("/about")
def about_page():
	return "Please subscribe  Artificial Intelligence Hub..!!!"

@app.route("/submit", methods = ['GET', 'POST'])
def get_output():
	if request.method == 'POST':
		img = request.files['my_image']
		path = "static/" + img.filename	
		img.save(path)
		p = predict_label(path)
		d = des[p]
	return render_template("index.html", prediction = p, path = path, img = path, d = d)

if __name__ =='__main__':
	#app.debug = True
	app.run(debug = True)