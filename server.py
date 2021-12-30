from flask import Flask,render_template
from speech_to_text import convert_to_text
from download_video import generate_ASL_gesture
import os
app = Flask(__name__)
from english_to_asl import english_to_asl
        

@app.route('/')
def index():
	return render_template('index.html')
	

@app.route('/translate',methods=['GET'])
def translate_asl():
    video_path = "../static/"
    try:
        os.remove(video_path)
    except Exception:
        pass
    try:
        text = convert_to_text()
        print("------> ", text)
        file_name = text.replace(" ","_")
        video_path = video_path + file_name+".mp4"
        text = english_to_asl(text)
        generate_ASL_gesture(text)
        return render_template('translate.html',video_path = video_path, sentence = text)
    except Exception as e:
        return render_template('error.html', deubg_error = str(e))

if __name__ == '__main__':
	app.run(debug=True)
