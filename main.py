
from flask import Flask, render_template
app = Flask(__name__)
import pickle


# open a file, where you ant to store the data
file = open('model.pkl', 'rb')
clf = pickle.load(file)
file.close()

@app.route('/')
def hello_world():
    inputFeatures = [100,1,22,1,1]
    infProb=clf.predict_proba([inputFeatures])[0][1]
    return render_template('index.html')

    # return 'Hello,World!' + str(infProb)

if __name__=="__main__":
    app.run(debug=True)
