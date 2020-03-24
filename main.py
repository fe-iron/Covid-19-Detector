from flask import Flask, render_template, request
import pickle

app = Flask(__name__)

file = open('model.pkl', 'rb')
clf = pickle.load(file)
file.close()


@app.route('/', methods=['GET', 'POST'])
def hello_world():
    if request.method == "POST":
        mydict = request.form
        fever = int(mydict['fever'])
        age = int(mydict['age'])
        pain = int(mydict['pain'])
        runnynose = int(mydict['runnynose'])
        diffbreathing = int(mydict['diffbreathing'])

        inputfeature = [fever, pain, age, runnynose, diffbreathing]
        infprob = clf.predict_proba([inputfeature])[0][1]
        #print(infprob)
        return render_template('show.html', inf=round(infprob*100))

    return render_template('index.html')
    #return 'Hello World '+ str(infprob)


if __name__ == '__main__':
    app.run(debug=True)


