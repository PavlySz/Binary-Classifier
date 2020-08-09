'''
Main file for website
'''
import os
import io
import pandas as pd
from flask import Flask, render_template, request
from binary_classifier import *

app = Flask(__name__, template_folder="templates")

extra_dirs = ['templates', 'static']
extra_files = extra_dirs[:]
for extra_dir in extra_dirs:
    for dirname, dirs, files in os.walk(extra_dir):
        for filename in files:
            filename = os.path.join(dirname, filename)
            if os.path.isfile(filename):
                extra_files.append(filename)


def predict_user_data(data, clf):
    df = pd.read_csv(io.StringIO(data), sep=';', header=None)
    df.columns = ['variable1', 'variable2', 'variable3', 'variable4', 'variable5',
                  'variable6', 'variable7', 'variable8', 'variable9', 'variable10',
                  'variable11', 'variable12', 'variable13', 'variable14', 'variable15',
                  'variable17', 'variable18', 'variable19']

    X_test = preprocess_df_test(df)
    result = predict(X_test, clf)
    result = 'no' if int(result) == 0 else 'yes'
    print(result)
    return result


@app.route("/", methods=['GET', 'POST'])
def home():
    if request.method == 'POST':
        data = request.form.get('text')
        print(f"The user entered: {data}")

        if data != '':
            if data == 'YEP':
                result = 'YEP COCK'
            elif data == 'NOP':
                result = 'Sadge'
            else:
                result = predict_user_data(data, clf)

            if len(result) != 0:
                return render_template('index.html', user_input=data,
                                       results=result)
            else:
                return render_template('index.html', error='true')

    return render_template('index.html')


if __name__ == '__main__':
    # Train the classifier when the site is launched
    FOLDER_PATH = 'binary_classifier_data'
    df_train = pd.read_csv(f'{FOLDER_PATH}/training.csv', sep=';')
    X_train, y_train = preprocess_df(df_train)
    clf = train(X_train, y_train)

    app.run(debug=True, extra_files=extra_files)
