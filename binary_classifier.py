'''
binary_classifier.ipynb converted to a .py file
'''
# Import statements
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import f1_score


def convert_cols(df):
    '''
    Given a specific datafame, convert the numerical string values to numeric values
    For example: '19,04' (type: str) would be 19.04 (type float)

    Args:
        df (dataframe): Original dataframe

    Returns:
        df (dataframe): dataframe after column conversion
    '''
    # Convert columns: 'variable2' 'variable3' 'variable8' to float
    df['variable2'] = df['variable2'].apply(lambda x: str(x).replace(',', '.'))
    df['variable2'] = pd.to_numeric(df['variable2'])

    df['variable3'] = df['variable3'].apply(lambda x: str(x).replace(',', '.'))
    df['variable3'] = pd.to_numeric(df['variable3'])

    df['variable8'] = df['variable8'].apply(lambda x: str(x).replace(',', '.'))
    df['variable8'] = pd.to_numeric(df['variable8'])
    return df


def encode_df(df):
    '''
    Given a dataframe, encode the categorical columns
    to numeric values

    Args:
        df (dataframe): original dataframe

    Returns:
        df_enc (dataframe): dataframe after encoding
    '''
    le = LabelEncoder()

    CATEGORICAL_COLS = ['variable1', 'variable4', 'variable5',
                        'variable6', 'variable7', 'variable9', 'variable10',
                        'variable12', 'variable13', 'variable18', 'classLabel']

    df_enc = df.copy()
    for col in CATEGORICAL_COLS:
        df_enc[col] = le.fit_transform(df[col])

    return df_enc


def preprocess_df(df):
    '''
    Given a dataframe, perform the necessary pre-processing steps

    Args:
        df (dataframe): original dataframe

    Returns:
        X: training features
        y: training labels
    '''
    df = df.drop_duplicates()
    df = df.reset_index(drop=True)

    df = df.dropna()
    df = df.reset_index(drop=True)

    df = convert_cols(df)
    df_enc = encode_df(df)

    X = df_enc.iloc[:, :-1]
    y = df_enc.iloc[:, -1]

    return X, y


def preprocess_df_test(df):
    '''
    Given a dataframe, perform the necessary pre-processing steps

    Args:
        df (dataframe): original (validation) dataframe

    Returns:
        X: valdation features
        y: validation labels
    '''
    df = df.drop_duplicates()
    df = df.reset_index(drop=True)

    df = df.dropna()
    df = df.reset_index(drop=True)

    df = convert_cols(df)
    le = LabelEncoder()

    # Does not include the classLabel column
    CATEGORICAL_COLS = ['variable1', 'variable4', 'variable5',
                        'variable6', 'variable7', 'variable9', 'variable10',
                        'variable12', 'variable13', 'variable18']

    df_enc = df.copy()
    for col in CATEGORICAL_COLS:
        df_enc[col] = le.fit_transform(df[col])

    return df_enc


def train(X_train, y_train):
    '''
    Given training data, train the RandomForrestClassifier

    Args:
        X_train: training features
        y_train: training labels

    Returns:
        clf (sklearn classifier): trained classifier on given data
    '''
    clf = RandomForestClassifier(max_depth=5, n_estimators=10, max_features=1)
    print(f'Training {clf.__class__.__name__}')
    clf.fit(X_train, y_train)
    return clf


def predict(X_val, clf):
    '''
    Given validation features and a trained classifier,
    make prediction

    Args:
        X_val: validation features
        clf: trained classifier

    Returns:
        preds (list): list of predicted values
    '''
    preds = clf.predict(X_val)
    return preds


def main():
    '''
    Main function

    - Read data
    - Preprocess data
    - Train model
    - Test model
    - Print the F1-score of the model
    '''
    FOLDER_PATH = 'binary_classifier_data'
    df_train = pd.read_csv(f'{FOLDER_PATH}/training.csv', sep=';')
    df_val = pd.read_csv(f'{FOLDER_PATH}/validation.csv', sep=';')

    X_train, y_train = preprocess_df(df_train)
    X_val, y_val = preprocess_df(df_val)

    clf = train(X_train, y_train)
    preds = predict(X_val, clf)
    print(f'F1-score: {f1_score(preds, y_val, average="micro"):.3f}')


if __name__ == '__main__':
    main()
