import pandas as pd
import sys
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


def main():
    
    labelled = pd.read_csv(sys.argv[1])
    unlabelled = pd.read_csv(sys.argv[2])
    
    outputFile = sys.argv[3]

    X = labelled.drop('city', axis=1)                           #drop the 'city' column in labelled data so that dataset can be used for training the model
    y = labelled['city']
    
    X_unlabel = unlabelled.drop('city', axis=1)                 #drop this column so that later the SVC model can fill with predicted results

    X_train, X_test, y_train, y_test = train_test_split(X,y)

    svc_model = make_pipeline(                                                  #create SVC model
                          StandardScaler(),                                 #normalize dataset so that the individual features are more or less standard normally distributed data
                          SVC(kernel='linear')
    )

    svc_model.fit(X_train, y_train)                                 #train model
    predictions = svc_model.predict(X_unlabel)                  
    print(svc_model.score(X_test, y_test))

    pd.Series(predictions).to_csv(outputFile, index=False)

#    df = pd.DataFrame({'truth': y_test, 'prediction': svc_model.predict(X_test)})
#    print(df[df['truth'] != df['prediction']])

if __name__ == '__main__':
    main()
    