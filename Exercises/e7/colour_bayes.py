import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from skimage.color import lab2rgb
import sys
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score
from skimage import color
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import FunctionTransformer


# representative RGB colours for each label, for nice display
COLOUR_RGB = {
    'red': (255, 0, 0),
    'orange': (255, 114, 0),
    'yellow': (255, 255, 0),
    'green': (0, 230, 0),
    'blue': (0, 0, 255),
    'purple': (187, 0, 187),
    'brown': (117, 60, 0),
    'pink': (255, 187, 187),
    'black': (0, 0, 0),
    'grey': (150, 150, 150),
    'white': (255, 255, 255),
}
name_to_rgb = np.vectorize(COLOUR_RGB.get, otypes=[np.uint8, np.uint8, np.uint8])


def plot_predictions(model, lum=71, resolution=256):
    """
    Create a slice of LAB colour space with given luminance; predict with the model; plot the results.
    """
    wid = resolution
    hei = resolution
    n_ticks = 5

    # create a hei*wid grid of LAB colour values, with L=lum
    ag = np.linspace(-100, 100, wid)
    bg = np.linspace(-100, 100, hei)
    aa, bb = np.meshgrid(ag, bg)
    ll = lum * np.ones((hei, wid))
    lab_grid = np.stack([ll, aa, bb], axis=2)

    # convert to RGB for consistency with original input
    X_grid = lab2rgb(lab_grid)

    # predict and convert predictions to colours so we can see what's happening
    y_grid = model.predict(X_grid.reshape((wid*hei, 3)))
    pixels = np.stack(name_to_rgb(y_grid), axis=1) / 255
    pixels = pixels.reshape((hei, wid, 3))

    # plot input and predictions
    plt.figure(figsize=(10, 5))
    plt.suptitle('Predictions at L=%g' % (lum,))
    plt.subplot(1, 2, 1)
    plt.title('Inputs')
    plt.xticks(np.linspace(0, wid, n_ticks), np.linspace(-100, 100, n_ticks))
    plt.yticks(np.linspace(0, hei, n_ticks), np.linspace(-100, 100, n_ticks))
    plt.xlabel('A')
    plt.ylabel('B')
    plt.imshow(X_grid.reshape((hei, wid, 3)))

    plt.subplot(1, 2, 2)
    plt.title('Predicted Labels')
    plt.xticks(np.linspace(0, wid, n_ticks), np.linspace(-100, 100, n_ticks))
    plt.yticks(np.linspace(0, hei, n_ticks), np.linspace(-100, 100, n_ticks))
    plt.xlabel('A')
    plt.imshow(pixels)


def main(infile):
    data = pd.read_csv(infile)
    
    rgb_data = data[['R', 'G', 'B']] / 255                          #normalizing data to a 0-1 scale 
    #print(rgb_data)
    X = rgb_data.as_matrix()                                        #convert dataframe to numpy-array
    
    y = data['Label']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y)       #split data into train and test subsets
    
    model_rgb = GaussianNB()                                        #create naive bayes model
    model_rgb.fit(X_train, y_train)                                 #train naive bayes model
    
    y_predicted = model_rgb.predict(X_test)                         #predict y-values with the model
    print(accuracy_score(y_test, y_predicted))                      #determines accuracy of the models predictions
    
    def rgbToLabColor(X):                                           #convert RGB colors to LAB colors
        rgbColor = X.reshape(1, -1, 3)
        labColor = color.rgb2lab(rgbColor)
        labColor = labColor.reshape(-1, 3)
        return labColor
    
    model_lab = make_pipeline(                                      #create new naive bayes model using LAB colors
        FunctionTransformer(rgbToLabColor),
        GaussianNB()
    )
    
    model_lab.fit(X_train, y_train)                                 #train the new naive bayes model
    
    y_pred = model_lab.predict(X_test)                              #predict y-values using new model
    print(accuracy_score(y_test, y_pred))
    

    plot_predictions(model_rgb)
    plt.savefig('predictions_rgb.png')
    plot_predictions(model_lab)
    plt.savefig('predictions_lab.png')


if __name__ == '__main__':
    main(sys.argv[1])
