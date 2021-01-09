# use autokeras to find a model for the insurance dataset
from numpy import asarray
from pandas import read_csv
from sklearn.model_selection import train_test_split
from autokeras import StructuredDataRegressor
# load dataset
url = 'https://raw.githubusercontent.com/jbrownlee/Datasets/master/auto-insurance.csv'
dataframe = read_csv(url, header=None)
print(dataframe.shape)
# split into input and output elements
data = dataframe.values
data = data.astype('float32')
X, y = data[:, :-1], data[:, -1]
print(X.shape, y.shape)
# separate into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=1)
print(X_train.shape, X_test.shape, y_train.shape, y_test.shape)
# define the search
search = StructuredDataRegressor(max_trials=15, loss='mean_absolute_error')
# perform the search
search.fit(x=X_train, y=y_train, verbose=0)
# evaluate the model
mae, _ = search.evaluate(X_test, y_test, verbose=0)
print('MAE: %.3f' % mae)
# use the model to make a prediction
X_new = asarray([[108]]).astype('float32')
yhat = search.predict(X_new)
print('Predicted: %.3f' % yhat[0])
# get the best performing model
model = search.export_model()
# summarize the loaded model
model.summary()
# save the best performing model to file
model.save('model_insurance.h5')
