#  import scikit learn
from sklearn.linear_model import LinearRegression
from sklearn import datasets


# loading dataset
boston = datasets.load_boston(return_X_y=False)
X = boston.data
y = boston.target

# splitting X and y into training and testing sets
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=1)

# create linear regression object
reg = LinearRegression()

# train the model using the training sets
reg.fit(X_train, y_train)

# regression coefficients
print('Coefficients: \n', reg.coef_)

# variance score: 1 means perfect prediction
print('Rscore: {}'.format(reg.score(X_test, y_test)))

print('Rscore: {}'.format(reg.predict(X_test)))