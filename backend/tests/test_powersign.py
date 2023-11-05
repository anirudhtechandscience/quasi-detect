from sklearn.model_selection import train_test_split
from sklearn.datasets import load_iris
from backend.src.powersign import powerSign
import tensorflow as tf
import keras

ps = powerSign(0.1, 0.2, 0.01, 1e-5)


def testVarInitializing():
    assert ps.alpha == 0.1
    assert ps.beta == 0.2
    assert ps.learningRate == 0.01
    assert ps.epsilon == 1e-5


def testgetConfig():
    gc = ps.get_config()
    assert gc["alpha"] == ps.alpha
    assert gc["beta"] == ps.beta
    assert gc["learningRate"] == ps.learningRate
    assert gc["epsilon"] == ps.epsilon
    assert gc["name"] == ps.name


def testOptimizer():

    iris = load_iris()
    x = iris['data']
    y = keras.utils.to_categorical(iris['target'])

    # Split the data into training and test sets
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

    # Define a simple model
    model = keras.Sequential([
        keras.layers.Dense(32, activation='relu', input_shape=(4,)),
        keras.layers.Dense(3, activation='softmax'),
    ])
    model.compile(optimizer=powerSign(),
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    model.fit(x_train, y_train, epochs=10, batch_size=32)
    loss_powersign, accuracy_powersign = model.evaluate(x_test, y_test)
    model.compile(optimizer='adam',
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    model.fit(x_train, y_train, epochs=10, batch_size=32)
    loss_adam, accuracy_adam = model.evaluate(x_test, y_test)
    assert accuracy_powersign > accuracy_adam  # Allow a tolerance of 0.1
