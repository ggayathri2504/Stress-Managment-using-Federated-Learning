import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import minmax_scale
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
df = pd.read_csv(r'C:\Users\HP\Downloads\stress_data.csv')
print(df.head())
x = df.copy();
x.drop('sl', axis = 1, inplace = True)
y = df['sl']
x = minmax_scale(x)
X_train, X_test, y_train, y_test = train_test_split(x, y,test_size=.2,random_state =123)
# Define the model architecture 5 output labels and 8 input features
model = Sequential()
model.add(Dense(64, input_shape=(8,), activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(32, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(5, activation='softmax'))

# Compile the model
model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# Train the model
model.fit(X_train, y_train, epochs=50, batch_size=16, validation_data=(X_test, y_test))
print(model.get_weights())
# Evaluate the model on the test set
loss, accuracy = model.evaluate(X_test, y_test)
print('Test accuracy:', accuracy)
#tf.keras.models.save_model(model, 'my_model')
model.save('model.h5')