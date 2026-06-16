import tensorflow as tf
import numpy as np

model = tf.keras.models.load_model(
    "models/lhs_rhs_model.keras"
)

img = tf.keras.preprocessing.image.load_img(
    "sample.png",
    target_size=(224,224)
)

img = tf.keras.preprocessing.image.img_to_array(img)

img = np.expand_dims(img, axis=0)

prediction = model.predict(img)

lhs_prob = float(prediction[0][0]) * 100
rhs_prob = float(prediction[0][1]) * 100

print("\nPrediction Results")
print(f"LHS : {lhs_prob:.2f}%")
print(f"RHS : {rhs_prob:.2f}%")

if lhs_prob > rhs_prob:
    print("\nFinal Prediction : LHS")
else:
    print("\nFinal Prediction : RHS")