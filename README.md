# 1st-project
# intern project here for face detection.
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_lfw_people
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical
from tensorflow.keras import layers, models

faces = fetch_lfw_people(min_faces_per_person=50, resize=0.5)
X = faces.data
y = faces.target
images = faces.images
target_names = faces.target_names

print("Number of samples:", X.shape[0])

fig, axes = plt.subplots(2, 5, figsize=(10, 5))
for i, ax in enumerate(axes.flat):
    ax.imshow(images[i], cmap='gray')
    ax.set_title(target_names[y[i]].split()[-1])
    ax.axis('off')
plt.show()

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

pca = PCA(n_components=150, whiten=True, random_state=42)
X_pca = pca.fit_transform(X_scaled)

print("PCA result shape:", X_pca.shape)

y_cat = to_categorical(y)  # One-hot encode labels

X_train, X_test, y_train, y_test = train_test_split(
    X_pca, y_cat, test_size=0.2, random_state=42
)

model = models.Sequential([
    layers.Input(shape=(150,)),
    layers.Dense(128, activation='relu'),
    layers.Dropout(0.3),
    layers.Dense(len(target_names), activation='softmax')
])

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
model.fit(X_train, y_train, epochs=25, batch_size=32, validation_split=0.1)

loss, acc = model.evaluate(X_test, y_test)
print(f"Test Accuracy: {acc * 100:.2f}%")

