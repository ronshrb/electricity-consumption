import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import preprocessing, decomposition, metrics
import tensorflow as tf
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)  # Avoids memory overflow
        print("Using GPU:", gpus)
    except RuntimeError as e:
        print(e)
# Load Data
train_path = r"train.csv"
test_path = r"test.csv"

df_train = pd.read_csv(train_path)
df_test = pd.read_csv(test_path)

# Encode Activity Labels
activities = df_train["Activity"].unique()
class_map = {i: val for i, val in enumerate(activities)}
class_map_reverse = {val: key for key, val in class_map.items()}

df_train["Activity"] = df_train["Activity"].apply(lambda x: class_map_reverse[x])
df_test["Activity"] = df_test["Activity"].apply(lambda x: class_map_reverse[x])

df_train.drop("subject", axis=1, inplace=True)
df_test.drop("subject", axis=1, inplace=True)

# Scale Features
scaler = preprocessing.StandardScaler()
X_train_ = scaler.fit_transform(df_train.drop("Activity", axis=1).values)
X_test_ = scaler.transform(df_test.drop("Activity", axis=1).values)

y_train_ = df_train["Activity"].values
y_test_ = df_test["Activity"].values

# Apply PCA for Dimensionality Reduction
pca = decomposition.PCA(n_components=200)  # Choose based on explained variance
X_train_pca = pca.fit_transform(X_train_)
X_test_pca = pca.transform(X_test_)

# Create Sequences
time_steps = 15

def create_sequences(X, y, time_steps=5):
    X_, y_ = [], []
    n = X.shape[0]
    for i in np.arange(n - time_steps):
        X_.append(X[i:(i + time_steps)])
        y_.append(y[i + time_steps])
    return np.array(X_), np.array(y_)

X_train, y_train = create_sequences(X_train_pca, y_train_, time_steps)
X_test, y_test = create_sequences(X_test_pca, y_test_, time_steps)

print(X_train.shape, X_test.shape)

# Build GRU Model
model = tf.keras.models.Sequential([
    tf.keras.layers.Input(shape=X_train.shape[1:]),
    tf.keras.layers.GRU(units=256, return_sequences=True),
    tf.keras.layers.Dropout(0.3),
    tf.keras.layers.GRU(units=128, return_sequences=True),
    tf.keras.layers.Dropout(0.3),
    tf.keras.layers.GRU(units=64),
    tf.keras.layers.Dense(units=32, activation="relu"),
    tf.keras.layers.Dense(units=len(class_map), activation="softmax")
])

model.compile(optimizer=tf.keras.optimizers.Adam(0.005),
              loss=tf.keras.losses.SparseCategoricalCrossentropy(),
              metrics=["accuracy"])
model.summary()

# Train Model
cb = tf.keras.callbacks.EarlyStopping(patience=10, restore_best_weights=True)
history = model.fit(X_train, y_train, epochs=100, validation_split=0.3, callbacks=cb)

# Plot Training History
history_df = pd.DataFrame(history.history)
history_df[["loss", "val_loss"]].plot()
plt.xlabel("#Epochs")
plt.ylabel("Cross-Entropy Loss")
plt.show()

history_df[["accuracy", "val_accuracy"]].plot()
plt.xlabel("#Epochs")
plt.ylabel("Accuracy")
plt.show()

# Evaluate Model
model.evaluate(X_test, y_test)

y_test_pred_probs = model.predict(X_test)
y_test_pred = np.array([np.argmax(probs) for probs in y_test_pred_probs])

clf_report = metrics.classification_report(y_test, y_test_pred, digits=4)
print(clf_report)

conf_matrix = metrics.confusion_matrix(y_test, y_test_pred)
sns.heatmap(conf_matrix, annot=True, xticklabels=class_map.values(), yticklabels=class_map.values())
plt.show()
