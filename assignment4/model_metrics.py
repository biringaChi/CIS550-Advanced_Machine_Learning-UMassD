__author__ = "biringaChi"
__email__ = "biringachidera@gmail.com"

import pickle
import pandas as pd
from matplotlib import pyplot as plt

# Model metric plots
train_losses = open("train_losses.pickle", "rb")
train_loss = pickle.load(train_losses)

val_losses = open("val_losses.pickle", "rb")
val_loss = pickle.load(val_losses)

train_accuracies = open("train_accuracies.pickle", "rb")
train_accuracy = pickle.load(train_accuracies)

val_accuracies = open("val_accuracies.pickle", "rb")
val_accuracy = pickle.load(val_accuracies)

epoch_train_acc = open("epoch_train_accuracies.pickle", "rb")
epoch_train_acc = pickle.load(epoch_train_acc)

epoch_train_loss = open("epoch_train_losses.pickle", "rb")
epoch_train_loss = pickle.load(epoch_train_loss)

epoch_val_acc = open("epoch_val_accuracies.pickle", "rb")
epoch_val_acc = pickle.load(epoch_val_acc)

epoch_val_loss = open("epoch_val_losses.pickle", "rb")
epoch_val_loss = pickle.load(epoch_val_loss)

model_metrics = list(zip(train_loss, val_loss, train_accuracy, val_accuracy))

df = pd.DataFrame(model_metrics, columns=["Training_loss", "Validation_loss", "Training_accuracy", "Validation_accuracy"])
df.reset_index(inplace=True)
df = df.rename(columns={"index": "Iterations"})
# print(df)

# Best training loss curve over iterations
df.plot(kind="scatter", title="Training accuracy over iterations", x="Iterations", y="Training_accuracy", color="DarkGreen")
plt.show()

epoch_metrics = list(zip(epoch_train_acc, epoch_val_acc, epoch_train_loss, epoch_val_loss))

df2 = pd.DataFrame(epoch_metrics, columns=["Epoch_Training_accuracy", "Epoch_Validation_accuracy", "Epoch_Training_Loss", "Epoch_Validation_Loss"])
df2.reset_index(inplace=True)
df2 = df2.rename(columns={"index": "Epochs"})
# print(df2)

df2.plot(kind="scatter", title="Training Accuracy", x="Epochs", y="Epoch_Training_accuracy", color="DarkGreen")
# plt.show()
