# prompt: write for code training and testing where epoch =10, batch_size = 512,  and where i can plot real time training and loss function 

import matplotlib.pyplot as plt
history = {"loss": [], "accuracy": []}

for epoch in range(10):
    start = time.time()
    for i, (x, y) in enumerate(train_ds):
        # Prepare the data
        x_train = x.numpy()
        y_train = y.numpy()
        x_train = pca.transform(x_train)
        # Train the model
        pegasos_qsvc.fit(x_train, y_train)
        # Calculate the loss and accuracy
        loss = pegasos_qsvc.loss(x_train, y_train)
        accuracy = pegasos_qsvc.score(x_train, y_train)
        # Update the history
        history["loss"].append(loss)
        history["accuracy"].append(accuracy)
        # Print the progress
        clear_output(wait=True)
        print(f"Epoch: {epoch + 1}/{10}")
        print(f"Batch: {i + 1}/{len(train_ds)}")
        print(f"Loss: {loss:.4f}")
        print(f"Accuracy: {accuracy:.4f}")
    # Evaluate the model on the test set
    x_test = test_sequences_pca
    y_test = test_labels
    loss = pegasos_qsvc.loss(x_test, y_test)
    accuracy = pegasos_qsvc.score(x_test, y_test)
    print(f"Test Loss: {loss:.4f}")
    print(f"Test Accuracy: {accuracy:.4f}")
    print(f"Time taken: {time.time() - start:.2f} seconds")
    # Plot the loss and accuracy
    plt.plot(history["loss"], label="Loss")
    plt.plot(history["accuracy"], label="Accuracy")
    plt.xlabel("Iteration")
    plt.ylabel("Loss/Accuracy")
    plt.legend()
    plt.show()


# prompt: write for code training and testing where epoch =10, batch_size = 512,  and where i can plot real time training and loss function 

import matplotlib.pyplot as plt
import numpy as np
history = {'loss': [], 'accuracy': []}
for epoch in range(10):
    for i, (x_batch, y_batch) in enumerate(train_sequences_pca):
        x_batch = np.array(x_batch)
        y_batch = np.array(y_batch)
        pegasos_qsvc.fit(x_batch, y_batch)
        loss = pegasos_qsvc.loss(x_batch, y_batch)
        accuracy = pegasos_qsvc.score(x_batch, y_batch)
        history['loss'].append(loss)
        history['accuracy'].append(accuracy)
        clear_output(wait=True)
        plt.title('Training Progress')
        plt.plot(history['loss'], label='Loss')
        plt.plot(history['accuracy'], label='Accuracy')
        plt.xlabel('Iteration')
        plt.ylabel('Loss/Accuracy')
        plt.legend()
        plt.show()
    print(f"Epoch {epoch + 1}: Loss = {loss}, Accuracy = {accuracy}")

# Evaluate the model on the test set
test_loss = pegasos_qsvc.loss(test_sequences_pca, test_labels)
test_accuracy = pegasos_qsvc.score(test_sequences_pca, test_labels)
print(f"Test Loss = {test_loss}, Test Accuracy = {test_accuracy}")


# prompt: write for code training and testing where epoch =10, batch_size = 512,  and where i can plot real time training and loss function 

import matplotlib.pyplot as plt
import numpy as np
history = {"loss": [], "accuracy": []}

for epoch in range(10):
    start = time.time()
    for i, (x_train, y_train) in enumerate(train_ds):
        # preprocess data
        x_train = tf.expand_dims(x_train, -1)
        x_train = vectorize_layer(x_train)-2
        x_train = pca.transform(x_train)

        # train the model
        pegasos_qsvc.fit(x_train, y_train)

    # evaluate the model
    y_pred = pegasos_qsvc.predict(test_sequences_pca)
    accuracy = np.sum(y_pred == test_labels) / len(test_labels)

    # update history
    history["loss"].append(pegasos_qsvc.loss)
    history["accuracy"].append(accuracy)

    # print epoch results
    clear_output(wait=True)
    print(f"Epoch: {epoch + 1}")
    print(f"Loss: {history['loss'][-1]:.4f}")
    print(f"Accuracy: {history['accuracy'][-1]:.4f}")

    # plot training and loss function
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 2, 1)
    plt.plot(history["loss"], label="Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.subplot(1, 2, 2)
    plt.plot(history["accuracy"], label="Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.show()
# write for code training and testing where epoch =10, batch_size = 512,  and where i can plot real time training and testing accuracy

# prompt: write for code training and testing where epoch =10, batch_size = 512,  and where i can plot real time training and testing accuracy

import matplotlib.pyplot as plt
import numpy as np
num_epochs = 10
batch_size = 512
# Create a progress bar
print("Dataset size:", len(train_sequences_pca))
print("Number of epochs:", num_epochs)
print("Batch size:", batch_size)
print("Number of batches:", len(train_sequences_pca) // batch_size)
train_acc = []
test_acc = []
for epoch in range(num_epochs):
    # Training loop
    for i in tqdm(range(len(train_sequences_pca) // batch_size)):
        # Get the next batch
        X_batch = train_sequences_pca[i * batch_size:(i + 1) * batch_size]
        y_batch = train_labels[i * batch_size:(i + 1) * batch_size]
        # Train the model
        pegasos_qsvc.fit(X_batch, y_batch)
    # Testing loop
    correct_predictions = 0
    total_predictions = 0
    for i in range(len(test_sequences_pca) // batch_size):
        # Get the next batch
        X_batch = test_sequences_pca[i * batch_size:(i + 1) * batch_size]
        y_batch = test_labels[i * batch_size:(i + 1) * batch_size]
        # Predict the labels
        predictions = pegasos_qsvc.predict(X_batch)
        # Update the accuracy metrics
        correct_predictions += np.sum(predictions == y_batch)
        total_predictions += len(y_batch)
    # Calculate and store the accuracies
    train_accuracy = pegasos_qsvc.score(train_sequences_pca, train_labels) * 100
    test_accuracy = correct_predictions / total_predictions * 100
    train_acc.append(train_accuracy)
    test_acc.append(test_accuracy)
    # Clear the output to update the progress bar
    clear_output()
    # Print the progress
    print("Epoch:", epoch + 1)
    print("Training Accuracy:", train_accuracy)
    print("Testing Accuracy:", test_accuracy)
# Plot the training and testing accuracy
plt.plot(train_acc, label="Training Accuracy")
plt.plot(test_acc, label="Testing Accuracy")
plt.xlabel("Epoch")
plt.ylabel("Accuracy (%)")
plt.legend()
plt.show()

import matplotlib.pyplot as plt
import time
total_time = 0
history = {
  "train_loss": [],
  "test_loss": [],
  "train_accuracy": [],
  "test_accuracy": [],
}
for epoch in range(10):
  start = time.time()
  progress_bar = tqdm(total=num_epochs, desc="Training")
  for epoch in range(num_epochs):# Perform training for each batch
    for i in range(0, len(train_sequences_pca), batch_size):
        batch_sequences = train_sequences_pca[i:i+batch_size]
        batch_labels = train_labels[i:i+batch_size]
        # Perform training for the batch
        pegasos_qsvc.fit(batch_sequences, batch_labels)
        print(f"Epoch {epoch + 1} - Batch {i // batch_size + 1} completed")
        # Update the progress bar
        progress_bar.update(1)

    progress_bar.close()
    end = time.time()
    total_time += end - start
    print(f"Epoch {epoch+1}/{10}: Training time: {end - start:.3f} seconds")



    # Evaluate the model on the training and test sets
    train_loss, train_accuracy = pegasos_qsvc.score(train_sequences_pca, train_labels)
    test_loss, test_accuracy = pegasos_qsvc.score(test_sequences_pca, test_labels)  

    # Update the history dictionary
    history["train_loss"].append(train_loss)
    history["test_loss"].append(test_loss)
    history["train_accuracy"].append(train_accuracy)
    history["test_accuracy"].append(test_accuracy)


    # Plot the training and test loss
    clear_output(wait=True)
    plt.figure(figsize=(12, 6))
    plt.plot(history["train_loss"], label="Train Loss")
    plt.plot(history["test_loss"], label="Test Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.show()
    plt.savefig("training_loss.png")



    # Plot the training and test accuracy

    plt.figure(figsize=(12, 6))
    plt.plot(history["train_accuracy"], label="Train Accuracy")
    plt.plot(history["test_accuracy"], label="Test Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.show()