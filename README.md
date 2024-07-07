# debarun-dutta-SPAM_DETECTION


(devtern_internships_tasks)





This project is a text classification task that aims to classify email messages as either 'ham' (not spam) or 'spam'. The project uses Python and several libraries such as TensorFlow, NumPy, Pandas, and Scikit-learn. The project begins by loading a dataset of email messages and preprocessing the text data. The text data is tokenized and converted into sequences using the Keras tokenizer. The sequences are then padded to a fixed length using the pad_sequences function from Keras. This is done to ensure that all input sequences have the same length, which is a requirement for feeding data into a neural network.
The preprocessed data is then split into training and testing sets using the train_test_split function from Scikit-learn. The training set is used to train a neural network model, while the testing set is used to evaluate the model's performance.
The neural network model used in this project is a simple feedforward neural network implemented using the Keras Sequential API. The model consists of an embedding layer, a global average pooling layer, a dense layer with ReLU activation, a dropout layer for regularization, and a dense layer with a sigmoid activation function for binary classification.The model is compiled with a binary cross-entropy loss function and the Adam optimizer. The model is trained using the fit function from Keras, and the training process is monitored using a validation set and an early stopping callback.
After training, the model's performance is evaluated using the testing set. The model's accuracy and loss are printed to the console, and a plot of the model's accuracy and validation accuracy is generated.
Finally, the project includes a function to predict whether a given email message is spam or not. The function takes a string of text as input, tokenizes and pads the input sequence, and passes it through the trained model to generate a prediction. The prediction is a probability score between 0 and 1, where a score closer to 1 indicates a higher likelihood of the message being spam.
Overall, this project demonstrates the use of deep learning techniques for text classification tasks. The project's code can be modified and extended to tackle other text classification problems, such as sentiment analysis or topic classification.
