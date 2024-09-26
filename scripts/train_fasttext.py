import fasttext

# Train the fastText model using the training data
model = fasttext.train_supervised(input='data/imdb_fasttext_train.txt', epoch=25, wordNgrams=2)

# Save the trained model to the 'models' folder
model.save_model('models/imdb_model.bin')

# Test the model on the separate test set
result = model.test('data/imdb_fasttext_test.txt')

# Print the test set size, precision, and recall
print(f"Test set size: {result[0]}")
print(f"Precision: {result[1]}")
print(f"Recall: {result[2]}")
