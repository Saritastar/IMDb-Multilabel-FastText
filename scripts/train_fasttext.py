import fasttext

# Train the fastText model using the preprocessed data
model = fasttext.train_supervised(input='data/imdb_fasttext.txt', epoch=25, wordNgrams=2)

# Save the model to the 'models' folder
model.save_model('models/imdb_model.bin')

# Test the model on the same training data (for now, you can use a different test set later)
result = model.test('data/imdb_fasttext.txt')
print(f"Test set size: {result[0]}")
print(f"Precision: {result[1]}")
print(f"Recall: {result[2]}")
