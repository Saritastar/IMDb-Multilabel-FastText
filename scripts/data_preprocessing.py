from sklearn.model_selection import train_test_split
import pandas as pd

# Load IMDb dataset
df = pd.read_csv('data/IMDb Movie Reviews Dataset.csv')

# Add '__label__' prefix to the sentiment column
df['label'] = df['sentiment'].apply(lambda x: '__label__' + x)

# Split the data into training and testing sets (80% train, 20% test)
train, test = train_test_split(df, test_size=0.2, random_state=42)

# Write the formatted train and test sets to text files for fastText
train[['label', 'review']].to_csv('data/imdb_fasttext_train.txt', index=False, sep=' ', header=False)
test[['label', 'review']].to_csv('data/imdb_fasttext_test.txt', index=False, sep=' ', header=False)
