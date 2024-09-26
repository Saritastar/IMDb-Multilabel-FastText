import pandas as pd
df = pd.read_csv('data/IMDb Movie Reviews Dataset.csv')
df['label'] = df['sentiment'].apply(lambda x: '__label__' + x)

# Create a file formatted for fastText
with open('data/imdb_fasttext.txt', 'w') as f:
    for row in df[['label', 'review']].itertuples(index=False):
        f.write(f"{row.label} {row.review}\n")

