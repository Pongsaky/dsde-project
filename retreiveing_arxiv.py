import pandas as pd

all_chunks = []

df_iterator = pd.read_json("arxiv-metadata-oai-snapshot.json", lines=True, chunksize=500000)

for i, chunk in enumerate(df_iterator):
    selected_chunk = chunk[["id", "authors", "title", "journal-ref", "categories", "abstract", "update_date"]]
    selected_chunk = selected_chunk.rename(columns={"journal-ref": "references", "categories" : "category", "update_date" : "year"})
    all_chunks.append(selected_chunk)

    print(f"Processed chunk {i}")

for i in range(6):
    all_chunks[i] = all_chunks[i].dropna()

for i in range(6):
    print(f"Chunk number is {i}")
    # display(chunk.head(5))
    print(all_chunks[i].shape)
    
combined_df = pd.concat(all_chunks, ignore_index=True)

combined_df.insert(7, "source", "arxiv") 
combined_df['year'] = combined_df['year'].str[:4]
combined_df.to_csv('arxiv_dataset.csv', index=False)

