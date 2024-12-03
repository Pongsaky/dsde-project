import pandas as pd

all_chunks = []

df_iterator = pd.read_json("arxiv-metadata-oai-snapshot.json", lines=True, chunksize=500000)

for i, chunk in enumerate(df_iterator):
    selected_chunk = chunk[["id", "submitter", "authors", "title", "journal-ref", "categories", "abstract", "update_date"]]
    all_chunks.append(selected_chunk)
    print(f"Processed chunk {i}")

for i in range(6):
    print(f"Chunk number is {i}")
    print(all_chunks[i]['journal-ref'].isnull().sum()/ all_chunks[i].shape[0])
    all_chunks[i] = all_chunks[i].dropna()

combined_df = pd.concat(all_chunks, ignore_index=True)
combined_df.to_csv('ref_null_droped_Arxiv.csv', index=False)
