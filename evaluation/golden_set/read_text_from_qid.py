from rag.retrieval import ChunkStore
import pandas as pd 
cs=ChunkStore("data//run_2//processed_chunks.parquet")

df=pd.read_csv("evaluation//data//labels.csv")

inp=input("Query name : ")

print_once=False
for row_tuple in df.loc[df["qid"]==inp].itertuples(index=False):
    print('############################################################## \n'*3)
    if not print_once:
        print_once=True
        print(f"Query:{row_tuple.query} " )
        print(f"#############################")
    print(f"Chunk ID : {row_tuple.chunk_id}" )
    print(f"#############################")
    print(f"Chunk Text : \n {cs.get(row_tuple.chunk_id).text}")
    

