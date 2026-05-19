### How to combine multiple document loaders in Langchain?
#### Bucket Method - Create buckets as per extension and add the file as per the file type in the relevant bucket. Then load the particular document loader as per the bucket if .pdf bucket then load PyPDFLoader etc. After loading all the documents store it in a DB or local DB etc. 

