## Data Collection
- Scrape r/wholesomegreentext using PRAW for the images and their corresponding metadata
- Perform data preprocessing and filtering
- Use embedding model like CLIP to generate embeddings
- Store on Google Drive/Huggingface Datasets

## Index Building
- Build FAISS index from these embeddings
- Save it in a shareable storage format

## Deploy Inference API
- Load the precomputed FAISS index and metadata on startup
- Generate the user query embedding
- Search top k images from FAISS embeddings
- Return matching image URLs

## User Fronted
- Create simple Nextjs frontend for the user to see the final search results
