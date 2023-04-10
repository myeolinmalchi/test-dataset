# Embed datasets and search from it.

## Scripts

-   `embed_dataset.py` : Embed .pdf files from dataset/ and vectorstore at Pinecone.
-   `search_papers.py` : Search questions from Pinecone and recommends papers.
-   `chat.py`: Search for papers related to the topic, pick one and chat with it.

## Usage

1. Clone repository and move directory.

```
git clone https://github.com/myeolinmalchi/test-dataset.git
cd test-dataset
```

1. Install dependencies.

```
pip install -r requirements.txt
```

2. Create `.env` file.

```dosini
# .env
OPENAI_API_KEY=your_openai_api_key
PINECONE_API_KEY=your_pinecone_api_key
PINECONE_ENV=youre_pinecone_environment
```

3. Create `dataset/` directory and move .pdf files.

```
mkdir dataset
```

4. Run Scripts.

```
python embed_dataset.py
python chat.py "기계학습을 활용한 특허 분쟁 예측"
```
