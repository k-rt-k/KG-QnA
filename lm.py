## code for loading and training the language model
import transformers
import torch
import langchain
import pinecone
import time
from langchain.embeddings.openai import OpenAIEmbeddings
import pandas as pd

def node_to_ans(nodename:str)->str:
    return nodename.split(':')[1].replace('_',' ')

### somehow scrape dataset.txt for finetuning the lm ###
def get_lm_data():
    pass

def parse_output(response:str):
    pass

### an LM, when given a query, extracts entities and relationships from the query ###
class ParserLM:
    def __init__(self,lm_name:str='flan-t5-base',tokenizer_name:str='flan-t5-base'):
        self.lm = transformers.AutoModelForSequenceClassification.from_pretrained(lm_name)
        self.tokenizer = transformers.AutoTokenizer.from_pretrained(tokenizer_name)
        self.lm.eval()
        self.lm.to('cuda')
        self.tokenizer.to('cuda')
        return
    def finetune(self,data)->None:
        raise NotImplementedError
        pass
    def parse(self,query:str)->str:
        encoded_text = self.tokenizer(query, return_tensors="pt")
        raise NotImplementedError
    


### if the rag model is to be extended with the description this will be needed ###
class Embedder: ## generate contextual embeddings and ids for given input texts, then return closest matches on queries
    def __init__(self,index_name:str,emb_dim=1536)->bool:
        self.index_name = index_name
        new = False
        if index_name not in pinecone.list_indexes():
            pinecone.create_index(
                index_name,
                dimension=emb_dim,
                metric='cosine'
            )
            while not pinecone.describe_index(index_name).status['ready']:
                time.sleep(1)
            new = True
        self.index = pinecone.Index(index_name)
        print(f"Index stats: {self.index.describe()}")

        self.embed_model = OpenAIEmbeddings(model="text-embedding-ada-002")## this is the model we are using for now, change it later
        return new
    
    def push(self,text:str)->str:
        raise NotImplementedError
        pass
    
    def compile(self)->None:
        self.vectorstore = Pinecone(
            self.index, embed_model.embed_query, text_field
        )
        return
    def query(self,q:str,top_k:int)->list[str]:
        return self.vectorstore.similarity_search(q,k=top_k)