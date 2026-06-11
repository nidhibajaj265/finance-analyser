import pandas as pd
from sentence_transformers import SentenceTransformer
from tqdm import tqdm
import yfinance as yf
import chromadb
from src.config import CHROMA_DB_PATH

def build_knowledge_base():
    db = chromadb.PersistentClient(path=CHROMA_DB_PATH)
    embedder = SentenceTransformer("all-MiniLM-L6-v2")
    
    company_collection = db.get_or_create_collection(name="sp500_companies")
    if company_collection.count() == 0:
        [id_list, text_list, metadata_list] = get_company_profiles()
        embeddings = embedder.encode(text_list).tolist()
        company_collection.add(ids=id_list,documents=text_list,embeddings=embeddings,metadatas=metadata_list)
    
    # Feel like its not needed
    # get_financial_teminology()

def get_company_profiles() -> tuple[list[str], list[str], list[dict]]:
    headers = {"User-Agent": "Mozilla/5.0"}
    url = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
    tables = pd.read_html(url, storage_options=headers)
    sp500 = tables[0]
    records = []
    
    for _,row in tqdm(sp500.iterrows(), total=len(sp500)):
        symbol = row['Symbol']
        company_name = row['Security']
        sector = row['GICS Sector']
        sub_industry = row['GICS Sub-Industry']
        
        # Fetch info from yahoo finance using company ticker
        ticker = symbol.replace('.','-')
        try:
            yf_info =  yf.Ticker(ticker).info
            description = yf_info.get('longBusinessSummary',
                                     f'Company {company_name} is a prominent company is {sector}')
        except Exception as e:
            # maybe add a retry here
            description = f"{company_name} is a prominent company in {sector}"

        records.append({
            "id" : f"sp500_{ticker}",
            "text" : f"ticker: {symbol} | Company name: {company_name} | Sector: {sector} | Sub industry: {sub_industry} | Description: {description}",
            "metadata": {
                "id": symbol,
                "Company name": company_name,
                "Sector": sector,
                "Sub industry": sub_industry,
                "Description": description
            }
            })
    id_list = [record["id"] for record in records]
    text_list = [record["text"] for record in records]
    metadata_list = [record["metadata"] for record in records]

    return (id_list, text_list, metadata_list)

def get_financial_teminology() -> None:
    pass

if __name__ == '__main__':
    build_knowledge_base()