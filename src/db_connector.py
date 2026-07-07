import pandas as pd
from sqlalchemy import create_engine
from config import DB_CONFIG

def get_engine():
    conn_str = (
        f"mysql+mysqlconnector://{DB_CONFIG['user']}:{DB_CONFIG['password']}"
        f"@{DB_CONFIG['host']}:{DB_CONFIG['port']}/{DB_CONFIG['database']}"
    )
    return create_engine(conn_str)

def fetch_data(query):
    engine = get_engine()
    df = pd.read_sql(query, engine)
    return df