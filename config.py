import os
from dotenv import load_dotenv

load_dotenv()

class Config:
    GROQ_API_KEY = os.getenv('GROQ_API_KEY')
    
    # Data paths
    DATA_DIR = 'data'
    EMBEDDINGS_PATH = os.path.join(DATA_DIR, 'embeddings.pkl')
    MODEL_PATH = os.path.join(DATA_DIR, 'model.pkl')
    SHOWS_PATH = os.path.join(DATA_DIR, 'shows_data.pkl')
    
    # Recommendation settings
    DEFAULT_VOTE_COUNT = 100
    DEFAULT_VOTE_AVERAGE = 6.0
    DEFAULT_MAX_SEASONS = 20
    DEFAULT_MAX_EPISODES = 500
    TOP_N_RECOMMENDATIONS = 6