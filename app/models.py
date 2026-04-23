import pickle
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from config import Config
import os
import warnings
warnings.filterwarnings('ignore')

# Import LLM components
from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import PydanticOutputParser
from pydantic import BaseModel, Field

# Define the Explanation Schema
class Explanation(BaseModel):
    explanation: str = Field(..., description="A one-sentence reason for why a recommended show is a good match.")

class RecommendationModel:
    _instance = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialize()
        return cls._instance
    
    def _initialize(self):
        """Load all precomputed data"""
        print("Loading recommendation data...")
        
        # Load shows data
        shows_path = Config.SHOWS_PATH
        if not os.path.exists(shows_path):
            raise FileNotFoundError(f"Shows data file not found at {shows_path}")
        
        with open(shows_path, 'rb') as f:
            self.shows = pickle.load(f)
        print(f"✓ Loaded {len(self.shows)} shows")
        
        # Load embeddings
        self.embeddings = self._load_embeddings()
        print(f"✓ Loaded embeddings with shape: {self.embeddings.shape}")
        
        # Initialize Groq LLM for explanations
        self._init_llm()
        
        print("Recommendation system ready!")
    
    def _load_embeddings(self):
        """Load embeddings from various formats"""

        npy_path = os.path.join(Config.DATA_DIR, 'embeddings_array.npy')
        if os.path.exists(npy_path):
            return np.load(npy_path, allow_pickle=True)
        
        pkl_path = Config.EMBEDDINGS_PATH
        if os.path.exists(pkl_path):
            with open(pkl_path, 'rb') as f:
                data = pickle.load(f)
            if isinstance(data, list):
                return np.array(data)
            return data
        
        raise FileNotFoundError("No embeddings file found")
    
    def _init_llm(self):
        """Initialize Groq LLM for generating explanations"""
        try:
            os.environ['GROQ_API_KEY'] = Config.GROQ_API_KEY
            self.llm = ChatGroq(model="llama-3.3-70b-versatile")
            self.parser = PydanticOutputParser(pydantic_object=Explanation)
            self.llm_available = True
            print("✓ LLM initialized for explanations")
        except Exception as e:
            print(f"⚠️ LLM not available: {e}")
            self.llm_available = False
    
    def get_explanation(self, target_row, rec_row):
        """Generate AI explanation for why two shows are similar"""
        if not self.llm_available:
            return "Similar themes and storytelling style."
        
        template = """
        You are a razor-sharp TV critic known for your witty, insightful observations, 
        and explaining why a fan of "{target_name}" would love "{rec_name}".
        
        ---
        METADATA FOR ANALYSIS:
        TARGET SHOW: {target_name}
        - Genres: {target_genres}
        - Creator/Team: {target_creator}
        - Synopsis: {target_overview}
        
        RECOMMENDED SHOW: {rec_name}
        - Genres: {rec_genres}
        - Creator/Team: {rec_creator}
        - Synopsis: {rec_overview}
        ---
        
        INSTRUCTIONS:
        1. IDENTIFY THE HOOK by looking for shared universes, shared creators, or strong thematic parallels.
        2. USE EXTERNAL LORE of these shows' plots, character arcs, and cultural impact.
        3. Be SPECIFIC about plot elements, character traits, or storytelling techniques
        4. Use ACTIVE, vivid language (not "this show has" or "features")
        5. NO generic phrases like "captivating storyline", "must-watch", "rollercoaster"
        6. NO starting with "Fans of X will enjoy..." or "If you liked X..."
        7. Be CONCISE and sound like a knowledgeable friend, not a marketing robot
        
        Write ONE sharp, specific explanation that makes me instantly want to watch {rec_name}.
        
        {format_instructions}
        
        JSON RESPONSE:"""
        
        prompt = PromptTemplate(
            input_variables=["target_name", "target_overview", "target_genres", "target_creator", 
                           "rec_name", "rec_overview", "rec_genres", "rec_creator"],
            template=template,
            partial_variables={"format_instructions": self.parser.get_format_instructions()}
        )
        
        chain = prompt | self.llm | self.parser
        
        try:
            response = chain.invoke({
                "target_name": target_row.get('name', ''),
                "target_overview": target_row.get('overview', '')[:500],
                "target_genres": ', '.join(target_row.get('genres', [])),
                "target_creator": ', '.join(target_row.get('created_by', [])) if target_row.get('created_by') else 'Various',
                "rec_name": rec_row.get('name', ''),
                "rec_overview": rec_row.get('overview', '')[:500],
                "rec_genres": ', '.join(rec_row.get('genres', [])),
                "rec_creator": ', '.join(rec_row.get('created_by', [])) if rec_row.get('created_by') else 'Various'
            })
            return response.explanation
        except Exception as e:
            print(f"Explanation generation failed: {e}")
            return f"Both shows deliver gripping {', '.join(rec_row.get('genres', ['storytelling']))} with complex characters."
    
    def get_show_by_name(self, name):
        """Find a show by its name"""
        mask = self.shows['name'].str.lower() == name.lower()
        matches = self.shows[mask]
        return matches.iloc[0].to_dict() if len(matches) > 0 else None
    
    def search_shows(self, query, limit=15):
        """Search shows by name with partial matching (contains)"""
        # Case-insensitive contains search
        mask = self.shows['name'].str.lower().str.contains(query.lower(), na=False, regex=False)
        results = self.shows[mask].head(limit)
        
        # If no results, try word-by-word matching
        if len(results) == 0:
            words = query.lower().split()
            for word in words:
                if len(word) > 2:
                    word_mask = self.shows['name'].str.lower().str.contains(word, na=False, regex=False)
                    if word_mask.any():
                        results = self.shows[word_mask].head(limit)
                        break
        
        return results.to_dict('records')
    
    def get_similar_shows(self, show_name, vote_count=0, vote_average=0, 
                          max_seasons=20, max_episodes=500, top_n=6):
        """Get similar shows with AI explanations"""
        
        # Find the target show
        mask = self.shows['name'].str.lower() == show_name.lower()
        if not mask.any():
            return None
        
        idx = mask[mask].index[0]
        target_row = self.shows.iloc[idx]
        target_embedding = self.embeddings[idx].reshape(1, -1)
        
        # Filter shows
        filtered_mask = (
            (self.shows['vote_count'] >= vote_count) &
            (self.shows['vote_average'] >= vote_average) &
            (self.shows['number_of_seasons'] <= max_seasons) &
            (self.shows['number_of_episodes'] <= max_episodes) &
            (self.shows.index != idx)
        )
        
        filtered_indices = self.shows[filtered_mask].index
        filtered_embeddings = self.embeddings[filtered_indices]
        
        if len(filtered_indices) == 0:
            return {'target_show': self._format_show(target_row), 'recommendations': []}
        
        # Calculate similarities
        similarities = cosine_similarity(target_embedding, filtered_embeddings)[0]
        
        # Get top matches
        top_indices = np.argsort(similarities)[-top_n:][::-1]
        
        recommendations = []
        for i in top_indices:
            rec_idx = filtered_indices[i]
            rec_row = self.shows.iloc[rec_idx]
            
            # Generate AI explanation
            explanation = self.get_explanation(target_row, rec_row)
            
            recommendations.append(self._format_show(rec_row, 
                similarity_score=float(similarities[i]), 
                explanation=explanation))
        
        return {
            'target_show': self._format_show(target_row),
            'recommendations': recommendations
        }
    
    def _format_show(self, row, similarity_score=None, explanation=None):
        """Format a show row for JSON response"""
        overview = row.get('overview', '')
        if not isinstance(overview, str):
            overview = ''
        
        # Handle genres
        genres = row.get('genres', [])
        if not isinstance(genres, list):
            if isinstance(genres, str):
                genres = [g.strip() for g in genres.split(',') if g.strip()]
            else:
                genres = []
        
        # Handle creator
        created_by = row.get('created_by', [])
        if not isinstance(created_by, list):
            if isinstance(created_by, str):
                created_by = [c.strip() for c in created_by.split(',') if c.strip()]
            else:
                created_by = []
        
        result = {
            'id': int(row.get('id', 0)),
            'name': str(row.get('name', '')),
            'overview': overview,
            'genres': genres,
            'poster_path': row.get('poster_path', ''),
            'vote_average': float(row.get('vote_average', 0)),
            'vote_count': int(row.get('vote_count', 0)),
            'number_of_seasons': int(row.get('number_of_seasons', 0)),
            'number_of_episodes': int(row.get('number_of_episodes', 0)),
            'created_by': created_by
        }
        
        if similarity_score is not None:
            result['similarity_score'] = similarity_score
        
        if explanation is not None:
            result['explanation'] = explanation
        
        return result

# Singleton instance
recommendation_model = RecommendationModel()