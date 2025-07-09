import openai
from dotenv import load_dotenv
from database_manager import DatabaseManager
from scoring_engine import ScoringEngine
from prompt_generator import PromptGenerator
from session_manager import SessionManager

load_dotenv('config.env')
load_dotenv('../.env')

# Re-export the constant for backward compatibility
from database_manager import BEHAVIORAL_CATEGORIES

class BehavioralTrainingGPT:
    def __init__(self, data_dir: str = "behavioral_data"):
        self.client = openai.OpenAI()
        self.db_manager = DatabaseManager(data_dir)
        self.scoring_engine = ScoringEngine(self.client)
        self.prompt_generator = PromptGenerator(self.client)
        self.session_manager = SessionManager(self.db_manager, self.scoring_engine, self.prompt_generator)

    def list_available_profiles(self):
        """List all available user profiles."""
        return self.db_manager.list_available_profiles()

    def create_profile(self, profile_id: str) -> dict:
        """Create a new user profile."""
        return self.db_manager.create_profile(profile_id)

    def load_user_profile(self, profile_id: str) -> dict:
        """Load a user profile, creating it if it doesn't exist."""
        return self.db_manager.load_user_profile(profile_id)

    def get_experience_data(self, profile_id: str) -> dict:
        """Get experience data for a user profile."""
        return self.db_manager.get_experience_data(profile_id)

    def run_incremental_story_session(self, profile_id: str) -> dict:
        """Run a micro-prompt, element-based behavioral story session."""
        return self.session_manager.run_incremental_story_session(profile_id)

    def get_user_progress(self, profile_id: str) -> dict:
        """Get comprehensive user progress data."""
        return self.db_manager.get_user_progress(profile_id) 