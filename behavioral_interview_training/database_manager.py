import os
import json
import uuid
from datetime import date
from typing import Dict, List, Optional

BEHAVIORAL_CATEGORIES = [
    "Teamwork", "Leadership", "Conflict", "Problem Solving", "Initiative",
    "Adaptability", "Failure", "Communication", "Time Management", "Integrity"
]

class DatabaseManager:
    def __init__(self, data_dir: str = "behavioral_data"):
        self.data_dir = data_dir
        self.user_stats_file = os.path.join(data_dir, "user_stats.json")
        self.experience_summaries_file = os.path.join(data_dir, "experience_summaries.json")
        self.prompts_file = os.path.join(data_dir, "prompts.json")
        self.responses_file = os.path.join(data_dir, "responses.json")
        os.makedirs(data_dir, exist_ok=True)
        self._initialize_databases()

    def _initialize_databases(self):
        """Initialize empty databases if not present."""
        for f in [self.user_stats_file, self.experience_summaries_file, self.prompts_file, self.responses_file]:
            if not os.path.exists(f):
                with open(f, 'w') as out:
                    json.dump({}, out) if 'user_stats' in f or 'experience_summaries' in f else json.dump([], out)

    def _get_today(self):
        return date.today().isoformat()

    def _load_json_db(self, path, default):
        if not os.path.exists(path):
            return default
        with open(path, 'r') as f:
            return json.load(f)

    def _save_json_db(self, path, data):
        with open(path, 'w') as f:
            json.dump(data, f, indent=2)

    def list_available_profiles(self) -> List[str]:
        """List all available user profiles."""
        stats = self._load_json_db(self.user_stats_file, {})
        return list(stats.keys())

    def create_profile(self, profile_id: str) -> dict:
        """Create a new user profile."""
        stats = self._load_json_db(self.user_stats_file, {})
        summaries = self._load_json_db(self.experience_summaries_file, {})
        
        if profile_id not in stats:
            stats[profile_id] = {
                "user_id": profile_id,
                "prompts_completed": 0,
                "composite_scores": {cat: 0.0 for cat in BEHAVIORAL_CATEGORIES}
            }
            self._save_json_db(self.user_stats_file, stats)
        
        if profile_id not in summaries:
            summaries[profile_id] = {
                "experience_frequencies": {},
                "experience_summaries": {}
            }
            self._save_json_db(self.experience_summaries_file, summaries)
        
        return stats[profile_id]

    def load_user_profile(self, profile_id: str) -> dict:
        """Load a user profile, creating it if it doesn't exist."""
        stats = self._load_json_db(self.user_stats_file, {})
        if profile_id not in stats:
            return self.create_profile(profile_id)
        return stats[profile_id]

    def get_experience_data(self, profile_id: str) -> dict:
        """Get experience data for a user profile."""
        summaries = self._load_json_db(self.experience_summaries_file, {})
        if profile_id not in summaries:
            self.create_profile(profile_id)
            summaries = self._load_json_db(self.experience_summaries_file, {})
        return summaries[profile_id]

    def save_response(self, profile_id: str, response_data: dict) -> str:
        """Save a response to the database."""
        unique_id = str(uuid.uuid4())
        today = self._get_today()
        
        # Save to responses.json
        responses_db = self._load_json_db(self.responses_file, [])
        response_entry = {
            "unique_id": unique_id,
            "user_id": profile_id,
            "date": today,
            "prompt_id": unique_id,
            "prompt": response_data.get("prompt", ""),
            "main_response": response_data.get("main_response", ""),
            "category": response_data.get("category", ""),
            "category_scores": response_data.get("category_scores", {}),
            "summary": response_data.get("summary", ""),
            "conversation": response_data.get("conversation", [])
        }
        responses_db.append(response_entry)
        self._save_json_db(self.responses_file, responses_db)
        
        return unique_id

    def update_experience_summaries(self, profile_id: str, story_hash: str, summary: str):
        """Update experience summaries for a user."""
        summaries = self._load_json_db(self.experience_summaries_file, {})
        if profile_id not in summaries:
            self.create_profile(profile_id)
            summaries = self._load_json_db(self.experience_summaries_file, {})
        
        freq = summaries[profile_id]["experience_frequencies"]
        summ = summaries[profile_id]["experience_summaries"]
        freq[story_hash] = freq.get(story_hash, 0) + 1
        summ[story_hash] = summary
        self._save_json_db(self.experience_summaries_file, summaries)

    def update_user_stats(self, profile_id: str, composite_scores: Dict[str, float]):
        """Update user statistics."""
        stats = self._load_json_db(self.user_stats_file, {})
        if profile_id not in stats:
            self.create_profile(profile_id)
            stats = self._load_json_db(self.user_stats_file, {})
        
        stats[profile_id]["prompts_completed"] += 1
        stats[profile_id]["composite_scores"] = composite_scores
        self._save_json_db(self.user_stats_file, stats)

    def save_initial_prompt(self, profile_id: str, category: str, prompt: str) -> str:
        """Save an initial prompt to avoid repetition."""
        unique_id = str(uuid.uuid4())
        today = self._get_today()
        
        # Save to prompts.json
        prompts_db = self._load_json_db(self.prompts_file, [])
        prompt_entry = {
            "unique_id": unique_id,
            "user_id": profile_id,
            "date": today,
            "category": category,
            "prompt": prompt,
            "prompt_type": "initial"
        }
        prompts_db.append(prompt_entry)
        self._save_json_db(self.prompts_file, prompts_db)
        
        return unique_id

    def get_initial_prompts_for_category(self, profile_id: str, category: str) -> List[str]:
        """Get all initial prompts for a specific category and user."""
        prompts_db = self._load_json_db(self.prompts_file, [])
        user_prompts = [
            p["prompt"] for p in prompts_db 
            if p["user_id"] == profile_id 
            and p.get("category") == category 
            and p.get("prompt_type") == "initial"
        ]
        return user_prompts

    def get_user_responses(self, profile_id: str) -> List[dict]:
        """Get all responses for a user."""
        responses_db = self._load_json_db(self.responses_file, [])
        return [r for r in responses_db if r["user_id"] == profile_id]

    def get_user_progress(self, profile_id: str) -> dict:
        """Get comprehensive user progress data."""
        stats = self._load_json_db(self.user_stats_file, {})
        summaries = self._load_json_db(self.experience_summaries_file, {})
        
        if profile_id not in stats or profile_id not in summaries:
            return {
                "user_id": profile_id,
                "total_responses": 0,
                "experience_frequencies": {},
                "experience_summaries": {},
                "composite_scores": {cat: 0.0 for cat in BEHAVIORAL_CATEGORIES}
            }
        
        return {
            "user_id": profile_id,
            "total_responses": stats[profile_id]["prompts_completed"],
            "experience_frequencies": summaries[profile_id]["experience_frequencies"],
            "experience_summaries": summaries[profile_id]["experience_summaries"],
            "composite_scores": stats[profile_id]["composite_scores"]
        } 