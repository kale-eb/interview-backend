import json
import uuid
import math
from typing import Dict, Tuple, List
import openai

BEHAVIORAL_CATEGORIES = [
    "Teamwork", "Leadership", "Conflict", "Problem Solving", "Initiative",
    "Adaptability", "Failure", "Communication", "Time Management", "Integrity"
]

class ScoringEngine:
    def __init__(self, client: openai.OpenAI):
        self.client = client

    def _load_prompt_config(self):
        """Load prompt configuration from file."""
        with open("prompt_config.json", "r") as f:
            return json.load(f)

    def _get_llm_prompt(self, key: str) -> str:
        """Get a specific LLM prompt from config."""
        return self._load_prompt_config()["llm_prompts"][key]

    def score_response_with_gpt(self, response: str, conversation: List[tuple] = None) -> Dict[str, int]:
        """Score a response using GPT for all behavioral categories."""
        system_prompt = self._get_llm_prompt("scoring")
        
        # Format the input - use conversation if available, otherwise just the response
        if conversation:
            # Format conversation as Q&A pairs
            conversation_text = ""
            for question, answer in conversation:
                conversation_text += f"Q: {question}\nA: {answer}\n\n"
            input_text = f"Full conversation:\n{conversation_text}\n\nComposed story:\n{response}"
        else:
            input_text = f"Score this response: {response}"
        
        try:
            gpt_response = self.client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": input_text}
                ],
                max_tokens=200,
                temperature=0.3
            )
            scores_text = gpt_response.choices[0].message.content.strip()
            if scores_text.startswith('```json'):
                scores_text = scores_text[7:-3]
            elif scores_text.startswith('```'):
                scores_text = scores_text[3:-3]
            scores = json.loads(scores_text)
            for category in BEHAVIORAL_CATEGORIES:
                if category not in scores:
                    scores[category] = 1
                else:
                    scores[category] = int(scores[category])
            return scores
        except Exception as e:
            print(f"Error scoring with GPT: {e}")
            return {cat: 1 for cat in BEHAVIORAL_CATEGORIES}

    def summarize_story(self, response: str) -> str:
        """Generate encouraging behavioral analysis of the story."""
        system_prompt = self._get_llm_prompt("summarization")
        try:
            result = self.client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": response}
                ],
                max_tokens=200,
                temperature=0.7
            )
            return result.choices[0].message.content.strip()
        except Exception as e:
            print(f"Error generating behavioral analysis: {e}")
            return "Great story! You showed strong problem-solving and communication skills."

    def check_story_similarity(self, profile_id: str, new_summary: str, experience_summaries: dict, similarity_threshold: float = 0.8) -> Tuple[bool, str]:
        """Check if a new story is similar to existing stories using AI semantic comparison."""
        if profile_id not in experience_summaries:
            return False, None
        
        existing_summaries = experience_summaries[profile_id]["experience_summaries"]
        if not existing_summaries:
            return False, None
        
        system_prompt = self._get_llm_prompt("similarity_check")
        
        for story_hash, existing_summary in existing_summaries.items():
            try:
                response = self.client.chat.completions.create(
                    model="gpt-4o-mini",
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": f"Story 1: {new_summary}\n\nStory 2: {existing_summary}\n\nAre these the same experience?"}
                    ],
                    max_tokens=10,
                    temperature=0.1
                )
                result = response.choices[0].message.content.strip().upper()
                if result == "SAME":
                    return True, story_hash
            except Exception as e:
                print(f"Error checking story similarity: {e}")
                continue
        
        return False, None

    def calculate_composite_scores(self, user_responses: List[dict], k: float = 0.3) -> Dict[str, float]:
        """Calculate composite scores for each category using the formula."""
        scores_by_cat = {cat: [] for cat in BEHAVIORAL_CATEGORIES}
        
        for resp in user_responses:
            for cat in BEHAVIORAL_CATEGORIES:
                score = resp.get("category_scores", {}).get(cat)
                if score is not None:
                    scores_by_cat[cat].append(score)
        
        composite = {}
        for cat, scores in scores_by_cat.items():
            if scores:
                avg_quality = sum(scores) / len(scores)
                num_strong = sum(1 for s in scores if s >= 6)
                composite_score = avg_quality * (1 - math.exp(-k * num_strong))
                composite[cat] = round(composite_score, 2)
            else:
                composite[cat] = 0.0
        
        return composite

    def semantic_story_hash(self, new_response: str) -> str:
        """Generate a semantic hash for a story."""
        return str(uuid.uuid5(uuid.NAMESPACE_DNS, new_response.strip().lower()))

    def check_story_completion(self, conversation: List[tuple], story_state: dict) -> bool:
        """Check if the story is complete and ready to wrap up."""
        system_prompt = self._get_llm_prompt("story_completion_check")
        
        # Format conversation for analysis
        conversation_text = ""
        for question, answer in conversation:
            conversation_text += f"Q: {question}\nA: {answer}\n\n"
        
        try:
            response = self.client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": f"Conversation:\n{conversation_text}\n\nStory state: {story_state}\n\nIs this story complete?"}
                ],
                max_tokens=10,
                temperature=0.1
            )
            result = response.choices[0].message.content.strip().upper()
            return result == "COMPLETE"
        except Exception as e:
            print(f"Error checking story completion: {e}")
            # Default to incomplete if there's an error
            return False 