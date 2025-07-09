import json
from typing import List, Optional
import openai

class PromptGenerator:
    def __init__(self, client: openai.OpenAI):
        self.client = client

    def _load_prompt_config(self):
        """Load prompt configuration from file."""
        with open("prompt_config.json", "r") as f:
            return json.load(f)

    def _get_llm_prompt(self, key: str) -> str:
        """Get a specific LLM prompt from config."""
        return self._load_prompt_config()["llm_prompts"][key]

    def _get_story_elements(self) -> List[str]:
        """Get the list of story elements from config."""
        return self._load_prompt_config()["story_elements"]

    def generate_initial_question(self, category: str, previous_prompts: List[str]) -> str:
        """Generate the initial question for a behavioral category, avoiding repetition."""
        system_prompt = self._get_llm_prompt("initial_question")
        
        if previous_prompts:
            user_message = f"Category: {category}\nPrevious initial prompts for this category:\n" + "\n".join([f"- {prompt}" for prompt in previous_prompts[-3:]]) + "\n\nGenerate a NEW, different initial question that hasn't been asked before."
        else:
            user_message = f"Category: {category}\nThis is the first time asking about this category.\n\nGenerate an initial question."
        
        try:
            response = self.client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_message}
                ],
                max_tokens=100,
                temperature=0.7
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            print(f"Error generating initial question: {e}")
            return f"Tell me about a time you demonstrated {category}."

    def generate_followup_question(self, category: str, conversation: List[tuple], missing_elements: List[str]) -> str:
        """Generate a follow-up question based on the conversation so far."""
        system_prompt = self._get_llm_prompt("followup_question")
        
        # Format conversation for the LLM
        conversation_text = ""
        for question, answer in conversation:
            conversation_text += f"Q: {question}\nA: {answer}\n\n"
        
        user_message = f"Category: {category}\nConversation so far:\n{conversation_text}\nMissing elements: {missing_elements}"
        
        try:
            response = self.client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_message}
                ],
                max_tokens=100,
                temperature=0.7
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            print(f"Error generating followup question: {e}")
            return "What happened next?"

    def generate_elaboration_prompt(self, category: str, element: str, user_response: str, story_state: dict) -> Optional[str]:
        """Generate an elaboration prompt if the user's response needs more detail."""
        system_prompt = self._get_llm_prompt("elaboration").format(
            category=category, 
            element=element, 
            user_response=user_response
        )
        
        try:
            response = self.client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": f"Story so far: {story_state}"}
                ],
                max_tokens=60,
                temperature=0.7
            )
            followup = response.choices[0].message.content.strip()
            if followup and not followup.lower().startswith("no follow-up"):
                return followup
            return None
        except Exception as e:
            print(f"Error generating elaboration follow-up: {e}")
            return None

    def get_story_elements(self) -> List[str]:
        """Get the list of story elements that need to be collected."""
        return self._get_story_elements() 