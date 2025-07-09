from typing import Dict, List, Optional
from database_manager import DatabaseManager
from scoring_engine import ScoringEngine
from prompt_generator import PromptGenerator

class SessionManager:
    def __init__(self, db_manager: DatabaseManager, scoring_engine: ScoringEngine, prompt_generator: PromptGenerator):
        self.db_manager = db_manager
        self.scoring_engine = scoring_engine
        self.prompt_generator = prompt_generator

    def run_incremental_story_session(self, profile_id: str) -> dict:
        """Run a micro-prompt, element-based behavioral story session with smart completion detection."""
        # Load user data
        stats = self.db_manager.load_user_profile(profile_id)
        experience_data = self.db_manager.get_experience_data(profile_id)
        
        # Choose category based on lowest composite score
        composite_scores = stats["composite_scores"]
        category = min(composite_scores.keys(), key=lambda cat: composite_scores.get(cat, 0.0))
        print(f"\n🎯 Today's Focus: {category}")
        
        # Initialize story state
        story_elements = self.prompt_generator.get_story_elements()
        story_state = {el: None for el in story_elements}
        conversation = []
        
        # Generate initial question with LLM
        previous_prompts = self.db_manager.get_initial_prompts_for_category(profile_id, category)
        initial_question = self.prompt_generator.generate_initial_question(category, previous_prompts)
        print(f"\n{initial_question}")
        user_response = input("> ").strip()
        if not user_response:
            print("(You can skip, but your story will be stronger with more detail!)")
        
        # Save the initial prompt to avoid repetition
        self.db_manager.save_initial_prompt(profile_id, category, initial_question)
        
        story_state[story_elements[0]] = user_response
        conversation.append((initial_question, user_response))
        
        # For each remaining story element, generate follow-up with LLM
        for el in story_elements[1:]:
            if story_state[el]:
                continue
            
            # Check if story is already complete before asking next question
            if self.scoring_engine.check_story_completion(conversation, story_state):
                print(f"\n✨ Perfect! Your story feels complete. Let's wrap it up!")
                break
            
            # Generate follow-up question with LLM
            missing_elements = [e for e in story_elements if not story_state[e]]
            followup_question = self.prompt_generator.generate_followup_question(category, conversation, missing_elements)
            print(f"\n{followup_question}")
            user_response = input("> ").strip()
            
            # Check for early completion commands
            if user_response.lower() in ['done', 'finish', 'end', 'complete']:
                print(f"\n✨ Got it! Let's wrap up your story.")
                break
            
            if not user_response:
                print("(You can skip, but your story will be stronger with more detail!)")
                continue
            
            story_state[el] = user_response
            conversation.append((followup_question, user_response))
            
            # Check if story is complete after this response
            if self.scoring_engine.check_story_completion(conversation, story_state):
                print(f"\n✨ Great! Your story feels complete. Let's wrap it up!")
                break
            
            # Optionally, use LLM to check if the answer is vague or interesting and ask for elaboration
            if el in ["challenge", "actions", "results"]:
                followup = self.prompt_generator.generate_elaboration_prompt(category, el, user_response, story_state)
                if followup:
                    print(f"\n{followup}")
                    followup_response = input("> ").strip()
                    
                    # Check for early completion commands
                    if followup_response.lower() in ['done', 'finish', 'end', 'complete']:
                        print(f"\n✨ Got it! Let's wrap up your story.")
                        break
                    
                    if followup_response:
                        story_state[el] += "\n" + followup_response
                        conversation.append((followup, followup_response))
                        
                        # Check if story is complete after elaboration
                        if self.scoring_engine.check_story_completion(conversation, story_state):
                            print(f"\n✨ Perfect! Your story feels complete. Let's wrap it up!")
                            break
        
        # Compose the full story
        full_story = "\n".join([f"{el.capitalize()}: {story_state[el]}" for el in story_elements if story_state[el]])
        print("\n📝 Here is your complete story:")
        print("-"*40)
        print(full_story)
        print("-"*40)
        
        # Score the story
        print("\n🤖 Analyzing your story with AI...")
        category_scores = self.scoring_engine.score_response_with_gpt(full_story, conversation)
        print("\nCategory Relevance Scores:")
        for cat, score in category_scores.items():
            print(f"  {cat}: {score}")
        
        # Generate behavioral analysis
        analysis = self.scoring_engine.summarize_story(full_story)
        print(f"\n🎯 Behavioral Analysis:")
        print("-"*40)
        print(analysis)
        print("-"*40)
        
        # Save everything to database
        response_data = {
            "prompt": f"Micro-prompted story for {category}",
            "main_response": full_story,
            "category": category,
            "category_scores": category_scores,
            "summary": analysis,
            "conversation": conversation
        }
        
        unique_id = self.db_manager.save_response(profile_id, response_data)
        
        # Update experience summaries
        story_hash = self.scoring_engine.semantic_story_hash(full_story)
        self.db_manager.update_experience_summaries(profile_id, story_hash, analysis)
        
        # Update user stats with new composite scores
        user_responses = self.db_manager.get_user_responses(profile_id)
        composite_scores = self.scoring_engine.calculate_composite_scores(user_responses, k=0.3)
        self.db_manager.update_user_stats(profile_id, composite_scores)
        
        return {
            "unique_id": unique_id,
            "category": category,
            "main_response": full_story,
            "category_scores": category_scores,
            "summary": analysis,
            "conversation": conversation
        } 