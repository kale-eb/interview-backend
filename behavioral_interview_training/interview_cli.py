import os
import sys
from datetime import datetime, date
from typing import Dict, List
import json
from interview_trainer import BehavioralTrainingGPT, BEHAVIORAL_CATEGORIES

class BehavioralCLI:
    def __init__(self):
        self.app = BehavioralTrainingGPT()
        self.current_profile = None
    
    def display_menu(self):
        """Display the main menu options."""
        print("\n" + "="*60)
        print("🧠 BEHAVIORAL INTERVIEW TRAINING APP (GPT-4o-mini)")
        print("="*60)
        print("1. Start Daily Training Session")
        print("2. View Progress Dashboard")
        print("3. View Response History")
        print("4. Switch Profile")
        print("5. Create New Profile")
        print("6. Delete Profile")
        print("7. Export Data")
        print("8. Help")
        print("0. Exit")
        print("-"*60)
    
    def get_user_selection(self) -> str:
        """Get user menu selection."""
        return input("Select an option (0-8): ").strip()
    
    def get_multiline_input(self, prompt: str = "Enter your response:") -> str:
        """Get multi-line input from user."""
        print(f"\n{prompt}")
        print("💡 Tip: You can add multiple lines. Press Enter twice when done, or submit empty line to continue.")
        
        lines = []
        while True:
            line = input("> ").strip()
            if not line:  # Empty line means done
                break
            lines.append(line)
        
        return "\n".join(lines)
    
    def analyze_and_save_responses(self, main_response: str, followup_response: str, category: str):
        """Analyze and save both main and follow-up responses together."""
        print("\n🤖 Analyzing responses with AI...")
        
        # Combine responses for analysis
        combined_response = main_response
        if followup_response:
            combined_response += "\n\nFollow-up Response:\n" + followup_response
        
        # Score the combined response
        category_scores = self.app.score_response_with_gpt(combined_response)
        
        # Display results
        print(f"\n📊 Response Analysis:")
        print("Category Relevance Scores (1-10):")
        for cat, score in category_scores.items():
            if score >= 3:  # Only show relevant scores
                print(f"  {cat}: {score}/10")
        
        # Save main response with scores
        today = date.today().isoformat()
        main_data = {
            "user_id": self.current_profile,
            "date": today,
            "response": main_response,
            "category_scores": category_scores,
            "timestamp": datetime.now().isoformat(),
            "is_followup": False
        }
        
        main_file = os.path.join(self.app.responses_dir, f"{self.current_profile}_{today}.json")
        with open(main_file, 'w') as f:
            json.dump(main_data, f, indent=2)
        
        # Save follow-up response if provided
        if followup_response:
            followup_data = {
                "user_id": self.current_profile,
                "date": today,
                "response": followup_response,
                "category_scores": category_scores,  # Same scores for consistency
                "timestamp": datetime.now().isoformat(),
                "is_followup": True
            }
            
            followup_file = os.path.join(self.app.responses_dir, f"{self.current_profile}_{today}_followup.json")
            with open(followup_file, 'w') as f:
                json.dump(followup_data, f, indent=2)
        
        # Update user profile
        self.app.update_user_profile(self.current_profile, category_scores)
        
        # Show updated profile
        updated_profile = self.app.load_user_profile(self.current_profile)
        print(f"\n📈 Updated Composite Scores:")
        for composite_key, score in updated_profile["composite_scores"].items():
            if score > 0:  # Only show categories with responses
                category_name = composite_key.replace("Composite_", "").replace("_Score", "")
                print(f"  {category_name}: {score}")
        
        print(f"\n✅ Session completed successfully!")
        print(f"Category practiced: {category}")
    
    def show_profile_selection(self):
        """Show profile selection interface."""
        available_profiles = self.app.list_available_profiles()
        
        if available_profiles:
            print(f"\n📁 Available Profiles: {', '.join(available_profiles)}")
        else:
            print("\n📁 No existing profiles found.")
            create_new = input("Would you like to create a new profile? (y/n): ").strip().lower()
            if create_new == 'y':
                self.create_new_profile()
                return
        
        profile_id = input("Enter profile ID (or press Enter for profile 1): ").strip()
        if not profile_id:
            profile_id = "1"
        
        # Create profile if it doesn't exist
        if profile_id not in available_profiles:
            create_choice = input(f"Profile {profile_id} doesn't exist. Create it? (y/n): ").strip().lower()
            if create_choice == 'y':
                self.app.create_profile(profile_id)
            else:
                return
        
        self.current_profile = profile_id
        print(f"\nWelcome to Profile {profile_id}!")
        
        # Show current progress
        self.show_progress_summary()
    
    def create_new_profile(self):
        """Create a new profile."""
        next_id = self.app.get_next_available_profile_id()
        print(f"\n🆕 Creating new profile...")
        
        custom_id = input(f"Enter profile ID (or press Enter for {next_id}): ").strip()
        if not custom_id:
            profile_id = next_id
        else:
            profile_id = custom_id
        
        # Check if profile already exists
        if profile_id in self.app.list_available_profiles():
            print(f"Profile {profile_id} already exists!")
            return
        
        # Create the profile
        self.app.create_profile(profile_id)
        print(f"✅ Profile {profile_id} created successfully!")
        
        self.current_profile = profile_id
        print(f"\nWelcome to Profile {profile_id}!")
    
    def delete_profile(self):
        """Delete a profile."""
        available_profiles = self.app.list_available_profiles()
        
        if not available_profiles:
            print("No profiles to delete.")
            return
        
        print(f"\n🗑️  Delete Profile")
        print(f"Available profiles: {', '.join(available_profiles)}")
        
        profile_id = input("Enter profile ID to delete: ").strip()
        
        if profile_id not in available_profiles:
            print(f"Profile {profile_id} not found.")
            return
        
        confirm = input(f"Are you sure you want to delete Profile {profile_id}? This cannot be undone. (y/n): ").strip().lower()
        if confirm == 'y':
            if self.app.delete_profile(profile_id):
                print(f"✅ Profile {profile_id} deleted successfully!")
                if self.current_profile == profile_id:
                    self.current_profile = None
            else:
                print(f"❌ Error deleting profile {profile_id}")
    
    def switch_profile(self):
        """Switch to a different profile."""
        self.show_profile_selection()
    
    def show_progress_summary(self):
        """Show a quick progress summary."""
        if not self.current_profile:
            print("No profile selected.")
            return
        progress = self.app.get_user_progress(self.current_profile)
        print(f"\n📊 Quick Progress Summary:")
        print(f"Total Responses: {progress['total_responses']}")
        freq = progress['experience_frequencies']
        summaries = progress['experience_summaries']
        if not freq:
            print("No experiences recorded yet.")
            return
        # Sort by frequency
        sorted_exps = sorted(freq.items(), key=lambda x: x[1], reverse=True)
        print(f"Unique Experiences: {len(sorted_exps)}")
        print(f"Top Experiences:")
        for i, (exp_hash, count) in enumerate(sorted_exps[:2], 1):
            summary = summaries.get(exp_hash, '[No summary]')
            print(f"  {i}. {summary} (x{count})")
        print("\nAll Experiences:")
        print("-"*40)
        for exp_hash, count in sorted_exps:
            summary = summaries.get(exp_hash, '[No summary]')
            print(f"{summary:35.35} | {count} time(s)")
    
    def run_daily_session(self):
        """Run a daily training session with GPT using the new incremental micro-prompt system."""
        if not self.current_profile:
            print("Please select a profile first (option 4).")
            return
        print(f"\n🚀 Starting Daily Training Session for Profile {self.current_profile}...")
        print("🤖 Using GPT-4o-mini for micro-prompted story building...")
        try:
            session_result = self.app.run_incremental_story_session(self.current_profile)
            print(f"\n✅ Story completed!")
            print(f"Category practiced: {session_result['category']}")
            print(f"\nSummary of your story: {session_result.get('summary', '')}")
            print(f"\nSession data saved.\n")
            print("Session scores:")
            for cat, score in session_result.get("category_scores", {}).items():
                print(f"  {cat}: {score}")
            print("\nSession data saved.\n")
            input("\nPress Enter to continue...")
        except Exception as e:
            print(f"Error during session: {e}")
    
    def handle_followup_question(self, session_result: Dict):
        """Handle the follow-up question from the session."""
        print(f"\n💭 Follow-up Question:")
        print(f"{session_result['follow_up']}")
        
        followup_response = self.get_multiline_input("Your follow-up response:")
        
        if followup_response:
            # Analyze both responses together
            self.analyze_and_save_responses(session_result['response'], followup_response, session_result['category'])
        else:
            # If no follow-up response, analyze just the main response
            self.analyze_and_save_responses(session_result['response'], "", session_result['category'])
    
    def show_progress_dashboard(self):
        """Show detailed progress dashboard with composite scores, prompts completed, and experience summaries."""
        if not self.current_profile:
            print("Please select a profile first (option 4).")
            return
        progress = self.app.get_user_progress(self.current_profile)
        print(f"\n📈 PROGRESS DASHBOARD - Profile {self.current_profile}")
        print("="*60)
        print(f"Prompts Completed: {progress['total_responses']}")
        print(f"\nComposite Scores (Weighted, higher = better):")
        print("-"*50)
        composite_scores = progress.get('composite_scores', {})
        for cat in BEHAVIORAL_CATEGORIES:
            score = composite_scores.get(cat, 0.0)
            print(f"  {cat:<16}: {score:4.2f}/10")
        freq = progress['experience_frequencies']
        summaries = progress['experience_summaries']
        if not freq:
            print("No experiences recorded yet.")
            return
        # Sort by frequency
        sorted_exps = sorted(freq.items(), key=lambda x: x[1], reverse=True)
        print(f"\nUnique Experiences: {len(sorted_exps)}")
        print(f"Top Experiences:")
        for i, (exp_hash, count) in enumerate(sorted_exps[:2], 1):
            summary = summaries.get(exp_hash, '[No summary]')
            print(f"  {i}. {summary} (x{count})")
        print("\nAll Experiences:")
        print("-"*40)
        for exp_hash, count in sorted_exps:
            summary = summaries.get(exp_hash, '[No summary]')
            print(f"{summary:35.35} | {count} time(s)")
    
    def show_recent_responses(self, limit: int = 5):
        """Show recent user responses."""
        print(f"\n📝 Recent Responses (last {limit}):")
        print("-"*40)
        
        # Get all response files for this profile
        response_files = []
        for filename in os.listdir(self.app.responses_dir):
            if filename.startswith(f"{self.current_profile}_") and filename.endswith(".json"):
                response_files.append(filename)
        
        # Sort by date (newest first)
        response_files.sort(reverse=True)
        
        for i, filename in enumerate(response_files[:limit]):
            filepath = os.path.join(self.app.responses_dir, filename)
            with open(filepath, 'r') as f:
                response_data = json.load(f)
            
            date_str = response_data.get('date', 'Unknown')
            category_scores = response_data.get('category_scores', {})
            response_text = response_data.get('response', '')[:100] + "..." if len(response_data.get('response', '')) > 100 else response_data.get('response', '')
            is_followup = response_data.get('is_followup', False)
            
            # Find highest scoring category
            if category_scores:
                top_category = max(category_scores, key=category_scores.get)
                top_score = category_scores[top_category]
                print(f"{i+1}. {date_str} - {top_category} ({top_score}/10)" + (" [Follow-up]" if is_followup else ""))
                print(f"   {response_text}")
                print()
    
    def show_response_history(self):
        """Show detailed response history for the current user from responses.json."""
        if not self.current_profile:
            print("Please select a profile first (option 4).")
            return
        print(f"\n📚 RESPONSE HISTORY - Profile {self.current_profile}")
        print("="*60)
        # Read all responses for this user from responses.json
        responses_path = os.path.join(self.app.data_dir, "responses.json")
        if not os.path.exists(responses_path):
            print("No responses found. Complete your first training session!")
            return
        with open(responses_path, 'r') as f:
            all_responses = json.load(f)
        user_responses = [r for r in all_responses if r.get("user_id") == self.current_profile]
        if not user_responses:
            print("No responses found. Complete your first training session!")
            return
        # Sort by date (newest first)
        user_responses.sort(key=lambda r: r.get('date', ''), reverse=True)
        for i, resp in enumerate(user_responses, 1):
            date_str = resp.get('date', 'Unknown')
            category = resp.get('category', 'Unknown')
            summary = resp.get('summary', '')
            print(f"\n{i}. {date_str} - {category}")
            print(f"   Summary: {summary}")
            print(f"   Main Response: {resp.get('main_response', '')[:100]}{'...' if len(resp.get('main_response', '')) > 100 else ''}")
            if resp.get('followup_response'):
                print(f"   Follow-up: {resp.get('followup_response', '')[:100]}{'...' if len(resp.get('followup_response', '')) > 100 else ''}")
            print(f"   Category Scores: {resp.get('category_scores', {})}")

    def export_data(self):
        """Export user data to JSON file."""
        if not self.current_profile:
            print("Please select a profile first (option 4).")
            return
        
        export_filename = f"profile_{self.current_profile}_gpt_export_{date.today().isoformat()}.json"
        
        # Collect all user data
        export_data = {
            "user_profile": self.app.load_user_profile(self.current_profile),
            "responses": [],
            "prompts": []
        }
        
        # Get all responses
        for filename in os.listdir(self.app.responses_dir):
            if filename.startswith(f"{self.current_profile}_") and filename.endswith(".json"):
                filepath = os.path.join(self.app.responses_dir, filename)
                with open(filepath, 'r') as f:
                    export_data["responses"].append(json.load(f))
        
        # Get all prompts (for reference)
        for filename in os.listdir(self.app.prompts_dir):
            if filename.endswith(".json"):
                filepath = os.path.join(self.app.prompts_dir, filename)
                with open(filepath, 'r') as f:
                    export_data["prompts"].append(json.load(f))
        
        # Save export file
        with open(export_filename, 'w') as f:
            json.dump(export_data, f, indent=2)
        
        print(f"\n✅ Data exported to: {export_filename}")
        print(f"Total responses: {len(export_data['responses'])}")
        print(f"Total prompts: {len(export_data['prompts'])}")
    
    def show_help(self):
        """Show help information."""
        print("\n📖 BEHAVIORAL TRAINING APP HELP (GPT Version)")
        print("="*60)
        print("This app uses GPT-4o-mini to generate prompts and score responses.")
        print()
        print("Profile System:")
        print("• Simple numeric profiles (1, 2, 3, etc.)")
        print("• Create new profiles as needed")
        print("• Switch between profiles easily")
        print("• Delete profiles when no longer needed")
        print()
        print("Key Features:")
        print("• AI-powered prompt generation for each category")
        print("• Adaptive follow-up questions based on your response")
        print("• Multi-category scoring (1-10 scale for each category)")
        print("• Composite scores calculated as total/10")
        print("• Only scores >= 3 are recorded for each category")
        print()
        print("Database Structure:")
        print("• UUID: Unique identifier for each profile")
        print("• Composite_[Category]_Score: Total score / 10 for each category")
        print("• [Category]_responses: Array of response data for each category")
        print()
        print("Scoring System (EXTREMELY STRICT):")
        print("• 10: EXCEPTIONAL - Perfect demonstration of this specific area with specific actions, clear outcomes, deep insight")
        print("• 8-9: EXCELLENT - Strong demonstration with concrete examples and measurable results")
        print("• 6-7: GOOD - Very clear demonstration but is not the main focus of the story")
        print("• 4-5: FAIR - Some demonstration but is not the main focus of the story")
        print("• 2-3: WEAK - Very minimal relevance to the response, mostly general statements")
        print("• 1: POOR - No meaningful demonstration, just mentions or vague references")
        print("• Most responses will score 1-4. High scores (7-10) should only be given if the response is DIRECTLY relevant to the category")
        print("• Only scores >= 3 are recorded")
        print()
        print("Behavioral Categories:")
        for i, category in enumerate(BEHAVIORAL_CATEGORIES, 1):
            print(f"  {i:2d}. {category}")
        print()
        print("Tips for Better Responses:")
        print("• Be specific with examples and outcomes")
        print("• Use the STAR method (Situation, Task, Action, Result)")
        print("• Include metrics and measurable results when possible")
        print("• Focus on what you learned and how you grew")
        print("• Draw from any life experience: work, school, personal life, sports, volunteer work, travel, family, hobbies, etc.")
        print("📚 Advanced Features:")
        print("• AI-powered prompt generation for each category")
        print("• Multi-category scoring (0-10 scale for each category)")
        print("• Semantic story similarity detection to encourage diversity")
        print("• Frequency penalties for repeated similar experiences")
        print("• Composite score calculation with exponential weighting")
        print("• Progress tracking with detailed analytics")
        print("")
        print("🔄 Story Diversity System:")
        print("• The AI detects when you share similar experiences")
        print("• Repeated stories receive exponentially reduced scores")
        print("• Frequency penalty: score × (1 / frequency^1.5)")
        print("• This encourages building a diverse portfolio of examples")
        print("")

    def run(self):
        """Main CLI loop."""
        print("Welcome to the Behavioral Interview Training App (GPT-4o-mini)!")
        print("Make sure you have set your OpenAI API key in config.env")
        
        # Initial profile selection
        self.show_profile_selection()
        
        while True:
            self.display_menu()
            choice = self.get_user_selection()
            
            if choice == '0':
                print("Thank you for using the Behavioral Interview Training App!")
                break
            elif choice == '1':
                self.run_daily_session()
            elif choice == '2':
                self.show_progress_dashboard()
            elif choice == '3':
                self.show_response_history()
            elif choice == '4':
                self.switch_profile()
            elif choice == '5':
                self.create_new_profile()
            elif choice == '6':
                self.delete_profile()
            elif choice == '7':
                self.export_data()
            elif choice == '8':
                self.show_help()
            else:
                print("Invalid option. Please try again.")
            
            input("\nPress Enter to continue...")

def main():
    """Main function to run the CLI."""
    cli = BehavioralCLI()
    try:
        cli.run()
    except KeyboardInterrupt:
        print("\n\nGoodbye! Keep practicing those behavioral questions!")
    except Exception as e:
        print(f"\nAn error occurred: {e}")
        print("Please check that your OpenAI API key is set in config.env")

if __name__ == "__main__":
    main() 