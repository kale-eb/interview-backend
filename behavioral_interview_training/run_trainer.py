#!/usr/bin/env python3
"""
Behavioral Interview Training App Launcher (GPT-4o-mini)

This script provides easy access to the GPT-powered behavioral training app.
Make sure to set your OpenAI API key in config.env before running.
"""

import sys
import os
from interview_trainer import BehavioralTrainingGPT
from interview_cli import BehavioralCLI

def show_launcher_menu():
    """Show the launcher menu."""
    print("\n" + "="*60)
    print("🚀 BEHAVIORAL INTERVIEW TRAINING APP LAUNCHER (GPT-4o-mini)")
    print("="*60)
    print("1. Launch Full CLI Interface")
    print("2. Quick Daily Training Session")
    print("3. Check Progress Only")
    print("4. View Help")
    print("0. Exit")
    print("-"*60)

def quick_training():
    """Run a quick daily training session with GPT."""
    print("\n🚀 Quick Daily Training Session (GPT-4o-mini)")
    print("="*50)
    
    app = BehavioralTrainingGPT()
    
    # Show available profiles
    available_profiles = app.list_available_profiles()
    if available_profiles:
        print(f"📁 Available Profiles: {', '.join(available_profiles)}")
    else:
        print("📁 No existing profiles found.")
    
    # Get profile ID
    profile_id = input("Enter profile ID (or press Enter for profile 1): ").strip()
    if not profile_id:
        profile_id = "1"
    
    # Create profile if it doesn't exist
    if profile_id not in available_profiles:
        create_choice = input(f"Profile {profile_id} doesn't exist. Create it? (y/n): ").strip().lower()
        if create_choice == 'y':
            app.create_profile(profile_id)
        else:
            return
    
    # Show current progress
    progress = app.get_user_progress(profile_id)
    print(f"\n📊 Current Progress for Profile {profile_id}:")
    print(f"Total Responses: {progress['total_responses']}")
    print(f"Weakest Category: {progress['weakest_category']}")
    print(f"UUID: {progress['uuid']}")
    
    # Run session
    print(f"\n🎯 Starting training session with GPT-4o-mini...")
    try:
        session_result = app.run_daily_session(profile_id)
        print(f"\n✅ Session completed!")
        print(f"Category: {session_result['category']}")
        
        # Show top scoring categories
        print(f"\n📊 Top Category Scores:")
        sorted_scores = sorted(session_result['category_scores'].items(), 
                             key=lambda x: x[1], reverse=True)
        for category, score in sorted_scores[:3]:
            if score >= 3:
                print(f"  {category}: {score}/10")
                
    except Exception as e:
        print(f"Error: {e}")
        print("Make sure your OpenAI API key is set in config.env")

def check_progress():
    """Check user progress only."""
    print("\n📊 Progress Check (GPT Version)")
    print("="*40)
    
    app = BehavioralTrainingGPT()
    
    # Show available profiles
    available_profiles = app.list_available_profiles()
    if available_profiles:
        print(f"📁 Available Profiles: {', '.join(available_profiles)}")
    else:
        print("📁 No existing profiles found.")
        return
    
    profile_id = input("Enter profile ID (or press Enter for profile 1): ").strip()
    if not profile_id:
        profile_id = "1"
    
    if profile_id not in available_profiles:
        print(f"Profile {profile_id} not found.")
        return
    
    progress = app.get_user_progress(profile_id)
    
    print(f"\n📈 Progress Summary for Profile {profile_id}:")
    print(f"UUID: {progress['uuid']}")
    print(f"Total Responses: {progress['total_responses']}")
    print(f"Strongest Category: {progress['strongest_category']}")
    print(f"Weakest Category: {progress['weakest_category']}")
    
    print(f"\n📊 Composite Scores (Total Score / 10):")
    composite_scores = progress['composite_scores']
    sorted_scores = sorted(composite_scores.items(), key=lambda x: x[1], reverse=True)
    
    for i, (score_key, score) in enumerate(sorted_scores, 1):
        if score > 0:  # Only show categories with responses
            category_name = score_key.replace("Composite_", "").replace("_Score", "")
            bar_length = 15
            filled_length = int((score / 5.0) * bar_length)  # Normalize to 5.0 max
            bar = "█" * filled_length + "░" * (bar_length - filled_length)
            print(f"{i:2d}. {category_name:<18} {score:4.2f}/5.0 {bar}")

def show_help():
    """Show help information."""
    print("\n📖 BEHAVIORAL TRAINING APP HELP (GPT-4o-mini)")
    print("="*60)
    print("This app uses GPT-4o-mini to generate prompts and score responses.")
    print()
    print("Profile System:")
    print("• Simple numeric profiles (1, 2, 3, etc.)")
    print("• Create new profiles as needed")
    print("• Switch between profiles easily")
    print("• Delete profiles when no longer needed")
    print()
    print("Setup Required:")
    print("1. Set your OpenAI API key in config.env")
    print("2. Install required packages: pip install openai python-dotenv")
    print()
    print("Key Features:")
    print("• AI-powered prompt generation for each category")
    print("• Multi-category scoring (1-10 scale for each category)")
    print("• Composite scores calculated as total/10")
    print("• Only scores >= 3 are recorded for each category")
    print()
    print("Database Structure:")
    print("• UUID: Unique identifier for each profile")
    print("• Composite_[Category]_Score: Total score / 10 for each category")
    print("• [Category]_responses: Array of response data for each category")
    print()
    print("Scoring System:")
    print("• 10: Perfectly relevant to the category")
    print("• 7-9: Highly relevant")
    print("• 5-6: Moderately relevant")
    print("• 3-4: Slightly relevant")
    print("• 1-2: Not relevant (not recorded)")
    print()
    print("Behavioral Categories:")
    categories = ["Teamwork", "Leadership", "Conflict", "Problem Solving", "Initiative",
                  "Adaptability", "Failure", "Communication", "Time Management", "Integrity"]
    for i, category in enumerate(categories, 1):
        print(f"  {i:2d}. {category}")
    print()
    print("Data Storage:")
    print("• All data is stored locally in JSON files")
    print("• User profiles: behavioral_data/users/")
    print("• Daily prompts: behavioral_data/prompts/")
    print("• Responses: behavioral_data/responses/")

def check_api_key():
    """Check if OpenAI API key is configured."""
    api_key = os.getenv('OPENAI_API_KEY')
    if not api_key or api_key == 'your_openai_api_key_here':
        print("⚠️  WARNING: OpenAI API key not configured!")
        print("Please set your API key in config.env")
        print("Example: OPENAI_API_KEY=sk-your-actual-api-key-here")
        return False
    return True

def main():
    """Main launcher function."""
    print("🧠 Behavioral Interview Training App (GPT-4o-mini)")
    print("AI-powered behavioral interview practice")
    
    # Check API key
    if not check_api_key():
        print("\nThe app will use fallback prompts and scoring without GPT.")
        print("For full AI functionality, please configure your API key.")
    
    while True:
        show_launcher_menu()
        choice = input("Select an option (0-4): ").strip()
        
        if choice == '0':
            print("Goodbye! Keep practicing those behavioral questions!")
            break
        elif choice == '1':
            print("\nLaunching full CLI interface...")
            cli = BehavioralCLI()
            cli.run()
        elif choice == '2':
            quick_training()
        elif choice == '3':
            check_progress()
        elif choice == '4':
            show_help()
        else:
            print("Invalid option. Please try again.")
        
        if choice != '1':  # Don't ask for continue if launching CLI
            input("\nPress Enter to continue...")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\nGoodbye!")
    except Exception as e:
        print(f"\nAn error occurred: {e}")
        print("Please check that all required files are present and API key is configured.") 