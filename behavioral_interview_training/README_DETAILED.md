# 🧠 AI-Powered Interview Trainer

An AI-powered local MVP for practicing behavioral interview questions using GPT-4o-mini for prompt generation and intelligent scoring across 10 behavioral categories.

## 🎯 Purpose

This app leverages GPT-4o-mini to provide intelligent behavioral interview training by:
- **AI-Generated Prompts**: Creates unique, contextual questions for each category
- **Multi-Category Scoring**: Evaluates responses across all 10 behavioral areas simultaneously
- **Smart Relevance Detection**: Scores how well each response relates to each category (1-10 scale)
- **Composite Score Tracking**: Calculates overall performance using total score / 10 formula
- **Intelligent Targeting**: Identifies weakest areas for focused practice

## 🤖 AI Integration

### GPT-4o-mini Features
- **Dynamic Prompt Generation**: Creates unique behavioral questions tailored to each category
- **Intelligent Scoring**: Evaluates response relevance across all categories simultaneously
- **Follow-up Questions**: Generates contextual follow-up questions for deeper reflection
- **Fallback System**: Graceful degradation to template prompts if API is unavailable

### API Requirements
- OpenAI API key required for full functionality
- Configure in `config.env`: `OPENAI_API_KEY=your_api_key_here`
- App works with fallback prompts if API key is not configured

## 📦 Features

### Core Functionality
- **AI-Powered Prompts**: GPT-4o-mini generates unique behavioral questions
- **Multi-Category Scoring**: Each response scored across all 10 categories (1-10 scale)
- **Composite Score Calculation**: Total score / 10 for each category
- **Relevance Filtering**: Only scores ≥ 3 are recorded for each category
- **UUID-Based Users**: Unique identifiers for each user profile
- **Progress Tracking**: Comprehensive analytics and visualization

### Behavioral Categories Tracked
1. **Teamwork** - Collaboration and team dynamics
2. **Leadership** - Leading teams and making decisions
3. **Conflict** - Resolving disagreements and difficult situations
4. **Problem Solving** - Analytical thinking and creative solutions
5. **Initiative** - Taking action and going above and beyond
6. **Adaptability** - Handling change and learning new skills
7. **Failure** - Learning from mistakes and setbacks
8. **Communication** - Clear expression and effective messaging
9. **Time Management** - Prioritization and meeting deadlines
10. **Integrity** - Ethical decision-making and honesty

## 🗂️ Database Structure

### User Profile Schema
```json
{
  "user_id": "demo_user",
  "uuid": "550e8400-e29b-41d4-a716-446655440000",
  "created_date": "2024-01-01T00:00:00",
  "composite_scores": {
    "Composite_Teamwork_Score": 2.4,
    "Composite_Leadership_Score": 1.8,
    "Composite_Conflict_Score": 0.0,
    // ... all categories
  },
  "responses_by_category": {
    "Teamwork_responses": [
      {
        "date": "2024-01-01",
        "score": 8,
        "response": "Response stored separately"
      }
    ],
    // ... all categories
  },
  "total_responses": 5
}
```

### Response Schema
```json
{
  "user_id": "demo_user",
  "date": "2024-01-01",
  "response": "User's detailed response...",
  "category_scores": {
    "Teamwork": 8,
    "Leadership": 3,
    "Conflict": 5,
    "Problem Solving": 7,
    // ... all categories scored
  },
  "timestamp": "2024-01-01T12:00:00",
  "is_followup": false
}
```

## 🚀 Getting Started

### Prerequisites
- Python 3.7+
- OpenAI API key
- Required packages: `openai`, `python-dotenv`

### Installation
1. Install dependencies:
   ```bash
   pip install openai python-dotenv
   ```

2. Configure API key:
   ```bash
   # Edit config.env
   OPENAI_API_KEY=sk-your-actual-api-key-here
   ```

3. Run the app:
   ```bash
   python run_trainer.py
   ```

### Usage Options

#### 1. Full CLI Interface
Complete interactive experience with all features:
```bash
python interview_cli.py
```

#### 2. Quick Training Session
Fast daily practice with GPT:
```bash
python run_trainer.py
# Select option 2
```

#### 3. Direct Module Usage
```python
from interview_trainer import BehavioralTrainingGPT

app = BehavioralTrainingGPT()
session_result = app.run_daily_session("your_user_id")
```

## 📊 Scoring System

### Multi-Category Scoring (1-10 Scale)
Each response is evaluated across all 10 behavioral categories:

- **10**: Perfectly relevant - story directly demonstrates this skill
- **7-9**: Highly relevant - strong connection to the category
- **5-6**: Moderately relevant - some connection to the category
- **3-4**: Slightly relevant - minimal connection
- **1-2**: Not relevant - no meaningful connection (not recorded)

### Composite Score Calculation
```
Composite_[Category]_Score = Total of all scores for that category / 10
```

Example:
- User has 3 responses with Teamwork scores: 8, 6, 9
- Total: 8 + 6 + 9 = 23
- Composite_Teamwork_Score = 23 / 10 = 2.3

### Relevance Filtering
- Only scores ≥ 3 are recorded for each category
- Prevents noise from irrelevant responses
- Ensures quality data for composite calculations

## 🔄 Daily Flow

1. **Load User Profile**: Retrieve current composite scores and history
2. **Identify Weakest Category**: Find area with lowest composite score
3. **Generate AI Prompt**: GPT-4o-mini creates targeted question + follow-up
4. **Capture Response**: Get user's detailed answer
5. **Multi-Category Scoring**: GPT evaluates relevance across all categories
6. **Update Profile**: Calculate new composite scores and store responses
7. **Save Data**: Store all information in structured JSON format

## 📈 Progress Tracking

### Metrics Tracked
- **Total Responses**: Number of completed sessions
- **Composite Scores**: Overall performance for each category (total/10)
- **Response Counts**: Number of responses per category
- **Strongest/Weakest**: Current best and worst performing categories
- **UUID**: Unique identifier for each user

### Visualization Features
- **Progress Bars**: Visual representation of composite scores
- **Category Breakdown**: Detailed view of performance across categories
- **Response History**: Browse past answers with multi-category scores
- **Recent Activity**: Latest responses with top scoring categories

## 🎨 User Interface

### CLI Features
- **Interactive Menus**: Easy navigation between features
- **AI Status Indicators**: Shows when GPT is being used
- **Multi-Category Display**: Shows scores across all categories
- **Progress Visualization**: ASCII progress bars and charts
- **Response History**: Browse past answers with category breakdowns
- **Data Export**: Save all data to JSON for backup/analysis

### Visual Elements
- 🤖 AI indicators for GPT-powered features
- 📊 Multi-category score displays
- 📈 Progress bars showing composite scores
- 📝 Response history with category breakdowns
- 🎯 Weakest category targeting

## 🔧 Customization

### Adding New Categories
Edit `BEHAVIORAL_CATEGORIES` in `behavioral_training_gpt.py`:
```python
BEHAVIORAL_CATEGORIES = [
    "Teamwork", "Leadership", "Conflict", "Problem Solving", "Initiative",
    "Adaptability", "Failure", "Communication", "Time Management", "Integrity",
    "Your New Category"  # Add here
]
```

### Customizing GPT Prompts
Modify the system prompts in `generate_prompt_with_gpt()` and `score_response_with_gpt()` methods.

### Adjusting Scoring Thresholds
Edit configuration in `config.env`:
```env
MIN_RELEVANCE_SCORE=3
MAX_RELEVANCE_SCORE=10
COMPOSITE_SCORE_DIVISOR=10
```

## 📋 Best Practices

### For Users
- **Be Specific**: Include concrete examples and measurable outcomes
- **Use STAR Method**: Situation, Task, Action, Result structure
- **Show Growth**: Focus on what you learned and how you improved
- **Practice Regularly**: Daily sessions build consistency and skill
- **Review History**: Learn from past responses and scoring patterns

### For Responses
- **Length**: Aim for 200-500 words for comprehensive coverage
- **Structure**: Clear beginning, middle, and end with specific examples
- **Outcomes**: Include measurable results and metrics when possible
- **Reflection**: Show what you learned and how you grew
- **Authenticity**: Be honest about challenges and how you overcame them

## 🔮 Future Enhancements

### Planned Features
- **Advanced LLM Integration**: More sophisticated response analysis
- **Diversity Scoring**: Track variety of examples used across categories
- **Growth Metrics**: Measure improvement over time with trend analysis
- **Web Interface**: Browser-based UI with real-time scoring
- **Data Analytics**: Advanced progress visualization and insights
- **Custom Prompt Templates**: User-defined question preferences

### Technical Improvements
- **Database Integration**: Move from JSON to proper database (PostgreSQL/MySQL)
- **API Development**: RESTful endpoints for web/mobile applications
- **User Authentication**: Secure user management and profiles
- **Cloud Storage**: Remote data backup and synchronization
- **Real-time Scoring**: Instant feedback during response composition

## 🐛 Troubleshooting

### Common Issues
1. **API Key Errors**: Ensure `OPENAI_API_KEY` is set in `config.env`
2. **Import Errors**: Install required packages: `pip install openai python-dotenv`
3. **File Permission Errors**: Ensure write access to `behavioral_data/` directory
4. **JSON Parsing Errors**: Check for malformed responses from GPT API
5. **Network Issues**: App falls back to template prompts if API is unavailable

### Debug Mode
Enable verbose logging by modifying the main functions to include debug prints.

### Fallback System
The app gracefully degrades to template prompts and simple scoring if:
- OpenAI API key is not configured
- Network connectivity issues occur
- API rate limits are exceeded
- GPT responses are malformed

## 📄 License

This project is part of the interview video analysis system. All data is stored locally and no external services are required beyond OpenAI API for enhanced functionality.

## 🤝 Contributing

This is an MVP implementation with AI integration. Future contributions could include:
- Enhanced GPT prompt engineering
- Additional behavioral categories
- Improved scoring algorithms
- Better user interface design
- Data visualization features
- Integration with other AI services

---

**Remember**: The goal is consistent practice and improvement. The AI helps identify your strengths and weaknesses, but your dedication to practice is what drives real improvement in behavioral interview skills! 