# 🧠 Behavioral Interview Training App

A comprehensive AI-powered application for practicing behavioral interview questions and tracking progress across 10 core behavioral categories.

## 🚀 Quick Start

### Prerequisites
- Python 3.7+
- OpenAI API key

### Installation
1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

2. Configure your OpenAI API key:
   ```bash
   # Edit config.env
   OPENAI_API_KEY=sk-your-actual-api-key-here
   ```

3. Run the app:
   ```bash
   python run_behavioral_training_gpt.py
   ```

## 📁 Project Structure

```
behavioral_interview_training/
├── README.md                           # This file (quick start guide)
├── requirements.txt                    # Python dependencies
├── config.env                         # Configuration (API keys, etc.)
├── interview_trainer.py                # Core AI-powered interview trainer
├── interview_cli.py                    # Interactive CLI interface
├── run_trainer.py                      # Quick launcher script
└── README_DETAILED.md                  # Complete documentation
```

## 🎯 AI-Powered Interview Trainer

- AI-generated prompts using GPT-4o-mini
- Multi-category scoring across all 10 behavioral areas
- Intelligent relevance detection
- Simple numeric profile system (1, 2, 3, etc.)
- Profile management (create, switch, delete)

## 🎮 Usage Options

### Quick Start
```bash
python run_trainer.py
```

### Full CLI Interface
```bash
python interview_cli.py
```

## 📊 Behavioral Categories

The app tracks proficiency across these 10 core behavioral areas:
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

## 🗂️ Data Storage

All data is stored locally in JSON files:
- **User Profiles**: `behavioral_data/users/`
- **Daily Prompts**: `behavioral_data/prompts/`
- **User Responses**: `behavioral_data/responses/`

## 📚 Documentation

- **[Complete Documentation](README_DETAILED.md)** - Full guide for the AI-powered interview trainer

## 🔧 Configuration

Edit `config.env` to customize:
- OpenAI API key
- Learning rate for score calculations
- Minimum relevance score threshold
- Composite score divisor

## 🐛 Troubleshooting

### Common Issues
1. **API Key Not Set**: Ensure `OPENAI_API_KEY` is configured in `config.env`
2. **Import Errors**: Install dependencies with `pip install -r requirements.txt`
3. **File Permissions**: Ensure write access to the project directory

### Fallback System
The GPT version gracefully falls back to template prompts if:
- API key is not configured
- Network connectivity issues occur
- API rate limits are exceeded

## 🤝 Contributing

This is an MVP implementation. Future contributions could include:
- Enhanced AI integration
- Additional behavioral categories
- Improved scoring algorithms
- Web interface development
- Data visualization features

## 📄 License

This project is part of the interview training system. All data is stored locally and no external services are required beyond OpenAI API for enhanced functionality.

---

**Start practicing today and improve your behavioral interview skills!** 🎯 