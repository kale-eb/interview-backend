# Interview Video Backend

A modular backend system for interview analysis and related services.

## Project Structure

```
interviewVideoBackend/
├── interview_video_analysis/     # Interview behavioral analysis system
│   ├── live_ui.py               # Local analysis UI
│   ├── live_ui_ecs.py           # API-based UI
│   ├── main.py                  # Analysis API service
│   ├── rate_interview.py        # Rating logic
│   ├── docker-compose.yml       # Docker services
│   └── README.md               # Detailed documentation
├── backend/                     # Backend utilities and shared code
├── ecs_deployment.py           # AWS ECS deployment scripts (legacy)
├── backend_architecture.py     # Architecture documentation
└── README.md                   # This file
```

## Current Modules

### 1. Interview Video Analysis
**Location**: `interview_video_analysis/`

A comprehensive behavioral analysis system for interview recordings:
- Real-time video analysis using MediaPipe
- Behavioral scoring (eye contact, posture, face touching)
- Session logging and rating
- Docker deployment ready
- Dual UI options (local and API-based)

**Quick Start**:
```bash
cd interview_video_analysis
python live_ui_ecs.py  # Connect to remote APIs
# or
python live_ui.py      # Run everything locally
```

## Adding New Modules

You can add new modules to this project by creating new directories. Each module should be self-contained with its own:

- `README.md` - Documentation
- `requirements.txt` - Dependencies (if different from main project)
- `docker-compose.yml` - Docker services (if needed)
- Configuration files

### Example Module Structure
```
new_module/
├── README.md
├── requirements.txt
├── main.py
├── docker-compose.yml
└── config/
```

## Development Guidelines

1. **Modular Design**: Keep modules self-contained
2. **Documentation**: Each module should have its own README
3. **Dependencies**: Use separate requirements.txt files if needed
4. **Configuration**: Keep config files in module directories
5. **Docker**: Use docker-compose.yml for containerized services

## Shared Resources

- `backend/` - Shared utilities and code
- `.gitignore` - Project-wide ignore rules
- `backend_architecture.py` - Overall architecture documentation

## Deployment

Each module can be deployed independently or together:

```bash
# Deploy specific module
cd interview_video_analysis
docker compose up -d --build

# Or deploy from root (if you add a root docker-compose.yml)
docker compose up -d --build
```

## Contributing

When adding new modules:
1. Create a new directory for your module
2. Add a comprehensive README.md
3. Include necessary configuration files
4. Update this main README.md to document the new module
5. Ensure the module is self-contained and doesn't break existing functionality 