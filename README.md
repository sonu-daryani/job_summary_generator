# 🚀 Job Summary Generator

A modern, AI-powered web application that transforms bullet points into professional job summaries using Llama 2 or Hugging Face's free API. Built with Python, FastAPI, and optimized for performance.

## ✨ Features

- **AI-Powered Generation**: Uses Llama 2 (local) or Hugging Face's free API to generate professional summaries
- **Modern UI**: Beautiful, responsive interface with smooth animations
- **Real-time Generation**: Fast, asynchronous processing
- **Copy & Regenerate**: Easy-to-use buttons for copying and regenerating summaries
- **Optimized Performance**: Code splitting, async operations, and efficient API calls
- **Error Handling**: Robust error handling with fallback mechanisms
- **Mobile Responsive**: Works perfectly on all device sizes

## 🛠️ Tech Stack

- **Backend**: Python 3.8+, FastAPI, Uvicorn
- **AI Service**: Llama 2 (local) or Hugging Face Inference API (Free tier)
- **Frontend**: HTML5, CSS3, Vanilla JavaScript
- **Configuration**: Python-dotenv for environment management
- **Validation**: Pydantic for request/response validation

<img width="1891" height="1049" alt="image" src="https://github.com/user-attachments/assets/6e7134a6-7933-42d2-b3b9-725820e6305d" />


## 🚀 Quick Start

### Prerequisites

- Python 3.8 or higher
- For Llama 2 (local): 8GB+ RAM recommended, GPU optional but recommended
- For Hugging Face API: A free Hugging Face account and API key

### Installation

1. **Clone the repository**
   ```bash
   git clone <repository-url>
   cd simple_content_generator
   ```

2. **Install dependencies with uv**
   ```bash
   uv sync
   ```
   
   Or if you prefer pip:
   ```bash
   pip install -r requirements.txt
   ```

4. **Set up environment variables**
   
   Create a `.env` file in the project root:
   ```env
   # AI Model Configuration
   # Set to true to use local Llama 2 model, false to use Hugging Face API
   USE_LOCAL_MODEL=true
   
   # Llama 2 Model Configuration (when USE_LOCAL_MODEL=true)
   LLAMA_MODEL_NAME=meta-llama/Llama-2-7b-chat-hf
   DEVICE=auto
   # MAX_MEMORY=8GB  # Optional: Set max GPU memory
   
   # Hugging Face API Configuration (fallback when USE_LOCAL_MODEL=false)
   HUGGINGFACE_API_KEY=your_free_huggingface_api_key_here
   HUGGINGFACE_API_URL=https://api-inference.huggingface.co/models/microsoft/DialoGPT-medium
   
   # Application Configuration
   APP_NAME=Job Summary Generator
   DEBUG=True
   HOST=0.0.0.0
   PORT=8000
   ```

5. **Configure your AI model**
   
   **Option A: Use Llama 2 (Recommended)**
   - Set `USE_LOCAL_MODEL=true` in your `.env` file
   - The app will automatically download and use Llama 2-7b-chat-hf
   - For better performance, use a GPU by setting `DEVICE=cuda`
   - For Apple Silicon Macs, set `DEVICE=mps`
   
   **Option B: Use Hugging Face API (Fallback)**
   - Set `USE_LOCAL_MODEL=false` in your `.env` file
   - Get your free Hugging Face API key:
     - Go to [Hugging Face](https://huggingface.co/)
     - Sign up for a free account
     - Go to Settings → Access Tokens
     - Create a new token
     - Copy the token and paste it in your `.env` file

6. **Test the integration (Optional)**
   ```bash
   python test_llama2.py
   ```

7. **Run the application**
   
   With uv:
   ```bash
   uv run python main.py
   ```
   
   Or with regular Python:
   ```bash
   python main.py
   ```

7. **Open your browser**
   Navigate to `http://localhost:8000`

## 📖 Usage

1. **Enter Bullet Points**: Type or paste your work experience bullet points in the text area
2. **Generate Summary**: Click the "Generate Summary" button
3. **Review Result**: The AI will generate a professional 2-3 sentence summary
4. **Copy or Regenerate**: Use the "Copy Summary" button to copy to clipboard or "Regenerate" for a new version

### Example Input:
```
• 5 years of Python development experience
• Led a team of 3 developers on a major project
• Implemented microservices architecture
• Experience with FastAPI and Django frameworks
```

### Example Output:
```
Experienced Python developer with 5 years of expertise in leading development teams and implementing scalable microservices architecture. Proven track record with FastAPI and Django frameworks, successfully managing projects and mentoring junior developers.
```

## 🏗️ Project Structure

```
simple_content_generator/
├── main.py                 # Main FastAPI application
├── config.py              # Configuration and environment variables
├── models.py              # Pydantic models for validation
├── ai_service.py          # Hugging Face API integration
├── api_routes.py          # FastAPI routes and endpoints
├── requirements.txt       # Python dependencies
├── templates/
│   └── index.html         # Frontend HTML template
└── README.md             # This file
```

## 🔧 API Endpoints

### POST `/api/v1/generate-summary`
Generate a job summary from bullet points.

**Request Body:**
```json
{
  "bullet_points": [
    "5 years of Python development experience",
    "Led a team of 3 developers"
  ]
}
```

**Response:**
```json
{
  "summary": "Experienced Python developer with 5 years of expertise...",
  "original_bullet_points": ["5 years of Python development experience", "Led a team of 3 developers"]
}
```

### GET `/health`
Health check endpoint.

### GET `/`
Main application page.

## ⚡ Optimization Features

### Code Splitting
- **Modular Architecture**: Separated concerns into different modules (AI service, API routes, models)
- **Async Operations**: Non-blocking API calls and database operations
- **Lazy Loading**: AI service is initialized only when needed

### Performance Optimizations
- **Connection Pooling**: Reuses HTTP connections for API calls
- **Caching**: Singleton pattern for AI service instance
- **Error Handling**: Graceful fallbacks when AI service fails
- **Request Timeouts**: Prevents hanging requests

### Frontend Optimizations
- **Vanilla JavaScript**: No external dependencies, faster loading
- **CSS Grid/Flexbox**: Efficient layout rendering
- **Responsive Design**: Mobile-first approach
- **Progressive Enhancement**: Works without JavaScript

## 🐛 Troubleshooting

### Common Issues

1. **"HUGGINGFACE_API_KEY is required" error**
   - Make sure you've created a `.env` file with your API key
   - Verify the API key is correct and active

2. **"Model is loading" message**
   - Hugging Face free tier models may take time to load
   - Wait a few seconds and try again

3. **Summary generation fails**
   - Check your internet connection
   - Verify your Hugging Face API key is valid
   - The app will show a fallback summary if AI generation fails

4. **Frontend not loading**
   - Make sure you're accessing `http://localhost:8000`
   - Check that the `templates/index.html` file exists

## 🔒 Security Notes

- Never commit your `.env` file to version control
- The `.env` file is already in `.gitignore`
- API keys are only used for server-side requests
- No sensitive data is stored or logged

## 📝 License

This project is open source and available under the MIT License.

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## 📞 Support

If you encounter any issues or have questions, please open an issue on GitHub.

---

**Happy summarizing! 🎉**
