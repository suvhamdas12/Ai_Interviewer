# 🚀 AI-Powered Java Spring Boot Interview System

A web-based platform that conducts AI-driven interviews for Java developers. Powered by FastAPI, CrewAI, and Groq LLM, it adapts questions based on user input and provides real-time feedback, coding challenges, and scoring.

---

## 📸 Screenshots
<img width="1432" alt="image" src="https://github.com/user-attachments/assets/2f2045fd-7355-47c6-8a29-2d6c9cc4a112" />


### 💻 Coding Question UI
<img width="1422" alt="image" src="https://github.com/user-attachments/assets/f71f1282-26b0-485c-97f4-99f3e15779c8" />

---

## 🧠 Features

- ✅ **Interactive AI-Powered Interviews**
- 🎯 Focused on: Core Java, Spring Boot, Microservices, Database, Testing
- 🔁 Adaptive Questioning based on candidate’s responses
- 📊 Real-time Feedback & Scoring
- 🧑‍💻 Coding Challenges with clear requirements
- 🔍 Sample Questions by topic & experience level
- 🌐 RESTful API (FastAPI + LLM)

---

## 📦 Tech Stack

| Tool/Library   | Purpose                            |
|----------------|------------------------------------|
| FastAPI        | Backend API framework              |
| CrewAI         | Multi-agent orchestration          |
| Groq LLM       | LLM for question/feedback generation |
| Python         | Backend logic                      |
| HTML/CSS       | Frontend UI                        |
| JavaScript     | API interaction from UI            |

---

## 🛠 Setup Instructions

### 1. Clone Repository

```bash
git clone https://github.com/your-username/java-interview-ai.git
cd java-interview-ai

2. Add .env File

Create a .env file:
GROQ_API_KEY=your_groq_api_key_here

3. Install Python Dependencies
pip install -r requirements.txt

4. Run the Application
uvicorn main:app --reload
