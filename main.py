import os
from typing import List, Dict, Any
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from crewai import Agent, Task, Crew, LLM
from crewai.tools import BaseTool
import uvicorn
from datetime import datetime
from dotenv import load_dotenv
import json
import random

load_dotenv()

# Configure Groq LLM using CrewAI's LLM class
def get_groq_llm():
    return LLM(
        model="groq/llama-3.1-70b-versatile",
        api_key=os.getenv("GROQ_API_KEY"),
        temperature=0.7
    )

# Pydantic models for API
class InterviewRequest(BaseModel):
    candidate_name: str
    experience_level: str  # "junior", "mid", "senior"
    focus_areas: List[str]  # ["core_java", "spring_boot", "microservices", "database"]

class QuestionResponse(BaseModel):
    question: str
    candidate_answer: str

class InterviewResponse(BaseModel):
    question: str
    question_type: str
    difficulty: str
    topic: str

class FeedbackResponse(BaseModel):
    score: int  # 1-10
    feedback: str
    suggestions: List[str]

# Knowledge base tool for interview questions
class JavaSpringBootKnowledgeBase(BaseTool):
    name: str = "java_springboot_knowledge"
    description: str = "Access Java and Spring Boot interview questions and concepts"
    
    def _run(self, topic: str, level: str) -> str:
        knowledge_base = {
            "core_java": {
                "junior": [
                    "What is the difference between JDK, JRE, and JVM?",
                    "Explain the concept of Object-Oriented Programming in Java.",
                    "What are the access modifiers in Java?",
                    "What is the difference between == and equals() method?",
                    "Explain method overloading vs method overriding."
                ],
                "mid": [
                    "Explain the Java Memory Model and garbage collection.",
                    "What are design patterns? Implement Singleton pattern.",
                    "Discuss multithreading in Java and synchronization.",
                    "Explain exception handling hierarchy in Java.",
                    "What are Java 8 features like Streams and Lambda expressions?"
                ],
                "senior": [
                    "Explain JVM internals and memory management strategies.",
                    "Discuss concurrent programming and java.util.concurrent package.",
                    "How would you optimize Java application performance?",
                    "Explain reflection and its use cases in frameworks.",
                    "Discuss design patterns used in enterprise applications."
                ]
            },
            "spring_boot": {
                "junior": [
                    "What is Spring Boot and its advantages?",
                    "Explain dependency injection in Spring.",
                    "What are Spring Boot annotations like @Component, @Service?",
                    "How do you create a REST API in Spring Boot?",
                    "What is application.properties file used for?"
                ],
                "mid": [
                    "Explain Spring Boot auto-configuration mechanism.",
                    "What are different ways to configure Spring Boot application?",
                    "Discuss Spring Data JPA and its benefits.",
                    "How do you handle exceptions in Spring Boot?",
                    "Explain Spring Boot Actuator and its endpoints."
                ],
                "senior": [
                    "How would you implement microservices with Spring Boot?",
                    "Discuss Spring Cloud components for distributed systems.",
                    "Explain transaction management in Spring Boot.",
                    "How do you implement security in Spring Boot applications?",
                    "Discuss performance optimization strategies for Spring Boot apps."
                ]
            },
            "microservices": {
                "mid": [
                    "What are microservices and their advantages?",
                    "How do you handle inter-service communication?",
                    "Explain service discovery patterns.",
                    "What is circuit breaker pattern?",
                    "How do you manage configuration in microservices?"
                ],
                "senior": [
                    "Design a microservices architecture for an e-commerce platform.",
                    "How do you handle distributed transactions?",
                    "Explain event-driven architecture in microservices.",
                    "Discuss monitoring and observability in microservices.",
                    "How do you implement API gateway pattern?"
                ]
            },
            "database": {
                "junior": [
                    "What is the difference between SQL and NoSQL databases?",
                    "Explain JDBC and how to connect to a database in Java.",
                    "What are primary keys and foreign keys?",
                    "How do you perform CRUD operations with Spring Data JPA?",
                    "What is the difference between @Entity and @Table annotations?"
                ],
                "mid": [
                    "Explain database connection pooling and its benefits.",
                    "What are JPA relationships and how do you map them?",
                    "Discuss transaction management in Spring Boot.",
                    "How do you optimize database queries in JPA?",
                    "Explain the difference between lazy and eager loading."
                ],
                "senior": [
                    "How would you handle database migrations in production?",
                    "Discuss database sharding and partitioning strategies.",
                    "How do you implement database caching strategies?",
                    "Explain distributed transaction patterns like Saga.",
                    "How do you monitor and optimize database performance?"
                ]
            },
            "testing": {
                "junior": [
                    "What is unit testing and why is it important?",
                    "How do you write unit tests with JUnit?",
                    "What is the difference between @Test and @TestMethodOrder?",
                    "How do you test REST APIs in Spring Boot?",
                    "What are assertions in testing?"
                ],
                "mid": [
                    "Explain mocking and how to use Mockito.",
                    "What is integration testing vs unit testing?",
                    "How do you test Spring Boot applications with @SpringBootTest?",
                    "Discuss test-driven development (TDD) approach.",
                    "How do you test database layers with @DataJpaTest?"
                ],
                "senior": [
                    "How do you implement comprehensive testing strategies?",
                    "Discuss contract testing and consumer-driven contracts.",
                    "How do you test microservices interactions?",
                    "Explain performance testing and load testing approaches.",
                    "How do you implement continuous testing in CI/CD pipelines?"
                ]
            }
        }
        
        if topic in knowledge_base and level in knowledge_base[topic]:
            questions = knowledge_base[topic][level]
            return json.dumps(questions)
        return json.dumps(["No questions found for the specified topic and level"])

# Create agents
def create_interview_agents():
    llm = get_groq_llm()
    knowledge_tool = JavaSpringBootKnowledgeBase()
    
    # Technical Interviewer Agent
    technical_interviewer = Agent(
        role='Senior Java Technical Interviewer',
        goal='Conduct comprehensive technical interviews for Java and Spring Boot positions',
        backstory='''You are an experienced technical interviewer with 10+ years in Java development.
        You specialize in evaluating candidates' knowledge of Java, Spring Boot, microservices, and related technologies.
        You ask progressive questions based on candidate responses and adjust difficulty accordingly.
        You focus purely on technical competency and coding skills.''',
        verbose=True,
        allow_delegation=False,
        llm=llm,
        tools=[knowledge_tool]
    )
    
    # Evaluator Agent
    evaluator = Agent(
        role='Technical Interview Evaluator',
        goal='Provide comprehensive feedback and scoring for interview responses',
        backstory='''You are an expert technical evaluator with deep knowledge of Java and Spring Boot.
        You analyze candidate responses for technical accuracy, depth of understanding, and best practices.
        You provide constructive feedback and actionable suggestions for improvement.
        You focus on technical competency, code quality, and implementation knowledge.''',
        verbose=True,
        allow_delegation=False,
        llm=llm
    )
    
    return technical_interviewer, evaluator

# FastAPI application
app = FastAPI(title="Java Spring Boot Interview System", version="1.0.0")

# Global variables to maintain interview state
current_interview_session = {}
agents = create_interview_agents()
technical_interviewer, evaluator = agents

@app.post("/start-interview", response_model=InterviewResponse)
async def start_interview(request: InterviewRequest):
    """Start a new interview session"""
    try:
        # Get a question from knowledge base first
        knowledge_tool = JavaSpringBootKnowledgeBase()
        topic = request.focus_areas[0] if request.focus_areas else "core_java"
        questions_json = knowledge_tool._run(topic, request.experience_level)
        questions = json.loads(questions_json)
        
        if questions and questions[0] != "No questions found for the specified topic and level":
            selected_question = random.choice(questions)
        else:
            selected_question = "What is your experience with Java development?"
        
        # Create task for personalizing the question
        task = Task(
            description=f'''You are interviewing {request.candidate_name}, a {request.experience_level} level Java developer.
            Focus areas: {', '.join(request.focus_areas)}
            
            Start with this technical question: "{selected_question}"
            
            Present it in a friendly, professional manner. You can add context or rephrase it slightly to make it more conversational, but keep the core technical content.''',
            agent=technical_interviewer,
            expected_output="A professionally presented technical question"
        )
        
        crew = Crew(
            agents=[technical_interviewer],
            tasks=[task],
            verbose=False
        )
        
        result = crew.kickoff()
        
        # Parse and structure the response
        question_data = {
            "question": str(result).strip(),
            "question_type": "technical",
            "difficulty": request.experience_level,
            "topic": topic
        }
        
        # Store session data
        session_id = f"{request.candidate_name}_{datetime.now().isoformat()}"
        current_interview_session[session_id] = {
            "candidate": request.candidate_name,
            "level": request.experience_level,
            "focus_areas": request.focus_areas,
            "questions_asked": [question_data],
            "responses": []
        }
        
        return InterviewResponse(**question_data)
        
    except Exception as e:
        # Fallback to knowledge base question
        knowledge_tool = JavaSpringBootKnowledgeBase()
        topic = request.focus_areas[0] if request.focus_areas else "core_java"
        questions_json = knowledge_tool._run(topic, request.experience_level)
        questions = json.loads(questions_json)
        
        fallback_question = questions[0] if questions else "Tell me about your Java experience."
        
        return InterviewResponse(
            question=fallback_question,
            question_type="technical",
            difficulty=request.experience_level,
            topic=topic
        )

@app.post("/next-question", response_model=InterviewResponse)
async def get_next_question(response: QuestionResponse):
    """Get the next question based on previous response"""
    try:
        # Create task for generating next question based on previous response
        task = Task(
            description=f'''Based on the candidate's previous answer: "{response.candidate_answer}"
            to the question: "{response.question}"
            
            Generate the next appropriate technical question. Consider:
            1. The quality and depth of their previous answer
            2. Whether to increase or decrease difficulty
            3. Whether to explore the same topic deeper or move to a new area
            
            Provide a follow-up question that builds on their response. Keep it technical and relevant to Java/Spring Boot development.''',
            agent=technical_interviewer,
            expected_output="Next appropriate technical question"
        )
        
        crew = Crew(
            agents=[technical_interviewer],
            tasks=[task],
            verbose=False
        )
        
        result = crew.kickoff()
        
        question_data = {
            "question": str(result).strip(),
            "question_type": "technical",
            "difficulty": "adaptive",
            "topic": "follow_up"
        }
        
        return InterviewResponse(**question_data)
        
    except Exception as e:
        # Fallback question
        return InterviewResponse(
            question="Can you explain a challenging technical problem you've solved recently?",
            question_type="technical",
            difficulty="adaptive",
            topic="problem_solving"
        )

@app.post("/coding-question", response_model=InterviewResponse)
async def get_coding_question():
    """Get a coding challenge question"""
    try:
        task = Task(
            description='''Generate a coding challenge question suitable for a Java developer.
            The question should require writing actual code and test their programming skills.
            Examples: implement a data structure, solve an algorithm problem, write a Spring Boot service method.
            Provide clear requirements and expected input/output.
            
            Make it practical and relevant to real-world Java development.''',
            agent=technical_interviewer,
            expected_output="A coding challenge question with clear requirements"
        )
        
        crew = Crew(
            agents=[technical_interviewer],
            tasks=[task],
            verbose=False
        )
        
        result = crew.kickoff()
        
        return InterviewResponse(
            question=str(result).strip(),
            question_type="coding",
            difficulty="practical",
            topic="programming"
        )
        
    except Exception as e:
        # Fallback coding question
        return InterviewResponse(
            question="Write a Java method to find the second largest element in an array of integers. Handle edge cases appropriately.",
            question_type="coding",
            difficulty="practical",
            topic="programming"
        )

@app.post("/evaluate-response", response_model=FeedbackResponse)
async def evaluate_response(response: QuestionResponse):
    """Evaluate candidate's response and provide feedback"""
    try:
        task = Task(
            description=f'''Evaluate this candidate's response to a technical question:
            
            Question: {response.question}
            Answer: {response.candidate_answer}
            
            Provide:
            1. A score from 1-10 based on technical accuracy and completeness
            2. Detailed feedback on their answer
            3. Specific suggestions for improvement
            4. What they did well (if anything)
            5. What they missed or could improve
            
            Be constructive and helpful in your evaluation. Focus on technical aspects.''',
            agent=evaluator,
            expected_output="Comprehensive evaluation with score, feedback, and suggestions"
        )
        
        crew = Crew(
            agents=[evaluator],
            tasks=[task],
            verbose=False
        )
        
        result = crew.kickoff()
        
        # Extract score from result (simple parsing)
        result_str = str(result)
        score = 7  # Default score
        
        # Try to extract score from response
        for i in range(1, 11):
            if f"score: {i}" in result_str.lower() or f"rating: {i}" in result_str.lower():
                score = i
                break
            if f"{i}/10" in result_str:
                score = i
                break
        
        suggestions = [
            "Practice with more examples",
            "Review the fundamentals", 
            "Focus on implementation details"
        ]
        
        return FeedbackResponse(
            score=score,
            feedback=result_str,
            suggestions=suggestions
        )
        
    except Exception as e:
        return FeedbackResponse(
            score=5,
            feedback="Unable to evaluate response at this time. Please try again.",
            suggestions=["Review the topic", "Practice more examples"]
        )

@app.get("/sample-questions/{topic}/{level}")
async def get_sample_questions(topic: str, level: str):
    """Get sample questions for a specific topic and level"""
    knowledge_tool = JavaSpringBootKnowledgeBase()
    questions_json = knowledge_tool._run(topic, level)
    questions = json.loads(questions_json)
    return {"topic": topic, "level": level, "questions": questions}

@app.get("/health")
async def health_check():
    return {"status": "healthy", "timestamp": datetime.now().isoformat()}

if __name__ == "__main__":
    # Make sure to set your GROQ_API_KEY environment variable
    if not os.getenv("GROQ_API_KEY"):
        print("Warning: GROQ_API_KEY environment variable not set")
        print("Please set GROQ_API_KEY in your .env file")
    else:
        print("GROQ_API_KEY found, starting server...")
    
    uvicorn.run(app, host="0.0.0.0", port=8000)

from fastapi.middleware.cors import CORSMiddleware

# Add this after `app = FastAPI(...)`
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # or specify: ["http://localhost:5500"] if using VSCode Live Server
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)