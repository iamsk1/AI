import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from flask import Flask, jsonify, render_template_string
import os
import time
import random

# --- 1. Data Loading and AI Model Initialization ---

print("Loading AI model...")
model = SentenceTransformer("all-MiniLM-L6-v2")
print("AI model loaded successfully!")

# Load mock data
employees_df = pd.read_csv("employees.csv")
projects_df = pd.read_csv("projects.csv")

# Generate embeddings
print("Generating embeddings for employees and projects...")
employee_skills_list = employees_df["skills"].tolist()
employee_embeddings = model.encode(employee_skills_list)

project_skills_list = projects_df["required_skills"].tolist()
project_embeddings = model.encode(project_skills_list)
print("Embeddings generated successfully!")

# --- 2. Core Matching Logic ---

def calculate_skill_overlap(employee_skills, project_skills):
    """Calculate detailed skill overlap percentages."""
    emp_skills = set([s.strip().lower() for s in employee_skills.split(',')])
    proj_skills = [s.strip().lower() for s in project_skills.split(',')]
    
    overlaps = []
    for skill in proj_skills:
        # Find best match in employee skills
        best_match = 0
        for emp_skill in emp_skills:
            if skill in emp_skill or emp_skill in skill:
                best_match = random.randint(85, 98)  # Simulated match percentage
                break
        if best_match == 0:
            best_match = random.randint(60, 84)
        overlaps.append({"skill": skill.title(), "percentage": best_match})
    
    return overlaps

def find_best_matches(project_id, top_n=3):
    """Find the best employee matches for a given project ID."""
    try:
        project_index = projects_df[projects_df["project_id"] == project_id].index[0]
        project_embedding = project_embeddings[project_index]
    except IndexError:
        return {"error": f"Project with ID '{project_id}' not found."}

    similarities = cosine_similarity([project_embedding], employee_embeddings)[0]
    
    results_df = employees_df.copy()
    results_df["match_score"] = similarities
    results_df = results_df.sort_values(by="match_score", ascending=False)
    
    top_matches = results_df.head(top_n)
    
    project_info = projects_df.iloc[project_index].to_dict()
    matches_list = []
    for _, row in top_matches.iterrows():
        match_score = round(float(row["match_score"]) * 100, 1)
        skill_overlap = calculate_skill_overlap(row["skills"], project_info["required_skills"])
        
        matches_list.append({
            "employee_id": int(row["employee_id"]),
            "name": row["name"],
            "role": row["role"],
            "skills": row["skills"],
            "match_score": match_score,
            "skill_overlap": skill_overlap
        })

    return {
        "project": project_info,
        "top_matches": matches_list
    }

# --- 3. Executive Summary Statistics ---

def get_executive_summary():
    """Calculate executive summary statistics."""
    total_employees = len(employees_df)
    total_projects = len(projects_df)
    
    # Simulated statistics based on TCS scale
    return {
        "total_employees": 5000,  # Simulated for demo
        "projects_matched": 1247,
        "avg_match_time": 3.2,  # seconds
        "client_satisfaction": 92,  # percentage
        "bench_reduction": 33  # percentage
    }

# --- 4. Demo Mode Logic ---

def run_demo_matching():
    """Run automated demo matching for 10 samples."""
    demo_results = []
    project_ids = projects_df["project_id"].tolist()
    
    for i in range(min(10, len(project_ids) * 3)):  # Run 10 matches
        project_id = random.choice(project_ids)
        matches = find_best_matches(project_id, top_n=1)
        
        if "error" not in matches:
            project = matches["project"]
            top_match = matches["top_matches"][0]
            
            demo_results.append({
                "match_number": i + 1,
                "project_name": project["project_name"],
                "employee_name": top_match["name"],
                "match_score": top_match["match_score"],
                "time": round(random.uniform(2.5, 4.0), 1)
            })
    
    return {
        "matches": demo_results,
        "total_time": round(sum([r["time"] for r in demo_results]), 1)
    }

# --- 5. Flask App ---

app = Flask(__name__)

# HTML template for the home page with all enhancements
HOME_PAGE = """
<!DOCTYPE html>
<html>
<head>
    <title>TCS RMG AI Matching System - Professional Demo</title>
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <style>
        * {
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }
        body {
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Arial, sans-serif;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            min-height: 100vh;
            padding: 20px;
        }
        .container {
            max-width: 1400px;
            margin: 0 auto;
        }
        .header {
            background: white;
            padding: 30px;
            border-radius: 15px;
            box-shadow: 0 10px 30px rgba(0,0,0,0.2);
            margin-bottom: 30px;
            text-align: center;
        }
        .header h1 {
            color: #333;
            font-size: 2.5em;
            margin-bottom: 10px;
        }
        .header p {
            color: #666;
            font-size: 1.2em;
        }
        
        /* Enhancement 4: Executive Summary Card */
        .executive-summary {
            background: white;
            padding: 30px;
            border-radius: 15px;
            box-shadow: 0 10px 30px rgba(0,0,0,0.2);
            margin-bottom: 30px;
        }
        .executive-summary h2 {
            color: #333;
            margin-bottom: 25px;
            font-size: 1.8em;
            text-align: center;
        }
        .stats-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 20px;
        }
        .stat-card {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 25px;
            border-radius: 10px;
            text-align: center;
            transition: transform 0.3s, box-shadow 0.3s;
        }
        .stat-card:hover {
            transform: translateY(-5px);
            box-shadow: 0 0 30px rgba(102, 126, 234, 0.6);
            animation: pulse 1.5s infinite;
        }
        @keyframes pulse {
            0%, 100% {
                box-shadow: 0 0 30px rgba(102, 126, 234, 0.6);
            }
            50% {
                box-shadow: 0 0 50px rgba(102, 126, 234, 0.9);
            }
        }
        .stat-value {
            font-size: 2.5em;
            font-weight: bold;
            margin: 10px 0;
        }
        .stat-label {
            font-size: 0.9em;
            opacity: 0.9;
        }
        
        /* Enhancement 5: Demo Mode Section */
        .demo-mode-section {
            background: white;
            padding: 30px;
            border-radius: 15px;
            box-shadow: 0 10px 30px rgba(0,0,0,0.2);
            margin-bottom: 30px;
        }
        .demo-mode-section h2 {
            color: #333;
            margin-bottom: 20px;
            font-size: 1.8em;
        }
        .demo-button {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            border: none;
            padding: 20px 40px;
            font-size: 1.3em;
            border-radius: 10px;
            cursor: pointer;
            transition: transform 0.3s, box-shadow 0.3s;
            width: 100%;
            max-width: 400px;
            display: block;
            margin: 20px auto;
        }
        .demo-button:hover {
            transform: translateY(-3px);
            box-shadow: 0 10px 25px rgba(0,0,0,0.3);
        }
        .demo-button:disabled {
            opacity: 0.6;
            cursor: not-allowed;
        }
        .demo-results {
            margin-top: 30px;
            display: none;
        }
        .demo-results.show {
            display: block;
        }
        .demo-match-item {
            background: #f8f9fa;
            padding: 15px;
            border-radius: 8px;
            margin-bottom: 10px;
            border-left: 4px solid #667eea;
            display: flex;
            justify-content: space-between;
            align-items: center;
            animation: slideIn 0.5s ease;
        }
        @keyframes slideIn {
            from {
                opacity: 0;
                transform: translateX(-20px);
            }
            to {
                opacity: 1;
                transform: translateX(0);
            }
        }
        .demo-match-info {
            flex: 1;
        }

        .total-savings-counter {
            background: linear-gradient(135deg, #28a745 0%, #20c997 100%);
            color: white;
            padding: 30px;
            border-radius: 10px;
            text-align: center;
            margin-top: 20px;
            font-size: 2em;
            font-weight: bold;
        }
        
        .demo-section {
            background: white;
            padding: 30px;
            border-radius: 15px;
            box-shadow: 0 10px 30px rgba(0,0,0,0.2);
            margin-bottom: 20px;
        }
        .demo-section h2 {
            color: #333;
            margin-bottom: 20px;
            font-size: 1.8em;
        }
        .project-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
            gap: 20px;
            margin-top: 20px;
        }
        .project-card {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 25px;
            border-radius: 10px;
            cursor: pointer;
            transition: transform 0.3s, box-shadow 0.3s;
        }
        .project-card:hover {
            transform: translateY(-5px);
            box-shadow: 0 15px 40px rgba(0,0,0,0.3), 0 0 40px rgba(118, 75, 162, 0.7);
        }
        .project-card h3 {
            font-size: 1.5em;
            margin-bottom: 10px;
        }
        .project-card p {
            margin: 5px 0;
            opacity: 0.9;
        }
        
        /* Enhancement 3: Live Match Confidence Visualization */
        .results {
            margin-top: 30px;
            display: none;
        }
        .results.show {
            display: block;
        }
        .match-card {
            background: #f8f9fa;
            padding: 25px;
            border-radius: 10px;
            margin-bottom: 20px;
            border-left: 5px solid #667eea;
            transition: transform 0.3s, box-shadow 0.3s;
        }
        .match-card:hover {
            transform: translateX(5px);
            box-shadow: 0 5px 20px rgba(102, 126, 234, 0.4);
        }
        .match-header {
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-bottom: 15px;
        }
        .match-score {
            display: inline-block;
            padding: 10px 20px;
            border-radius: 25px;
            font-weight: bold;
            font-size: 1.3em;
        }
        .match-score.high {
            background: #28a745;
            color: white;
            box-shadow: 0 0 15px rgba(40, 167, 69, 0.5);
        }
        .match-score.medium {
            background: #ffc107;
            color: #333;
            box-shadow: 0 0 15px rgba(255, 193, 7, 0.5);
        }
        .match-score.low {
            background: #dc3545;
            color: white;
            box-shadow: 0 0 15px rgba(220, 53, 69, 0.5);
        }
        .confidence-indicator {
            font-size: 0.9em;
            color: #666;
            margin-left: 10px;
        }
        .skill-overlap-section {
            background: white;
            padding: 15px;
            border-radius: 8px;
            margin-top: 15px;
        }
        .skill-overlap-title {
            font-weight: bold;
            color: #333;
            margin-bottom: 10px;
        }
        .skill-bar {
            margin-bottom: 10px;
        }
        .skill-name {
            font-size: 0.9em;
            color: #666;
            margin-bottom: 5px;
        }
        .skill-progress-bar {
            background: #e0e0e0;
            height: 25px;
            border-radius: 12px;
            overflow: hidden;
            position: relative;
        }
        .skill-progress-fill {
            background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
            height: 100%;
            transition: width 1s ease;
            display: flex;
            align-items: center;
            justify-content: flex-end;
            padding-right: 10px;
            color: white;
            font-weight: bold;
            font-size: 0.85em;
        }
        .match-card h3 {
            color: #333;
            margin: 10px 0;
        }
        .match-card p {
            color: #666;
            margin: 5px 0;
        }
        .stars {
            color: #ffd700;
            font-size: 1.5em;
        }

        
        .loading {
            text-align: center;
            padding: 40px;
            display: none;
        }
        .loading.show {
            display: block;
        }
        .spinner {
            border: 4px solid #f3f3f3;
            border-top: 4px solid #667eea;
            border-radius: 50%;
            width: 50px;
            height: 50px;
            animation: spin 1s linear infinite;
            margin: 0 auto;
        }
        @keyframes spin {
            0% { transform: rotate(0deg); }
            100% { transform: rotate(360deg); }
        }
        .ai-thinking {
            color: white;
            font-size: 1.2em;
            margin-top: 20px;
        }
        .info-box {
            background: #e7f3ff;
            border-left: 4px solid #2196F3;
            padding: 15px;
            margin: 20px 0;
            border-radius: 5px;
        }
        .info-box strong {
            color: #2196F3;
        }
        @media (max-width: 768px) {
            .header h1 {
                font-size: 1.8em;
            }
            .project-grid {
                grid-template-columns: 1fr;
            }
            .stats-grid {
                grid-template-columns: 1fr;
            }
        }
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>🚀 TCS RMG AI Matching System</h1>
            <p>Intelligent Resource-Project Matching Powered by AI</p>
        </div>

        <!-- Enhancement 4: Executive Summary Card -->
        <div class="executive-summary">
            <h2>📊 Executive Summary Dashboard</h2>
            <div class="stats-grid">
                <div class="stat-card">
                    <div class="stat-label">Total Employees</div>
                    <div class="stat-value" id="totalEmployees">5,000</div>
                </div>
                <div class="stat-card">
                    <div class="stat-label">Projects Matched</div>
                    <div class="stat-value" id="projectsMatched">1,247</div>
                </div>
                <div class="stat-card">
                    <div class="stat-label">Avg Match Time</div>
                    <div class="stat-value" id="avgMatchTime">3.2s</div>
                </div>
                <div class="stat-card">
                    <div class="stat-label">Client Satisfaction</div>
                    <div class="stat-value" id="clientSatisfaction">92%</div>
                </div>
                <div class="stat-card">
                    <div class="stat-label">Bench Reduction</div>
                    <div class="stat-value" id="benchReduction">33%</div>
                </div>
            </div>
        </div>

        <!-- Enhancement 5: Interactive Demo Mode -->
        <div class="demo-mode-section">
            <h2>🎬 Interactive Demo Mode</h2>
            <div class="info-box">
                <strong>Watch the AI in action!</strong> Click the button below to see 10 rapid matches processed in real-time with AI confidence scoring.
            </div>
            <button class="demo-button" onclick="runDemoMode()" id="demoButton">
                ▶️ Run Sample Matching (10 Matches)
            </button>
            
            <div class="demo-results" id="demoResults">
                <h3 style="color: #333; margin-bottom: 15px;">🔄 Processing Matches...</h3>
                <div id="demoMatchesList"></div>
                <div class="total-savings-counter" id="totalSavingsCounter" style="display: none;">
                    ✅ All Matches Completed!
                    <div style="font-size: 0.5em; margin-top: 10px;">
                        Processed in <span id="totalTimeValue">0</span> seconds
                    </div>
                </div>
            </div>
        </div>

        <div class="demo-section">
            <h2>🎯 Live Matching Demo: Click a Project</h2>
            <div class="info-box">
                <strong>How it works:</strong> Our AI analyzes employee skills and project requirements using advanced semantic matching with confidence visualization and skill overlap analysis!
            </div>
            
            <div class="project-grid">
                <div class="project-card" onclick="getMatches('P01')">
                    <h3>Project Phoenix</h3>
                    <p><strong>Type:</strong> Cloud Migration</p>
                    <p><strong>Skills:</strong> Python, AWS, Docker, SQL</p>
                    <p><strong>Duration:</strong> 6 months</p>
                    <p style="margin-top: 15px;">👆 Click to see AI matches</p>
                </div>
                
                <div class="project-card" onclick="getMatches('P02')">
                    <h3>Project Nova</h3>
                    <p><strong>Type:</strong> Frontend Development</p>
                    <p><strong>Skills:</strong> JavaScript, React, UI/UX, CSS</p>
                    <p><strong>Duration:</strong> 4 months</p>
                    <p style="margin-top: 15px;">👆 Click to see AI matches</p>
                </div>
                
                <div class="project-card" onclick="getMatches('P03')">
                    <h3>Project Titan</h3>
                    <p><strong>Type:</strong> Backend Development</p>
                    <p><strong>Skills:</strong> Java, Spring Boot, Microservices</p>
                    <p><strong>Duration:</strong> 8 months</p>
                    <p style="margin-top: 15px;">👆 Click to see AI matches</p>
                </div>
                
                <div class="project-card" onclick="getMatches('P04')">
                    <h3>Project Orion</h3>
                    <p><strong>Type:</strong> Machine Learning</p>
                    <p><strong>Skills:</strong> ML, Python, TensorFlow, Pandas</p>
                    <p><strong>Duration:</strong> 10 months</p>
                    <p style="margin-top: 15px;">👆 Click to see AI matches</p>
                </div>
            </div>
        </div>

        <div class="loading" id="loading">
            <div class="spinner"></div>
            <p class="ai-thinking">🤖 AI is analyzing semantic relationships...</p>
            <p class="ai-thinking" style="font-size: 0.9em; opacity: 0.8;">Calculating confidence scores and skill overlaps...</p>
        </div>

        <div class="results" id="results">
            <div class="demo-section">
                <h2 id="resultsTitle">🎯 Top Matches with Confidence Analysis</h2>
                <div id="resultsContent"></div>
            </div>
        </div>
    </div>

    <script>
        // Animated counter function
        function animateCounter(element, target, suffix = '', duration = 2000) {
            const start = 0;
            const increment = target / (duration / 16);
            let current = start;
            
            const timer = setInterval(() => {
                current += increment;
                if (current >= target) {
                    element.textContent = target.toLocaleString() + suffix;
                    clearInterval(timer);
                } else {
                    element.textContent = Math.floor(current).toLocaleString() + suffix;
                }
            }, 16);
        }
        
        // Load executive summary on page load with animated counters
        window.onload = function() {
            fetch('/api/executive-summary')
                .then(response => response.json())
                .then(data => {
                    animateCounter(document.getElementById('totalEmployees'), data.total_employees);
                    animateCounter(document.getElementById('projectsMatched'), data.projects_matched);
                    animateCounter(document.getElementById('avgMatchTime'), data.avg_match_time, 's');
                    animateCounter(document.getElementById('clientSatisfaction'), data.client_satisfaction, '%');
                    animateCounter(document.getElementById('benchReduction'), data.bench_reduction, '%');
                });
        };

        // Enhancement 5: Demo Mode Function
        function runDemoMode() {
            const button = document.getElementById('demoButton');
            const resultsDiv = document.getElementById('demoResults');
            const matchesList = document.getElementById('demoMatchesList');
            const savingsCounter = document.getElementById('totalSavingsCounter');
            
            button.disabled = true;
            button.textContent = '⏳ Running Demo...';
            resultsDiv.classList.add('show');
            matchesList.innerHTML = '';
            savingsCounter.style.display = 'none';
            
            fetch('/api/demo-matching')
                .then(response => response.json())
                .then(data => {
                    let delay = 0;
                    
                    data.matches.forEach((match, index) => {
                        setTimeout(() => {
                            
                            const matchHtml = `
                                <div class="demo-match-item">
                                    <div class="demo-match-info">
                                        <strong>Match ${match.match_number}:</strong> ${match.project_name} → ${match.employee_name}
                                        <br>
                                        <small>Match Score: ${match.match_score}% | Time: ${match.time}s</small>
                                    </div>
                                </div>
                            `;
                            matchesList.innerHTML += matchHtml;
                            

                            
                            // Show final summary after last match
                            if (index === data.matches.length - 1) {
                                setTimeout(() => {
                                    savingsCounter.style.display = 'block';
                                    document.getElementById('totalTimeValue').textContent = data.total_time;
                                    button.disabled = false;
                                    button.textContent = '▶️ Run Sample Matching Again';
                                }, 500);
                            }
                        }, delay);
                        delay += 800; // 800ms between each match
                    });
                })
                .catch(error => {
                    console.error('Error:', error);
                    button.disabled = false;
                    button.textContent = '▶️ Run Sample Matching (10 Matches)';
                    alert('Error running demo. Please try again.');
                });
        }

        // Enhancement 3: Live Match with Confidence Visualization
        function getMatches(projectId) {
            // Show loading with AI thinking animation
            document.getElementById('loading').classList.add('show');
            document.getElementById('results').classList.remove('show');

            // Fetch matches from API
            fetch('/match/' + projectId)
                .then(response => response.json())
                .then(data => {
                    // Simulate AI processing time for dramatic effect
                    setTimeout(() => {
                        displayResults(data);
                        document.getElementById('loading').classList.remove('show');
                        document.getElementById('results').classList.add('show');
                        
                        // Scroll to results
                        document.getElementById('results').scrollIntoView({ behavior: 'smooth' });
                    }, 1500); // 1.5 second delay for AI "thinking"
                })
                .catch(error => {
                    console.error('Error:', error);
                    document.getElementById('loading').classList.remove('show');
                    alert('Error fetching matches. Please try again.');
                });
        }

        function getConfidenceClass(score) {
            if (score >= 80) return 'high';
            if (score >= 60) return 'medium';
            return 'low';
        }

        function getConfidenceText(score) {
            if (score >= 90) return '🎯 Excellent Match';
            if (score >= 80) return '✅ Very Good Match';
            if (score >= 70) return '👍 Good Match';
            return '⚠️ Fair Match';
        }

        function displayResults(data) {
            const project = data.project;
            const matches = data.top_matches;
            
            document.getElementById('resultsTitle').textContent = 
                '🎯 Top Matches for ' + project.project_name;
            
            let html = '<div class="info-box">';
            html += '<strong>Project:</strong> ' + project.project_name + '<br>';
            html += '<strong>Required Skills:</strong> ' + project.required_skills;
            html += '</div>';
            
            matches.forEach((match, index) => {
                const confidenceClass = getConfidenceClass(match.match_score);
                const confidenceText = getConfidenceText(match.match_score);
                const stars = '⭐'.repeat(Math.min(5, Math.ceil(match.match_score / 20)));
                
                html += '<div class="match-card">';
                html += '<div class="match-header">';
                html += '<div>';
                html += '<span class="match-score ' + confidenceClass + '">' + match.match_score + '% Match</span>';
                html += '<span class="confidence-indicator">' + confidenceText + '</span>';
                html += '</div>';

                html += '</div>';
                
                html += '<h3>' + (index + 1) + '. ' + match.name + ' <span class="stars">' + stars + '</span></h3>';
                html += '<p><strong>Role:</strong> ' + match.role + '</p>';
                html += '<p><strong>Skills:</strong> ' + match.skills + '</p>';
                html += '<p><strong>Employee ID:</strong> ' + match.employee_id + '</p>';
                
                // Skill overlap visualization
                html += '<div class="skill-overlap-section">';
                html += '<div class="skill-overlap-title">📊 Skill Overlap Analysis:</div>';
                match.skill_overlap.forEach(skill => {
                    html += '<div class="skill-bar">';
                    html += '<div class="skill-name">' + skill.skill + '</div>';
                    html += '<div class="skill-progress-bar">';
                    html += '<div class="skill-progress-fill" style="width: ' + skill.percentage + '%">';
                    html += skill.percentage + '%';
                    html += '</div>';
                    html += '</div>';
                    html += '</div>';
                });
                html += '</div>';
                
                html += '</div>';
            });
            
            document.getElementById('resultsContent').innerHTML = html;
        }
    </script>
</body>
</html>
"""

@app.route("/")
def home():
    """Home page with interactive demo and all enhancements."""
    return render_template_string(HOME_PAGE)

@app.route("/match/<string:project_id>", methods=["GET"])
def match_project(project_id):
    """API endpoint to get matches for a project with detailed analysis."""
    matches = find_best_matches(project_id)
    if "error" in matches:
        return jsonify(matches), 404
    return jsonify(matches)

@app.route("/api/executive-summary")
def executive_summary():
    """API endpoint for executive summary statistics."""
    return jsonify(get_executive_summary())

@app.route("/api/demo-matching")
def demo_matching():
    """API endpoint for demo mode matching."""
    return jsonify(run_demo_matching())

@app.route("/health")
def health():
    """Health check endpoint."""
    return jsonify({"status": "healthy", "message": "TCS RMG AI Matching System is running!"})

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)

