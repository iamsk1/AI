import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from flask import Flask, jsonify, render_template_string
import os

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
        matches_list.append({
            "employee_id": int(row["employee_id"]),
            "name": row["name"],
            "role": row["role"],
            "skills": row["skills"],
            "match_score": round(float(row["match_score"]) * 100, 1)
        })

    return {
        "project": project_info,
        "top_matches": matches_list
    }

# --- 3. Flask App ---

app = Flask(__name__)

# HTML template for the home page
HOME_PAGE = """
<!DOCTYPE html>
<html>
<head>
    <title>TCS RMG AI Matching System - Demo</title>
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
            max-width: 1200px;
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
            box-shadow: 0 15px 40px rgba(0,0,0,0.3);
        }
        .project-card h3 {
            font-size: 1.5em;
            margin-bottom: 10px;
        }
        .project-card p {
            margin: 5px 0;
            opacity: 0.9;
        }
        .results {
            margin-top: 30px;
            display: none;
        }
        .results.show {
            display: block;
        }
        .match-card {
            background: #f8f9fa;
            padding: 20px;
            border-radius: 10px;
            margin-bottom: 15px;
            border-left: 5px solid #667eea;
        }
        .match-score {
            display: inline-block;
            background: #667eea;
            color: white;
            padding: 5px 15px;
            border-radius: 20px;
            font-weight: bold;
            font-size: 1.2em;
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
        }
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>🚀 TCS RMG AI Matching System</h1>
            <p>Intelligent Resource-Project Matching Powered by AI</p>
        </div>

        <div class="demo-section">
            <h2>📊 Live Demo: Click a Project to See AI Matches</h2>
            <div class="info-box">
                <strong>How it works:</strong> Our AI analyzes employee skills and project requirements using advanced semantic matching to find the best candidates in seconds!
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
            <p style="margin-top: 20px; color: white; font-size: 1.2em;">AI is analyzing matches...</p>
        </div>

        <div class="results" id="results">
            <div class="demo-section">
                <h2 id="resultsTitle">🎯 Top Matches</h2>
                <div id="resultsContent"></div>
            </div>
        </div>
    </div>

    <script>
        function getMatches(projectId) {
            // Show loading
            document.getElementById('loading').classList.add('show');
            document.getElementById('results').classList.remove('show');

            // Fetch matches from API
            fetch('/match/' + projectId)
                .then(response => response.json())
                .then(data => {
                    displayResults(data);
                    document.getElementById('loading').classList.remove('show');
                    document.getElementById('results').classList.add('show');
                    
                    // Scroll to results
                    document.getElementById('results').scrollIntoView({ behavior: 'smooth' });
                })
                .catch(error => {
                    console.error('Error:', error);
                    document.getElementById('loading').classList.remove('show');
                    alert('Error fetching matches. Please try again.');
                });
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
                const stars = '⭐'.repeat(Math.min(5, Math.ceil(match.match_score / 20)));
                html += '<div class="match-card">';
                html += '<span class="match-score">' + match.match_score + '% Match</span>';
                html += '<span class="stars"> ' + stars + '</span>';
                html += '<h3>' + (index + 1) + '. ' + match.name + '</h3>';
                html += '<p><strong>Role:</strong> ' + match.role + '</p>';
                html += '<p><strong>Skills:</strong> ' + match.skills + '</p>';
                html += '<p><strong>Employee ID:</strong> ' + match.employee_id + '</p>';
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
    """Home page with interactive demo."""
    return render_template_string(HOME_PAGE)

@app.route("/match/<string:project_id>", methods=["GET"])
def match_project(project_id):
    """API endpoint to get matches for a project."""
    matches = find_best_matches(project_id)
    if "error" in matches:
        return jsonify(matches), 404
    return jsonify(matches)

@app.route("/health")
def health():
    """Health check endpoint."""
    return jsonify({"status": "healthy", "message": "TCS RMG AI Matching System is running!"})

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)

