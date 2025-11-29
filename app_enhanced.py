"""
TCS RMG AI Matching System - Enhanced Version with Patentable AI
================================================================

This version integrates three novel AI innovations:
1. Continuous Learning Framework - learns from feedback
2. Explainability Engine - explains WHY matches were made
3. Fairness Monitor - ensures unbiased recommendations

Author: Soumik Karmakar
Date: November 2025
Version: 2.0.0 (Enhanced with Patentable AI)
"""

import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from flask import Flask, jsonify, render_template_string, request
import os
import numpy as np
from datetime import datetime

# Import our novel AI modules
from src.continuous_learning import ContinuousLearningEngine
from src.explainability import ExplainabilityEngine, generate_explanation_report
from src.fairness_monitor import FairnessMonitor

# --- 1. Data Loading and AI Model Initialization ---

print("=" * 70)
print("TCS RMG AI Matching System - Enhanced Version 2.0.0")
print("=" * 70)
print("\n🤖 Loading AI model...")
model = SentenceTransformer("all-MiniLM-L6-v2")
print("✓ AI model loaded successfully!")

# Load data
print("\n📊 Loading employee and project data...")
employees_df = pd.read_csv("employees.csv")
projects_df = pd.read_csv("projects.csv")
print(f"✓ Loaded {len(employees_df)} employees and {len(projects_df)} projects")

# Generate embeddings
print("\n🧠 Generating semantic embeddings...")
employee_skills_list = employees_df["skills"].tolist()
employee_embeddings = model.encode(employee_skills_list)

project_skills_list = projects_df["required_skills"].tolist()
project_embeddings = model.encode(project_skills_list)
print("✓ Embeddings generated successfully!")

# Initialize our novel AI systems
print("\n🚀 Initializing patentable AI enhancements...")

print("  1. Continuous Learning Engine...")
learning_engine = ContinuousLearningEngine(
    base_model=model,
    learning_rate=0.001,
    buffer_size=10,  # Learn after every 10 feedback samples
    confidence_threshold=0.7
)
print("  ✓ Continuous learning enabled")

print("  2. Explainability Engine...")
explainer = ExplainabilityEngine(model)
print("  ✓ Explainability enabled")

print("  3. Fairness Monitor...")
fairness_monitor = FairnessMonitor(
    fairness_thresholds={
        'gender': 0.10,
        'location': 0.10,
        'experience_level': 0.15
    },
    enable_auto_correction=True
)
print("  ✓ Fairness monitoring enabled")

print("\n" + "=" * 70)
print("✅ All systems initialized successfully!")
print("=" * 70 + "\n")

# --- 2. Enhanced Matching Logic ---

def find_best_matches_enhanced(project_id, top_n=5):
    """
    Find best employee matches with AI enhancements.
    
    This function extends the basic matching with:
    - Confidence scores (from continuous learning)
    - Detailed explanations (from explainability engine)
    - Fairness checking (from fairness monitor)
    """
    try:
        # Get project data
        project_index = projects_df[projects_df["project_id"] == project_id].index[0]
        project_embedding = project_embeddings[project_index]
        project_info = projects_df.iloc[project_index].to_dict()
    except IndexError:
        return {"error": f"Project with ID '{project_id}' not found."}
    
    # Calculate similarities
    similarities = cosine_similarity([project_embedding], employee_embeddings)[0]
    
    # Create results dataframe
    results_df = employees_df.copy()
    results_df["match_score"] = similarities
    results_df = results_df.sort_values(by="match_score", ascending=False)
    
    # Get top matches
    top_matches = results_df.head(top_n)
    
    # Enhance each match with AI features
    matches_list = []
    for _, row in top_matches.iterrows():
        match_score = float(row["match_score"])
        
        # 1. Calculate confidence score (Continuous Learning)
        confidence_score = learning_engine.calculate_confidence(
            match_score,
            project_embedding,
            employee_embeddings[row.name]
        )
        
        # 2. Check if should defer to human (Continuous Learning)
        should_defer, defer_reason = learning_engine.should_defer_to_human(
            confidence_score,
            match_score
        )
        
        # 3. Generate explanation (Explainability Engine)
        explanation = explainer.explain_match(
            employee_skills=row["skills"],
            project_requirements=project_info["required_skills"],
            match_score=match_score,
            employee_metadata={
                'name': row["name"],
                'role': row["role"],
                'employee_id': row["employee_id"]
            },
            project_metadata={
                'project_name': project_info["project_name"],
                'project_id': project_info["project_id"]
            }
        )
        
        # Build enhanced match object
        match = {
            'employee_id': int(row["employee_id"]),
            'name': row["name"],
            'role': row["role"],
            'skills': row["skills"],
            'match_score': round(match_score * 100, 1),
            
            # NEW: Continuous Learning Features
            'confidence_score': round(confidence_score * 100, 1),
            'should_defer': should_defer,
            'defer_reason': defer_reason,
            'model_version': learning_engine.model_version,
            
            # NEW: Explainability Features
            'explanation': explanation['natural_language'],
            'skill_contributions': explanation['skill_contributions'][:5],  # Top 5
            'improvements': explanation['improvements'][:3],  # Top 3
            'confidence_explanation': explanation['confidence_explanation'],
            
            # Metadata
            'timestamp': datetime.now().isoformat()
        }
        
        matches_list.append(match)
    
    # 4. Check fairness (Fairness Monitor)
    fairness_report = fairness_monitor.monitor_match_fairness(
        matches_list,
        employees_df,
        demographic_fields=['role', 'skills']  # Use available fields
    )
    
    # 5. Apply fairness correction if needed
    if not fairness_report['compliance_status'] and fairness_monitor.enable_auto_correction:
        matches_list = fairness_monitor.apply_fairness_constraints(
            matches_list,
            employees_df,
            target_fairness=0.90
        )
    
    return {
        'project': project_info,
        'top_matches': matches_list,
        'fairness_report': fairness_report,
        'learning_stats': learning_engine.get_learning_stats(),
        'metadata': {
            'total_candidates_analyzed': len(employees_df),
            'model_version': learning_engine.model_version,
            'timestamp': datetime.now().isoformat()
        }
    }

def record_decision_enhanced(project_id, employee_id, decision, reason=""):
    """
    Record Accept/Reject decision and trigger continuous learning.
    
    This is where the magic happens - every human decision makes the AI smarter!
    """
    try:
        # Get project and employee data
        project_idx = projects_df[projects_df["project_id"] == project_id].index[0]
        employee_idx = employees_df[employees_df["employee_id"] == employee_id].index[0]
        
        project_embedding = project_embeddings[project_idx]
        employee_embedding = employee_embeddings[employee_idx]
        
        # Calculate match score
        match_score = float(cosine_similarity(
            [project_embedding],
            [employee_embedding]
        )[0][0])
        
        # Record feedback in continuous learning system
        feedback_result = learning_engine.collect_feedback(
            project_id=str(project_id),
            employee_id=str(employee_id),
            project_embedding=project_embedding,
            employee_embedding=employee_embedding,
            match_score=match_score,
            decision=decision,
            metadata={
                'reason': reason,
                'timestamp': datetime.now().isoformat()
            }
        )
        
        return {
            'status': 'success',
            'message': f'Decision recorded: {decision}',
            'feedback_result': feedback_result
        }
        
    except Exception as e:
        return {
            'status': 'error',
            'message': str(e)
        }

# --- 3. Flask App ---

app = Flask(__name__)

@app.route('/')
def home():
    """Enhanced home page with AI features showcase."""
    return render_template_string(ENHANCED_HOME_PAGE)

@app.route('/api/match/<project_id>')
def api_match(project_id):
    """API endpoint for enhanced matching."""
    top_n = request.args.get('top_n', default=5, type=int)
    results = find_best_matches_enhanced(project_id, top_n)
    return jsonify(results)

@app.route('/api/decision', methods=['POST'])
def api_decision():
    """API endpoint to record Accept/Reject decision."""
    data = request.json
    result = record_decision_enhanced(
        project_id=data.get('project_id'),
        employee_id=data.get('employee_id'),
        decision=data.get('decision'),
        reason=data.get('reason', '')
    )
    return jsonify(result)

@app.route('/api/learning-stats')
def api_learning_stats():
    """API endpoint to get continuous learning statistics."""
    stats = learning_engine.get_learning_stats()
    return jsonify(stats)

@app.route('/api/fairness-report')
def api_fairness_report():
    """API endpoint to get fairness compliance report."""
    report = fairness_monitor.generate_compliance_report()
    return jsonify(report)

# --- 4. Enhanced UI Template ---

ENHANCED_HOME_PAGE = """
<!DOCTYPE html>
<html>
<head>
    <title>TCS RMG AI Matching System v2.0 - Enhanced with Patentable AI</title>
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
        .version-badge {
            display: inline-block;
            background: linear-gradient(135deg, #667eea, #764ba2);
            color: white;
            padding: 8px 20px;
            border-radius: 20px;
            font-size: 0.9em;
            font-weight: bold;
            margin-top: 10px;
        }
        .ai-features {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
            gap: 20px;
            margin-bottom: 30px;
        }
        .feature-card {
            background: white;
            padding: 25px;
            border-radius: 15px;
            box-shadow: 0 5px 15px rgba(0,0,0,0.1);
        }
        .feature-card h3 {
            color: #667eea;
            margin-bottom: 15px;
            font-size: 1.3em;
        }
        .feature-card ul {
            list-style: none;
            padding-left: 0;
        }
        .feature-card li {
            padding: 8px 0;
            border-bottom: 1px solid #f0f0f0;
        }
        .feature-card li:before {
            content: "✓ ";
            color: #10b981;
            font-weight: bold;
            margin-right: 8px;
        }
        .demo-section {
            background: white;
            padding: 30px;
            border-radius: 15px;
            box-shadow: 0 10px 30px rgba(0,0,0,0.2);
        }
        .demo-section h2 {
            color: #333;
            margin-bottom: 20px;
        }
        select, button {
            padding: 12px 20px;
            font-size: 1em;
            border-radius: 8px;
            border: 2px solid #667eea;
            margin-right: 10px;
        }
        button {
            background: linear-gradient(135deg, #667eea, #764ba2);
            color: white;
            cursor: pointer;
            font-weight: bold;
        }
        button:hover {
            transform: translateY(-2px);
            box-shadow: 0 5px 15px rgba(102, 126, 234, 0.4);
        }
        .results {
            margin-top: 30px;
        }
        .match-card {
            background: #f9fafb;
            padding: 20px;
            border-radius: 10px;
            margin-bottom: 15px;
            border-left: 4px solid #667eea;
        }
        .match-header {
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-bottom: 15px;
        }
        .match-score {
            font-size: 2em;
            font-weight: bold;
            color: #667eea;
        }
        .confidence-badge {
            display: inline-block;
            padding: 5px 15px;
            border-radius: 15px;
            font-size: 0.9em;
            font-weight: bold;
        }
        .confidence-high {
            background: #d1fae5;
            color: #065f46;
        }
        .confidence-medium {
            background: #fef3c7;
            color: #92400e;
        }
        .confidence-low {
            background: #fee2e2;
            color: #991b1b;
        }
        .explanation {
            background: white;
            padding: 15px;
            border-radius: 8px;
            margin: 15px 0;
            font-size: 0.95em;
            line-height: 1.6;
        }
        .skill-contributions {
            margin-top: 15px;
        }
        .skill-bar {
            margin: 10px 0;
        }
        .skill-bar-label {
            display: flex;
            justify-content: space-between;
            margin-bottom: 5px;
            font-size: 0.9em;
        }
        .skill-bar-fill {
            height: 8px;
            background: linear-gradient(90deg, #667eea, #764ba2);
            border-radius: 4px;
            transition: width 0.5s ease;
        }
        .action-buttons {
            margin-top: 15px;
            display: flex;
            gap: 10px;
        }
        .btn-accept {
            background: #10b981;
            border-color: #10b981;
        }
        .btn-reject {
            background: #ef4444;
            border-color: #ef4444;
        }
        .loading {
            text-align: center;
            padding: 40px;
            font-size: 1.2em;
            color: #667eea;
        }
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>🚀 TCS RMG AI Matching System</h1>
            <div class="version-badge">v2.0.0 - Enhanced with Patentable AI</div>
            <p style="margin-top: 15px; color: #666;">
                Powered by Continuous Learning • Explainable AI • Fairness Monitoring
            </p>
        </div>

        <div class="ai-features">
            <div class="feature-card">
                <h3>🧠 Continuous Learning</h3>
                <ul>
                    <li>Learns from every Accept/Reject decision</li>
                    <li>Real-time model updates without retraining</li>
                    <li>Confidence calibration improves over time</li>
                    <li>Automatic model versioning</li>
                </ul>
            </div>

            <div class="feature-card">
                <h3>💡 Explainability Engine</h3>
                <ul>
                    <li>Explains WHY each match was recommended</li>
                    <li>Skill-by-skill contribution analysis</li>
                    <li>Actionable improvement suggestions</li>
                    <li>Natural language explanations</li>
                </ul>
            </div>

            <div class="feature-card">
                <h3>⚖️ Fairness Monitoring</h3>
                <ul>
                    <li>Real-time bias detection</li>
                    <li>Automatic fairness correction</li>
                    <li>Compliance reporting for audits</li>
                    <li>Multi-dimensional fairness metrics</li>
                </ul>
            </div>
        </div>

        <div class="demo-section">
            <h2>🎯 Try the Enhanced Matching System</h2>
            <div style="margin-bottom: 20px;">
                <select id="projectSelect">
                    <option value="">Select a project...</option>
                    <option value="PROJ001">PROJ001 - E-Commerce Platform Development</option>
                    <option value="PROJ002">PROJ002 - Mobile Banking App</option>
                    <option value="PROJ003">PROJ003 - Data Analytics Dashboard</option>
                </select>
                <button onclick="findMatches()">Find Best Matches</button>
            </div>

            <div id="results" class="results"></div>
        </div>
    </div>

    <script>
        async function findMatches() {
            const projectId = document.getElementById('projectSelect').value;
            if (!projectId) {
                alert('Please select a project');
                return;
            }

            const resultsDiv = document.getElementById('results');
            resultsDiv.innerHTML = '<div class="loading">🔍 Analyzing candidates with AI...</div>';

            try {
                const response = await fetch(`/api/match/${projectId}?top_n=3`);
                const data = await response.json();

                if (data.error) {
                    resultsDiv.innerHTML = `<p style="color: red;">${data.error}</p>`;
                    return;
                }

                displayResults(data);
            } catch (error) {
                resultsDiv.innerHTML = `<p style="color: red;">Error: ${error.message}</p>`;
            }
        }

        function displayResults(data) {
            const resultsDiv = document.getElementById('results');
            let html = `<h3>Top Matches for ${data.project.project_name}</h3>`;
            
            // Fairness Report
            html += `
                <div style="background: #f0f9ff; padding: 15px; border-radius: 8px; margin-bottom: 20px;">
                    <strong>Fairness Status:</strong> ${data.fairness_report.summary}
                </div>
            `;

            // Matches
            data.top_matches.forEach((match, index) => {
                const confidenceClass = match.confidence_score >= 80 ? 'confidence-high' : 
                                       match.confidence_score >= 60 ? 'confidence-medium' : 'confidence-low';
                
                html += `
                    <div class="match-card">
                        <div class="match-header">
                            <div>
                                <h3>${match.name}</h3>
                                <p style="color: #666;">${match.role}</p>
                            </div>
                            <div class="match-score">${match.match_score}%</div>
                        </div>

                        <div>
                            <span class="confidence-badge ${confidenceClass}">
                                Confidence: ${match.confidence_score}%
                            </span>
                            ${match.should_defer ? 
                                '<span style="color: #f59e0b; margin-left: 10px;">⚠️ Human review recommended</span>' : 
                                '<span style="color: #10b981; margin-left: 10px;">✓ High confidence</span>'}
                        </div>

                        <div class="explanation">
                            <strong>Why this match?</strong><br>
                            ${match.explanation}
                        </div>

                        <div class="skill-contributions">
                            <strong>Top Skill Matches:</strong>
                            ${match.skill_contributions.slice(0, 3).map(skill => `
                                <div class="skill-bar">
                                    <div class="skill-bar-label">
                                        <span>${skill.required_skill} → ${skill.matched_with}</span>
                                        <span>${skill.contribution_percentage.toFixed(0)}%</span>
                                    </div>
                                    <div style="background: #e5e7eb; border-radius: 4px;">
                                        <div class="skill-bar-fill" style="width: ${skill.contribution_percentage}%"></div>
                                    </div>
                                </div>
                            `).join('')}
                        </div>

                        ${match.improvements.length > 0 ? `
                            <div style="margin-top: 15px; padding: 10px; background: #fef3c7; border-radius: 5px;">
                                <strong>💡 Improvement Suggestion:</strong><br>
                                ${match.improvements[0].recommended_action}
                                (Potential improvement: +${(match.improvements[0].potential_improvement * 100).toFixed(1)}%)
                            </div>
                        ` : ''}

                        <div class="action-buttons">
                            <button class="btn-accept" onclick="recordDecision('${data.project.project_id}', ${match.employee_id}, 'accept')">
                                ✓ Accept Match
                            </button>
                            <button class="btn-reject" onclick="recordDecision('${data.project.project_id}', ${match.employee_id}, 'reject')">
                                ✗ Reject Match
                            </button>
                        </div>
                    </div>
                `;
            });

            // Learning Stats
            html += `
                <div style="background: #f9fafb; padding: 15px; border-radius: 8px; margin-top: 20px;">
                    <strong>🧠 AI Learning Status:</strong><br>
                    Model Version: ${data.learning_stats.model_version} | 
                    Total Feedback: ${data.learning_stats.total_feedback} | 
                    Current Accuracy: ${(data.learning_stats.current_accuracy * 100).toFixed(1)}%
                </div>
            `;

            resultsDiv.innerHTML = html;
        }

        async function recordDecision(projectId, employeeId, decision) {
            try {
                const response = await fetch('/api/decision', {
                    method: 'POST',
                    headers: {'Content-Type': 'application/json'},
                    body: JSON.stringify({
                        project_id: projectId,
                        employee_id: employeeId,
                        decision: decision
                    })
                });

                const result = await response.json();
                
                if (result.status === 'success') {
                    alert(`✓ Decision recorded: ${decision.toUpperCase()}\\n\\n${
                        result.feedback_result.learning_triggered ? 
                        '🧠 AI model has been updated with your feedback!' :
                        `📊 Feedback collected (${result.feedback_result.buffer_size}/${result.feedback_result.total_feedback})`
                    }`);
                    
                    // Refresh results to show updated model
                    findMatches();
                } else {
                    alert('Error recording decision: ' + result.message);
                }
            } catch (error) {
                alert('Error: ' + error.message);
            }
        }
    </script>
</body>
</html>
"""

# --- 5. Run the App ---

if __name__ == '__main__':
    print("\n" + "=" * 70)
    print("🌐 Starting Enhanced TCS RMG AI Matching System...")
    print("=" * 70)
    print("\n📍 Access the system at: http://localhost:5000")
    print("\n💡 Features enabled:")
    print("  • Continuous Learning from feedback")
    print("  • Explainable AI recommendations")
    print("  • Fairness monitoring and correction")
    print("\n" + "=" * 70 + "\n")
    
    app.run(debug=True, host='0.0.0.0', port=5000)
