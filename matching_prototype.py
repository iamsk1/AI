
import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from flask import Flask, jsonify

# --- 1. Data Loading and AI Model Initialization ---

# Load the pre-trained model for generating sentence/skill embeddings
# This model is optimized for semantic similarity tasks.
model = SentenceTransformer("all-MiniLM-L6-v2")

# Load mock data for employees and projects from CSV files
try:
    employees_df = pd.read_csv("employees.csv")
    projects_df = pd.read_csv("projects.csv")
except FileNotFoundError:
    print("Error: Make sure employees.csv and projects.csv are in the same directory.")
    exit()

# --- 2. Generate AI Embeddings ---

# Convert the string of skills for each employee and project into a numerical vector (embedding)
# This is the core of the AI matching. The embeddings capture the semantic meaning of the skills.

# Generate embeddings for employee skills
employee_skills_list = employees_df["skills"].tolist()
employee_embeddings = model.encode(employee_skills_list)

# Generate embeddings for project requirements
project_skills_list = projects_df["required_skills"].tolist()
project_embeddings = model.encode(project_skills_list)

# --- 3. Core Matching Logic ---

def find_best_matches(project_id, top_n=3):
    """
    Finds the best employee matches for a given project ID.

    Args:
        project_id (str): The ID of the project to match (e.g., "P01").
        top_n (int): The number of top matches to return.

    Returns:
        dict: A dictionary containing project details and a list of the top N matched employees.
    """
    # Find the target project and its skill embedding
    try:
        project_index = projects_df[projects_df["project_id"] == project_id].index[0]
        project_embedding = project_embeddings[project_index]
    except IndexError:
        return {"error": f"Project with ID 	{project_id}	 not found."}

    # Calculate the cosine similarity between the project and ALL employees
    # This measures how similar the project's requirements are to each employee's skills.
    similarities = cosine_similarity([project_embedding], employee_embeddings)[0]

    # Create a DataFrame with employees and their calculated match scores
    results_df = employees_df.copy()
    results_df["match_score"] = similarities

    # Sort the results by match score in descending order
    results_df = results_df.sort_values(by="match_score", ascending=False)

    # Get the top N matches
    top_matches = results_df.head(top_n)

    # Format the output
    project_info = projects_df.iloc[project_index].to_dict()
    matches_list = []
    for _, row in top_matches.iterrows():
        matches_list.append({
            "employee_id": row["employee_id"],
            "name": row["name"],
            "role": row["role"],
            "match_score": round(row["match_score"], 2)
        })

    return {
        "project": project_info,
        "top_matches": matches_list
    }

# --- 4. Flask API to Serve the Results ---

app = Flask(__name__)

@app.route("/match/<string:project_id>", methods=["GET"])
def match_project(project_id):
    """API endpoint to get matches for a project."""
    matches = find_best_matches(project_id)
    if "error" in matches:
        return jsonify(matches), 404
    return jsonify(matches)

# --- Main Execution Block ---

if __name__ == "__main__":
    print("AI Matching System Prototype Initialized.")
    print("Embeddings generated for all employees and projects.")
    print("Starting Flask server...")
    print("To test, run: curl http://127.0.0.1:5000/match/P01")
    app.run(host="0.0.0.0", port=5000)

