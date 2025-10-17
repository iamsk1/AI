'''
# AI-Powered Resource Matching System - Proof of Concept

This directory contains a working proof-of-concept for the AI-Powered Intelligent Resource Matching System for TCS RMG.

**Author:** Manus AI  
**Date:** October 17, 2025

---

## 1. Purpose

This prototype demonstrates the core functionality of the AI matching system: **calculating a skill-based match score between employees and projects.**

It is a simplified, command-line version designed to prove the technical feasibility of using AI embeddings for resource allocation before building the full-featured application.

## 2. Technology Stack

- **Language:** Python 3.11
- **Core Libraries:**
  - `pandas`: For data manipulation.
  - `sentence-transformers`: For generating deep semantic embeddings for skills.
  - `scikit-learn`: For calculating cosine similarity.
  - `Flask`: For creating a simple web API to serve the results.

## 3. File Structure

```
/ai_matching_prototype
|-- matching_prototype.py   # The main Python script with the matching logic and API
|-- employees.csv           # Mock data for TCS employees and their skills
|-- projects.csv            # Mock data for available projects and their required skills
|-- README.md               # This file
```

## 4. How to Run the Prototype

Follow these steps to run the proof-of-concept on your local machine.

### Step 4.1: Install Dependencies

First, you need to install the required Python libraries. Open your terminal and run the following command:

```bash
pip3 install pandas sentence-transformers scikit-learn Flask
```

*(Note: The `sentence-transformers` library will download a pre-trained language model on its first run, which may take a few minutes and requires an internet connection.)*

### Step 4.2: Run the Flask Web Server

Once the dependencies are installed, navigate to the `ai_matching_prototype` directory in your terminal and run the main script:

```bash
python3.11 matching_prototype.py
```

If successful, you will see output similar to this, indicating that the server is running:

```
 * Serving Flask app 'matching_prototype'
 * Debug mode: off
WARNING: This is a development server. Do not use it in a production deployment.
Use a production WSGI server instead.
 * Running on http://127.0.0.1:5000
Press CTRL+C to quit
```

This means the AI matching service is now active and ready to receive requests.

### Step 4.3: Test the Matching API

Now you can test the matching functionality by sending a request to the running server.

Open a **new terminal window** (while the server is still running) and use `curl` to get matches for a specific project. For example, to find the best matches for **Project Phoenix (P01)**:

```bash
curl http://127.0.0.1:5000/match/P01
```

**Expected Output:**

You will receive a JSON response containing a ranked list of employees who are the best fit for the project, sorted by their match score:

```json
{
  "project": {
    "project_id": "P01",
    "project_name": "Project Phoenix",
    "required_skills": "Python,AWS,Docker,SQL"
  },
  "top_matches": [
    {
      "employee_id": 101,
      "name": "Anjali Sharma",
      "match_score": 0.92,
      "role": "Senior Software Engineer"
    },
    {
      "employee_id": 108,
      "name": "Karan Gupta",
      "match_score": 0.85,
      "role": "DevOps Engineer"
    },
    {
      "employee_id": 104,
      "name": "Amit Singh",
      "match_score": 0.78,
      "role": "Cloud Architect"
    }
  ]
}
```

*(Note: The exact scores may vary slightly depending on the model version, but the ranking should be consistent.)*

You can try this for other projects by changing the project ID in the URL (e.g., `/match/P02`, `/match/P03`, `/match/P04`).

## 5. How It Works: The Logic

1.  **Load Data:** The script loads `employees.csv` and `projects.csv` into pandas DataFrames.
2.  **Generate Embeddings:** It uses the `all-MiniLM-L6-v2` model from `sentence-transformers` to convert the string of skills for each employee and project into a 384-dimensional numerical vector (an "embedding"). This embedding captures the semantic meaning of the skills.
3.  **Calculate Similarity:** When you request a match for a project, the script takes the project's skill embedding and calculates the **cosine similarity** against the skill embeddings of all employees.
4.  **Rank and Return:** The employees are ranked by their cosine similarity score (from highest to lowest), and the top 3 matches are returned as a JSON response.

## 6. Next Steps

This prototype validates the core AI matching concept. The next steps in a full implementation would be:

-   **Develop a full-featured web interface** (UI) using React.
-   **Integrate with real TCS databases** (HRMS, Project Systems) instead of CSV files.
-   **Expand the matching algorithm** to include more factors (e.g., seniority, availability, employee preference, location).
-   **Deploy the system on a scalable cloud infrastructure** (AWS/Azure) using Docker and Kubernetes.
-   **Build out the other services** (notifications, user management, etc.) as detailed in the implementation guide.
'''
