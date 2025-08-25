from flask import Flask, request, render_template_string
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import datetime
import matplotlib.pyplot as plt
import io
import base64
import uuid

app = Flask(__name__)

# Hardcoded current date as per the given context
current_date = datetime.date(2025, 8, 24)

# In-memory storage for tasks
tasks = []

# Task class with additional attributes
class Task:
    def __init__(self, description, urgency, importance, due_date):
        self.unique_id = str(uuid.uuid4())
        self.description = description
        self.urgency = int(urgency)
        self.importance = int(importance)
        self.due_date = due_date
        self.days_left = max((datetime.datetime.strptime(due_date, '%Y-%m-%d').date() - current_date).days, 0)
        self.factor = max(10 - self.days_left / 3.0, 1)
        self.cluster = 0
        self.priority = 'Unclassified'

# HTML template with Bootstrap for improved UI
HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Project Management System</title>
    <link rel="stylesheet" href="https://stackpath.bootstrapcdn.com/bootstrap/4.5.2/css/bootstrap.min.css">
    <style>
        body { background-color: #f8f9fa; }
        .container { max-width: 1200px; }
        .table { background-color: white; }
        .btn-sm { margin-right: 5px; }
        .chart-container { max-width: 600px; margin: 20px auto; }
    </style>
</head>
<body>
    <div class="container mt-4">
        <h1 class="text-center mb-4">Project Management System</h1>
        
        <div class="card mb-4">
            <div class="card-body">
                <h2 class="card-title">Add New Task</h2>
                <form method="POST" action="/add_task">
                    <div class="form-group">
                        <label for="description">Task Description</label>
                        <input type="text" class="form-control" id="description" name="description" placeholder="Enter task description" required>
                    </div>
                    <div class="form-row">
                        <div class="form-group col-md-6">
                            <label for="urgency">Urgency (1-10)</label>
                            <input type="number" class="form-control" id="urgency" name="urgency" min="1" max="10" required>
                        </div>
                        <div class="form-group col-md-6">
                            <label for="importance">Importance (1-10)</label>
                            <input type="number" class="form-control" id="importance" name="importance" min="1" max="10" required>
                        </div>
                    </div>
                    <div class="form-group">
                        <label for="due_date">Due Date</label>
                        <input type="date" class="form-control" id="due_date" name="due_date" required>
                    </div>
                    <button type="submit" class="btn btn-primary">Add Task</button>
                </form>
            </div>
        </div>

        <h2 class="mt-4">Tasks</h2>
        {% if sorted_tasks %}
            <div class="table-responsive">
                <table class="table table-striped table-hover">
                    <thead class="thead-dark">
                        <tr>
                            <th>Description</th>
                            <th>Urgency</th>
                            <th>Importance</th>
                            <th>Due Date</th>
                            <th>Days Left</th>
                            <th>Priority</th>
                            <th>Actions</th>
                        </tr>
                    </thead>
                    <tbody>
                        {% for task in sorted_tasks %}
                            <tr class="{{ 'table-danger' if task.priority == 'High' else 'table-warning' if task.priority == 'Medium' else 'table-success' if task.priority == 'Low' else 'table-secondary' }}">
                                <td>{{ task.description }}</td>
                                <td>{{ task.urgency }}</td>
                                <td>{{ task.importance }}</td>
                                <td>{{ task.due_date }}</td>
                                <td>{{ task.days_left }}</td>
                                <td>{{ task.priority }}</td>
                                <td>
                                    <form method="POST" action="/delete/{{ task.unique_id }}" style="display:inline;">
                                        <button type="submit" class="btn btn-danger btn-sm">Delete</button>
                                    </form>
                                </td>
                            </tr>
                        {% endfor %}
                    </tbody>
                </table>
            </div>
        {% else %}
            <div class="alert alert-info">No tasks available.</div>
        {% endif %}

        {% if img_base64 %}
            <h2 class="mt-4 text-center">Task Cluster Visualization</h2>
            <div class="chart-container">
                <img src="data:image/png;base64,{{ img_base64 }}" class="img-fluid" alt="Cluster Visualization">
            </div>
        {% endif %}
    </div>
</body>
</html>
"""

@app.route('/')
def index():
    img_base64 = None
    sorted_tasks = tasks[:]  # Default to unsorted if no clustering

    if len(tasks) >= 2:
        # Prepare data for clustering
        X = np.array([[t.urgency, t.importance, t.factor] for t in tasks])
        
        # Scale features
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        # Apply KMeans clustering
        n_clusters = min(3, len(tasks))
        kmeans = KMeans(n_clusters=n_clusters, random_state=0)
        clusters = kmeans.fit_predict(X_scaled)
        
        # Update task clusters
        for t, c in zip(tasks, clusters):
            t.cluster = c
        
        # Assign meaningful priority labels based on average scores
        if n_clusters > 1:
            cluster_scores = []
            for i in range(n_clusters):
                cluster_tasks = [t for t in tasks if t.cluster == i]
                if cluster_tasks:
                    avg_urg = np.mean([t.urgency for t in cluster_tasks])
                    avg_imp = np.mean([t.importance for t in cluster_tasks])
                    avg_fac = np.mean([t.factor for t in cluster_tasks])
                    score = avg_urg + avg_imp + avg_fac
                    cluster_scores.append(score)
                else:
                    cluster_scores.append(0)
            
            priority_order = np.argsort(-np.array(cluster_scores))
            priority_labels = ['High', 'Medium', 'Low'][:n_clusters]
            cluster_to_priority = {priority_order[rank]: priority_labels[rank] for rank in range(n_clusters)}
            
            for t in tasks:
                t.priority = cluster_to_priority.get(t.cluster, 'Unclassified')
        else:
            for t in tasks:
                t.priority = 'Medium'
        
        # Generate visualization
        plt.figure(figsize=(8, 6))
        colors = ['#ff4444', '#ffbb33', '#00C851']  # Red for High, Yellow Medium, Green Low
        for i in range(n_clusters):
            cluster_urg = [t.urgency for t in tasks if t.cluster == i]
            cluster_imp = [t.importance for t in tasks if t.cluster == i]
            label = priority_labels[np.where(priority_order == i)[0][0]] if n_clusters > 1 else 'Cluster'
            plt.scatter(cluster_urg, cluster_imp, c=colors[i], label=label, s=100, alpha=0.7)
        plt.xlabel('Urgency', fontsize=12)
        plt.ylabel('Importance', fontsize=12)
        plt.title('Task Clusters (Urgency vs Importance)', fontsize=14)
        plt.legend()
        plt.grid(True, linestyle='--', alpha=0.5)
        buf = io.BytesIO()
        plt.savefig(buf, format='png', bbox_inches='tight')
        buf.seek(0)
        img_base64 = base64.b64encode(buf.read()).decode('utf-8')
        plt.close()
    
    # Sort tasks by priority for display (High > Medium > Low > Unclassified)
    priority_order = {'High': 0, 'Medium': 1, 'Low': 2, 'Unclassified': 3}
    sorted_tasks = sorted(tasks, key=lambda t: priority_order.get(t.priority, 3))
    
    return render_template_string(HTML_TEMPLATE, sorted_tasks=sorted_tasks, img_base64=img_base64)

@app.route('/add_task', methods=['POST'])
def add_task():
    description = request.form['description']
    urgency = request.form['urgency']
    importance = request.form['importance']
    due_date = request.form['due_date']
    
    # Basic validation
    try:
        urgency = int(urgency)
        importance = int(importance)
        if not (1 <= urgency <= 10 and 1 <= importance <= 10):
            return "Invalid urgency or importance value. Must be between 1 and 10.", 400
        datetime.datetime.strptime(due_date, '%Y-%m-%d')  # Validate date format
    except ValueError:
        return "Invalid input values.", 400
    
    task = Task(description, urgency, importance, due_date)
    tasks.append(task)
    
    return index()

@app.route('/delete/<string:unique_id>', methods=['POST'])
def delete(unique_id):
    global tasks
    tasks = [t for t in tasks if t.unique_id != unique_id]
    return index()

if __name__ == '__main__':
    app.run(debug=True)