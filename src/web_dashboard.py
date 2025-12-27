"""
FastAPI Web Dashboard for Predictive Maintenance System
Modern web interface for non-technical users
"""

from fastapi import FastAPI, File, UploadFile, BackgroundTasks, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse, JSONResponse, FileResponse
from fastapi.staticfiles import StaticFiles
import uvicorn
import os
import subprocess
import sys
import json
import pandas as pd
from datetime import datetime
import asyncio
import numpy as np
import joblib

# Import centralized configuration
try:
    import config
    print("✓ Configuration loaded")
except ImportError:
    print("⚠️  No config.py found, using defaults")
    config = None

app = FastAPI(title="Predictive Maintenance System")

# Global state
system_state = {
    "models_trained": os.path.exists("models/xgboost_model.pkl"),
    "training_in_progress": False,
    "demo_running": False,
    "last_prediction": None
}

@app.get("/", response_class=HTMLResponse)
async def home():
    """Serve main dashboard"""
    html_content = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Predictive Maintenance System</title>
    <style>
        * {
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }
        
        body {
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            min-height: 100vh;
            padding: 20px;
        }
        
        .container {
            max-width: 1200px;
            margin: 0 auto;
        }
        
        .header {
            text-align: center;
            color: white;
            margin-bottom: 30px;
        }
        
        .header h1 {
            font-size: 2.5em;
            margin-bottom: 10px;
            text-shadow: 2px 2px 4px rgba(0,0,0,0.2);
        }
        
        .header p {
            font-size: 1.2em;
            opacity: 0.9;
        }
        
        .card {
            background: white;
            border-radius: 15px;
            padding: 25px;
            margin-bottom: 20px;
            box-shadow: 0 10px 30px rgba(0,0,0,0.2);
        }
        
        .status-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
            gap: 20px;
            margin-bottom: 30px;
        }
        
        .status-card {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 20px;
            border-radius: 10px;
            text-align: center;
        }
        
        .status-card h3 {
            margin-bottom: 10px;
            font-size: 1em;
            opacity: 0.9;
        }
        
        .status-card .value {
            font-size: 2em;
            font-weight: bold;
        }
        
        .btn {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            border: none;
            padding: 15px 30px;
            border-radius: 8px;
            font-size: 1.1em;
            cursor: pointer;
            transition: transform 0.2s, box-shadow 0.2s;
            margin: 10px;
        }
        
        .btn:hover {
            transform: translateY(-2px);
            box-shadow: 0 5px 15px rgba(0,0,0,0.3);
        }
        
        .btn:disabled {
            opacity: 0.5;
            cursor: not-allowed;
        }
        
        .btn-success {
            background: linear-gradient(135deg, #11998e 0%, #38ef7d 100%);
        }
        
        .btn-danger {
            background: linear-gradient(135deg, #eb3349 0%, #f45c43 100%);
        }
        
        .btn-info {
            background: linear-gradient(135deg, #4facfe 0%, #00f2fe 100%);
        }
        
        .actions {
            text-align: center;
            margin: 30px 0;
        }
        
        .log {
            background: #f8f9fa;
            border: 1px solid #dee2e6;
            border-radius: 8px;
            padding: 15px;
            max-height: 300px;
            overflow-y: auto;
            font-family: 'Courier New', monospace;
            font-size: 0.9em;
        }
        
        .log-entry {
            padding: 5px 0;
            border-bottom: 1px solid #e9ecef;
        }
        
        .log-entry:last-child {
            border-bottom: none;
        }
        
        .success { color: #28a745; }
        .error { color: #dc3545; }
        .info { color: #17a2b8; }
        
        .spinner {
            border: 4px solid #f3f3f3;
            border-top: 4px solid #667eea;
            border-radius: 50%;
            width: 40px;
            height: 40px;
            animation: spin 1s linear infinite;
            margin: 20px auto;
        }
        
        @keyframes spin {
            0% { transform: rotate(0deg); }
            100% { transform: rotate(360deg); }
        }
        
        .hidden {
            display: none;
        }
        
        #results-table {
            width: 100%;
            border-collapse: collapse;
            margin-top: 20px;
        }
        
        #results-table th,
        #results-table td {
            padding: 12px;
            text-align: left;
            border-bottom: 1px solid #dee2e6;
        }
        
        #results-table th {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
        }
        
        #results-table tr:hover {
            background: #f8f9fa;
        }
        
        .risk-high { color: #dc3545; font-weight: bold; }
        .risk-medium { color: #ffc107; font-weight: bold; }
        .risk-low { color: #28a745; }
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>🏭 Predictive Maintenance System</h1>
            <p>AI-Powered Machine Failure Prediction</p>
        </div>
        
        <div class="status-grid">
            <div class="status-card">
                <h3>System Status</h3>
                <div class="value" id="system-status">Loading...</div>
            </div>
            <div class="status-card">
                <h3>Models Trained</h3>
                <div class="value" id="models-status">Loading...</div>
            </div>
            <div class="status-card">
                <h3>Last Update</h3>
                <div class="value" id="last-update" style="font-size: 1.2em;">Never</div>
            </div>
        </div>
        
        <div class="card">
            <h2 style="margin-bottom: 20px;">🚀 Quick Actions</h2>
            <div class="actions">
                <button class="btn btn-success" onclick="trainModels()" id="train-btn">
                    Train Models
                </button>
                <button class="btn btn-info" onclick="runDemo()" id="demo-btn">
                    Run Real-Time Demo
                </button>
                <button class="btn" onclick="viewResults()" id="results-btn">
                    View Results
                </button>
                <button class="btn btn-danger" onclick="openModelPerformance()">
                    📊 Model Performance
                </button>
                <button class="btn btn-danger" onclick="openProductionDashboard()">
                    🏭 Production Dashboard
                </button>
                <button class="btn btn-info" onclick="openRealtimeDashboard()">
                    ⚡ Real-Time Monitor
                </button>
            </div>
        </div>
        
        <div class="card">
            <h2 style="margin-bottom: 15px;">📊 Activity Log</h2>
            <div class="log" id="activity-log">
                <div class="log-entry info">System ready. Waiting for actions...</div>
            </div>
        </div>
        
        <div class="card hidden" id="results-card">
            <h2 style="margin-bottom: 15px;">📈 Prediction Results</h2>
            <div id="results-summary"></div>
            <table id="results-table"></table>
        </div>
    </div>
    
    <script>
        let logEntries = [];
        
        function addLog(message, type = 'info') {
            const log = document.getElementById('activity-log');
            const entry = document.createElement('div');
            entry.className = `log-entry ${type}`;
            entry.textContent = `[${new Date().toLocaleTimeString()}] ${message}`;
            logEntries.push(entry);
            log.innerHTML = '';
            logEntries.slice(-20).forEach(e => log.appendChild(e));
            log.scrollTop = log.scrollHeight;
        }
        
        async function updateStatus() {
            try {
                const response = await fetch('/api/status');
                const data = await response.json();
                
                document.getElementById('system-status').textContent = '✓ Ready';
                document.getElementById('models-status').textContent = 
                    data.models_trained ? '✓ Yes' : '✗ No';
                document.getElementById('last-update').textContent = 
                    new Date().toLocaleTimeString();
                    
                document.getElementById('train-btn').disabled = data.training_in_progress;
                document.getElementById('demo-btn').disabled = !data.models_trained || data.demo_running;
                document.getElementById('results-btn').disabled = !data.models_trained;
                
            } catch (error) {
                addLog('Error updating status', 'error');
            }
        }
        
        async function trainModels() {
            addLog('Starting model training...', 'info');
            document.getElementById('train-btn').disabled = true;
            
            try {
                const response = await fetch('/api/train', { method: 'POST' });
                const data = await response.json();
                
                if (data.status === 'started') {
                    addLog('Training started. This will take 1-2 minutes...', 'info');
                    
                    // Poll for completion
                    const interval = setInterval(async () => {
                        const status = await fetch('/api/status').then(r => r.json());
                        if (!status.training_in_progress) {
                            clearInterval(interval);
                            addLog('✓ Model training complete!', 'success');
                            updateStatus();
                        }
                    }, 3000);
                }
            } catch (error) {
                addLog('✗ Error training models: ' + error, 'error');
                document.getElementById('train-btn').disabled = false;
            }
        }
        
        async function runDemo() {
            addLog('Starting real-time demo...', 'info');
            document.getElementById('demo-btn').disabled = true;
            
            try {
                const response = await fetch('/api/demo', { method: 'POST' });
                const data = await response.json();
                
                if (data.status === 'started') {
                    addLog('Real-time demo running for 30 seconds...', 'info');
                    
                    setTimeout(async () => {
                        addLog('✓ Demo complete!', 'success');
                        updateStatus();
                    }, 35000);
                }
            } catch (error) {
                addLog('✗ Error running demo: ' + error, 'error');
                document.getElementById('demo-btn').disabled = false;
            }
        }
        
        async function viewResults() {
            addLog('Loading results...', 'info');
            
            try {
                const response = await fetch('/api/results');
                const data = await response.json();
                
                if (data.error) {
                    addLog('⚠️ ' + data.error, 'error');
                    addLog('Please train models first (Train Models button)', 'info');
                    return;
                }
                
                if (data.results) {
                    const card = document.getElementById('results-card');
                    card.classList.remove('hidden');
                    
                    // Summary
                    const summary = document.getElementById('results-summary');
                    summary.innerHTML = `
                        <p><strong>Total Machines:</strong> ${data.total}</p>
                        <p class="risk-high"><strong>High Risk:</strong> ${data.high_risk}</p>
                        <p class="risk-medium"><strong>Medium Risk:</strong> ${data.medium_risk}</p>
                        <p class="risk-low"><strong>Low Risk:</strong> ${data.low_risk}</p>
                    `;
                    
                    // Table
                    const table = document.getElementById('results-table');
                    
                    if (data.results.length === 0) {
                        table.innerHTML = '<tr><td colspan="3">No results available yet. Please run training first.</td></tr>';
                        addLog('No results found', 'info');
                        return;
                    }
                    
                    let html = '<thead><tr><th>Machine ID</th><th>Failure Probability</th><th>Risk Level</th></tr></thead><tbody>';
                    
                    const highRiskMachines = data.results.filter(r => r.risk === 'High').slice(0, 10);
                    
                    if (highRiskMachines.length === 0) {
                        html += '<tr><td colspan="3">No high-risk machines found. All machines are operating normally! ✓</td></tr>';
                    } else {
                        highRiskMachines.forEach(r => {
                            html += `<tr>
                                <td>${r.machine_id}</td>
                                <td>${(r.probability * 100).toFixed(1)}%</td>
                                <td class="risk-${r.risk.toLowerCase()}">${r.risk}</td>
                            </tr>`;
                        });
                    }
                    
                    html += '</tbody>';
                    table.innerHTML = html;
                    
                    addLog('✓ Results loaded', 'success');
                    card.scrollIntoView({ behavior: 'smooth' });
                }
            } catch (error) {
                addLog('✗ Error loading results: ' + error, 'error');
            }
        }
        
        function openModelPerformance() {
            addLog('Opening Model Performance dashboard...', 'info');
            
            fetch('/model_performance').then(response => {
                if (response.ok) {
                    window.open('/model_performance', '_blank');
                    addLog('✓ Model Performance dashboard opened', 'success');
                } else {
                    addLog('⚠️ Model Performance dashboard not found. Please train models first.', 'error');
                }
            }).catch(error => {
                addLog('✗ Error opening dashboard: ' + error, 'error');
            });
        }
        
        function openProductionDashboard() {
            addLog('Opening Production Monitoring dashboard...', 'info');
            
            fetch('/production_dashboard').then(response => {
                if (response.ok) {
                    window.open('/production_dashboard', '_blank');
                    addLog('✓ Production dashboard opened', 'success');
                } else {
                    addLog('⚠️ Production dashboard not found. Please train models first.', 'error');
                }
            }).catch(error => {
                addLog('✗ Error opening dashboard: ' + error, 'error');
            });
        }
        
        function openRealtimeDashboard() {
            addLog('Opening Real-Time Monitor...', 'info');
            
            fetch('/realtime_dashboard').then(response => {
                if (response.ok) {
                    window.open('/realtime_dashboard', '_blank');
                    addLog('✓ Real-Time Monitor opened', 'success');
                } else {
                    addLog('⚠️ Real-Time dashboard not found.', 'error');
                }
            }).catch(error => {
                addLog('✗ Error opening dashboard: ' + error, 'error');
            });
        }
        
        // Initial status update
        updateStatus();
        setInterval(updateStatus, 5000);
    </script>
</body>
</html>
    """
    return HTMLResponse(content=html_content)

@app.get("/api/status")
async def get_status():
    """Get system status"""
    system_state["models_trained"] = os.path.exists("models/xgboost_model.pkl")
    return JSONResponse(content=system_state)

@app.post("/api/train")
async def train_models(background_tasks: BackgroundTasks):
    """Train models in background"""
    if system_state["training_in_progress"]:
        return JSONResponse(content={"status": "already_running"})
    
    def run_training():
        system_state["training_in_progress"] = True
        try:
            subprocess.run([sys.executable, "predictive_maintenance.py"])
            system_state["models_trained"] = True
        finally:
            system_state["training_in_progress"] = False
    
    background_tasks.add_task(run_training)
    return JSONResponse(content={"status": "started"})

@app.post("/api/demo")
async def run_demo(background_tasks: BackgroundTasks):
    """Run real-time demo"""
    if system_state["demo_running"]:
        return JSONResponse(content={"status": "already_running"})
    
    def run_demo_task():
        system_state["demo_running"] = True
        try:
            # Run with default 30-second duration
            process = subprocess.Popen(
                [sys.executable, "realtime_predictor.py"],
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True
            )
            process.communicate(input="1\n", timeout=40)
        finally:
            system_state["demo_running"] = False
    
    background_tasks.add_task(run_demo_task)
    return JSONResponse(content={"status": "started"})

@app.get("/api/results")
async def get_results():
    """Get prediction results"""
    results_file = "results/prediction_results.csv"
    
    if not os.path.exists(results_file):
        return JSONResponse(content={"error": "No results found. Please train models first."})
    
    try:
        df = pd.read_csv(results_file)
        
        results = []
        for _, row in df.iterrows():
            results.append({
                "machine_id": row['Machine_ID'],
                "probability": float(row['Failure_Probability']),
                "risk": row['Risk_Level']
            })
        
        return JSONResponse(content={
            "results": results,
            "total": len(df),
            "high_risk": int((df['Risk_Level'] == 'High').sum()),
            "medium_risk": int((df['Risk_Level'] == 'Medium').sum()),
            "low_risk": int((df['Risk_Level'] == 'Low').sum())
        })
    except Exception as e:
        return JSONResponse(content={"error": f"Error reading results: {str(e)}"})

@app.get("/dashboard")
@app.get("/dashboard.html")
async def serve_dashboard():
    """Serve the interactive dashboard HTML file"""
    dashboard_path = os.path.join("dashboards", "dashboard.html")
    
    if not os.path.exists(dashboard_path):
        return HTMLResponse(
            content="<html><body><h1>Dashboard not found</h1><p>Please run model training first.</p></body></html>",
            status_code=404
        )
    
    return FileResponse(dashboard_path, media_type="text/html")

@app.get("/model_performance")
@app.get("/model_performance.html")
async def serve_model_performance():
    """Serve the model performance dashboard"""
    dashboard_path = os.path.join("dashboards", "model_performance.html")
    
    if not os.path.exists(dashboard_path):
        return HTMLResponse(
            content="<html><body><h1>Model Performance Dashboard not found</h1><p>Please run model training first.</p></body></html>",
            status_code=404
        )
    
    return FileResponse(dashboard_path, media_type="text/html")

@app.get("/production_dashboard")
@app.get("/production_dashboard.html")
async def serve_production_dashboard():
    """Serve the production monitoring dashboard"""
    dashboard_path = os.path.join("dashboards", "production_dashboard.html")
    
    if not os.path.exists(dashboard_path):
        return HTMLResponse(
            content="<html><body><h1>Production Dashboard not found</h1><p>Please run model training first.</p></body></html>",
            status_code=404
        )
    
    return FileResponse(dashboard_path, media_type="text/html")

@app.get("/realtime_dashboard")
@app.get("/realtime_dashboard.html")
async def serve_realtime_dashboard():
    """Serve the real-time animated dashboard"""
    dashboard_path = os.path.join("dashboards", "realtime_dashboard.html")
    
    if not os.path.exists(dashboard_path):
        return HTMLResponse(
            content="<html><body><h1>Real-Time Dashboard not found</h1><p>Dashboard file is missing.</p></body></html>",
            status_code=404
        )
    
    return FileResponse(dashboard_path, media_type="text/html")

# =============================================================================
# WEBSOCKET ENDPOINT FOR REAL-TIME PREDICTIONS
# =============================================================================
@app.websocket("/ws/realtime")
async def websocket_realtime(websocket: WebSocket):
    """Stream real-time predictions using trained ML models"""
    await websocket.accept()
    
    try:
        # Get paths from config
        model_path = config.PATHS["xgb_model"] if config else "models/xgboost_model.pkl"
        scaler_path = config.PATHS["scaler"] if config else "models/scaler.pkl"
        features_path = config.PATHS["features"] if config else "models/selected_features.txt"
        
        # Check if models exist
        if not all([os.path.exists(p) for p in [model_path, scaler_path, features_path]]):
            await websocket.send_json({
                "error": "Models not found. Please train models first.",
                "status": "error"
            })
            await websocket.close()
            return
        
        # Load models
        model = joblib.load(model_path)
        scaler = joblib.load(scaler_path)
        
        with open(features_path) as f:
            features = [line.strip() for line in f]
        
        print("✓ Models loaded for real-time streaming")
        
        # Get config values
        update_interval = config.WEBSOCKET["update_interval"] if config else 1.0
        degradation_rate = config.SIMULATION["degradation_rate"] if config else 0.02
        
        time_step = 0
        degradation = 0.0
        
        while True:
            # Generate realistic sensor data with degradation
            data = {
                'vibration': 2.4 + np.random.normal(0, 0.2) + (degradation * 5),
                'temperature': 69.5 + np.random.normal(0, 2) + (degradation * 25),
                'pressure': 5.39 + np.random.normal(0, 0.1) - (degradation * 1),
                'runtime_hours': time_step * 0.5,
                'voltage': 220 + np.random.normal(0, 2),
                'current': 15 + np.random.normal(0, 1) + (degradation * 5),
                'acoustic_emission': 66 + np.random.normal(0, 2) + (degradation * 10),
                'rotation_speed': 1495 + np.random.normal(0, 5) - (degradation * 20),
                'torque': 147 + np.random.normal(0, 3) - (degradation * 10),
                'power_consumption': 22.5 + np.random.normal(0, 1) + (degradation * 5),
                'vibration_rolling_avg': 2.4 + (degradation * 5),
                'temperature_rolling_avg': 69.5 + (degradation * 25),
                'pressure_rolling_avg': 5.39 - (degradation * 1),
                'vibration_rolling_std': 0.1 + (degradation * 0.5),
                'current_rolling_std': 0.5 + (degradation * 1.5),
                'power_efficiency': 0,
                'temp_vib_interaction': 0
            }
            
            data['power_efficiency'] = data['power_consumption'] / (data['torque'] + 1)
            data['temp_vib_interaction'] = data['temperature'] * data['vibration']
            
            # Prepare features in correct order
            X = np.array([data.get(f, 0) for f in features]).reshape(1, -1)
            X_scaled = scaler.transform(X)
            
            # Make prediction
            prediction = int(model.predict(X_scaled)[0])
            probability = float(model.predict_proba(X_scaled)[0][1])
            
            # Send to client
            await websocket.send_json({
                'time': time_step,
                'vibration': float(data['vibration']),
                'temperature': float(data['temperature']),
                'pressure': float(data['pressure']),
                'probability': probability,
                'prediction': prediction,
                'degradation': degradation,
                'timestamp': datetime.now().isoformat()
            })
            
            time_step += 1
            degradation = min(1.0, degradation + degradation_rate)
            
            await asyncio.sleep(update_interval)
            
    except WebSocketDisconnect:
        print("Client disconnected from real-time stream")
    except Exception as e:
        print(f"WebSocket error: {e}")
        try:
            await websocket.send_json({"error": str(e), "status": "error"})
        except:
            pass
    finally:
        try:
            await websocket.close()
        except:
            pass

if __name__ == "__main__":
    print("🌐 Starting Predictive Maintenance Web Dashboard...")
    print("📍 Server will be available at: http://localhost:8000")
    print("⚠️  Press Ctrl+C to stop the server\n")
    
    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="info")
