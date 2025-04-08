"""
Script to generate visualizations from performance metrics.
"""

import json
import os
import datetime
import matplotlib.pyplot as plt
import pandas as pd
from pathlib import Path

def load_metrics():
    """Load all metrics files and combine into a DataFrame."""
    metrics_dir = Path("learning/performance")
    metrics_files = list(metrics_dir.glob("metrics_*.json"))
    
    if not metrics_files:
        print("No metrics files found")
        return None
    
    data = []
    for file in metrics_files:
        with open(file, "r") as f:
            metrics = json.load(f)
            
            # Extract timestamp and metrics
            timestamp = datetime.datetime.fromisoformat(metrics["timestamp"])
            metrics_data = metrics["metrics"]
            metrics_data["timestamp"] = timestamp
            
            data.append(metrics_data)
    
    # Convert to DataFrame
    df = pd.DataFrame(data)
    df = df.sort_values("timestamp")
    
    return df

def generate_accuracy_plot(df):
    """Generate plot for accuracy metrics."""
    if df is None or df.empty:
        print("No data available for accuracy plot")
        return
        
    plt.figure(figsize=(12, 6))
    
    plt.plot(df["timestamp"], df["ui_detection_accuracy"], label="UI Detection Accuracy")
    plt.plot(df["timestamp"], df["llm_response_quality"], label="LLM Response Quality")
    plt.plot(df["timestamp"], df["task_completion_rate"], label="Task Completion Rate")
    
    plt.title("AI.NEXUS Accuracy Metrics Over Time")
    plt.xlabel("Date")
    plt.ylabel("Accuracy (0-1)")
    plt.legend()
    plt.grid(True)
    
    # Ensure directory exists
    vis_dir = Path("learning/performance/visualizations")
    vis_dir.mkdir(parents=True, exist_ok=True)
    
    plt.savefig(vis_dir / "accuracy_metrics.png")
    plt.close()

def generate_performance_plot(df):
    """Generate plot for performance metrics."""
    if df is None or df.empty:
        print("No data available for performance plot")
        return
        
    plt.figure(figsize=(12, 6))
    
    plt.plot(df["timestamp"], df["response_time"], label="Response Time (s)")
    
    plt.title("AI.NEXUS Performance Metrics Over Time")
    plt.xlabel("Date")
    plt.ylabel("Response Time (seconds)")
    plt.legend()
    plt.grid(True)
    
    # Ensure directory exists
    vis_dir = Path("learning/performance/visualizations")
    vis_dir.mkdir(parents=True, exist_ok=True)
    
    plt.savefig(vis_dir / "performance_metrics.png")
    plt.close()

def generate_learning_plot(df):
    """Generate plot for learning metrics."""
    if df is None or df.empty:
        print("No data available for learning plot")
        return
        
    plt.figure(figsize=(12, 6))
    
    plt.plot(df["timestamp"], df["learning_rate"], label="Learning Rate")
    
    plt.title("AI.NEXUS Learning Rate Over Time")
    plt.xlabel("Date")
    plt.ylabel("Learning Rate")
    plt.legend()
    plt.grid(True)
    
    # Ensure directory exists
    vis_dir = Path("learning/performance/visualizations")
    vis_dir.mkdir(parents=True, exist_ok=True)
    
    plt.savefig(vis_dir / "learning_metrics.png")
    plt.close()

def generate_summary_report(df):
    """Generate a summary report of metrics."""
    if df is None or df.empty:
        print("No data available for summary report")
        return
    
    # Calculate improvement from first to last measurement
    first = df.iloc[0]
    last = df.iloc[-1]
    
    improvements = {
        "ui_detection_accuracy": last["ui_detection_accuracy"] - first["ui_detection_accuracy"],
        "llm_response_quality": last["llm_response_quality"] - first["llm_response_quality"],
        "task_completion_rate": last["task_completion_rate"] - first["task_completion_rate"],
        "response_time": first["response_time"] - last["response_time"],  # Lower is better
        "learning_rate": last["learning_rate"] - first["learning_rate"]
    }
    
    # Create report
    report = f"""# AI.NEXUS Performance Summary

## Date Range
From: {first['timestamp'].strftime('%Y-%m-%d')}
To: {last['timestamp'].strftime('%Y-%m-%d')}

## Improvements

- UI Detection Accuracy: {improvements['ui_detection_accuracy']:.2%}
- LLM Response Quality: {improvements['llm_response_quality']:.2%}
- Task Completion Rate: {improvements['task_completion_rate']:.2%}
- Response Time: {improvements['response_time']:.2f}s faster
- Learning Rate: {improvements['learning_rate']:.2%}
"""
    
    # Ensure directory exists
    vis_dir = Path("learning/performance/visualizations")
    vis_dir.mkdir(parents=True, exist_ok=True)
    
    with open(vis_dir / "summary_report.md", "w") as f:
        f.write(report)
    
    print(f"Summary report saved to {vis_dir / 'summary_report.md'}")

def main():
    """Generate all visualizations."""
    df = load_metrics()
    
    if df is None:
        print("No metrics data available. Run collect_metrics.py first.")
        return
    
    generate_accuracy_plot(df)
    generate_performance_plot(df)
    generate_learning_plot(df)
    generate_summary_report(df)
    
    print("Visualizations generated successfully!")

if __name__ == "__main__":
    main()
