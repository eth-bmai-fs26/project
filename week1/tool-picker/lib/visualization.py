"""
Visualization and helper functions for the Tool Router demo.
Used by both the notebook and the Colab HTML widget.
"""

import re
import datetime
import numpy as np
import torch

# ─────────────────────────────────────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────────────────────────────────────

LABEL_NAMES = {0: "FLIGHT_BOOKER", 1: "HOTEL_BOOKER", 2: "CAR_RENTAL_BOOKER"}

TOOL_EMOJIS = {
    "FLIGHT_BOOKER": "✈️",
    "HOTEL_BOOKER": "🏨",
    "CAR_RENTAL_BOOKER": "🚗",
}

TOOL_COLORS = {
    "FLIGHT_BOOKER": "#3498db",      # Blue
    "HOTEL_BOOKER": "#e74c3c",       # Red
    "CAR_RENTAL_BOOKER": "#2ecc71",  # Green
}


# ─────────────────────────────────────────────────────────────────────────────
# Plan Parsing
# ─────────────────────────────────────────────────────────────────────────────

def parse_plan_file(file_path: str) -> dict:
    """Parse a .txt travel plan file into structured data.
    
    Expected format:
        Plan ID: P-XXXX
        Trip: Origin -> Destination, dates
        Plan Steps:
        1) Step description...
        2) Step description...
        ...
    
    Args:
        file_path: Path to the .txt plan file
        
    Returns:
        dict with keys: plan_id, trip_summary, steps
    """
    with open(file_path, 'r') as f:
        content = f.read()
    return parse_plan_text(content)


def parse_plan_text(content: str) -> dict:
    """Parse plan text content into structured data.
    
    Args:
        content: Raw text content of the plan
        
    Returns:
        dict with keys: plan_id, trip_summary, steps
    """
    # Extract plan ID
    plan_id_match = re.search(r'Plan ID:\s*(\S+)', content)
    plan_id = plan_id_match.group(1) if plan_id_match else "UNKNOWN"
    
    # Extract trip summary
    trip_match = re.search(r'Trip:\s*(.+)', content)
    trip_summary = trip_match.group(1).strip() if trip_match else ""
    
    # Extract plan steps (numbered lines)
    steps = []
    step_pattern = re.compile(r'^\s*(\d+)\)\s*(.+)$', re.MULTILINE)
    for match in step_pattern.finditer(content):
        step_index = int(match.group(1))
        step_text = match.group(2).strip()
        steps.append({
            "step_index": step_index,
            "step_id": f"{plan_id}-S{step_index}",
            "step_text": step_text,
        })
    
    return {
        "plan_id": plan_id,
        "trip_summary": trip_summary,
        "steps": steps,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Routing
# ─────────────────────────────────────────────────────────────────────────────

def route_plan_steps(plan: dict, model, embedder, device) -> list:
    """Embed plan steps and route them through the trained MLP.
    
    Args:
        plan: Parsed plan dict with 'steps' key
        model: Trained PyTorch classifier
        embedder: SentenceTransformer model
        device: torch device (cpu/cuda)
        
    Returns:
        List of dicts with routing results for each step
    """
    if not plan["steps"]:
        return []
    
    # Extract step texts
    step_texts = [s["step_text"] for s in plan["steps"]]
    
    # Generate embeddings
    embeddings = embedder.encode(step_texts)
    tensor = torch.tensor(embeddings, dtype=torch.float32).to(device)
    
    # Run through classifier
    model.eval()
    with torch.no_grad():
        logits = model(tensor)
        probs = torch.softmax(logits, dim=1)
        preds = logits.argmax(dim=1).cpu().numpy()
        confs = probs.max(dim=1).values.cpu().numpy()
    
    # Build results
    results = []
    for step, pred, conf in zip(plan["steps"], preds, confs):
        tool = LABEL_NAMES[int(pred)]
        results.append({
            "step_index": step["step_index"],
            "step_id": step["step_id"],
            "step_text": step["step_text"],
            "tool": tool,
            "emoji": TOOL_EMOJIS[tool],
            "confidence": round(float(conf) * 100, 1),
        })
    
    return results


# ─────────────────────────────────────────────────────────────────────────────
# Display Functions
# ─────────────────────────────────────────────────────────────────────────────

def display_routing_table(results: list, plan: dict):
    """Display routing results as a styled ASCII table.
    
    Args:
        results: List of routing result dicts
        plan: Parsed plan dict
    """
    print(f"📋 Plan: {plan['plan_id']}")
    print(f"🗺️  Trip: {plan['trip_summary']}")
    print(f"\n{'─' * 80}")
    print(f"{'Step':<6} {'Description':<45} {'Routed To':<20} {'Conf.':<8}")
    print(f"{'─' * 80}")
    
    for r in results:
        step_text = r["step_text"][:42] + "..." if len(r["step_text"]) > 45 else r["step_text"]
        tool_display = f"{r['emoji']} {r['tool']}"
        print(f"{r['step_index']:<6} {step_text:<45} {tool_display:<20} {r['confidence']:>5.1f}%")
    
    print(f"{'─' * 80}\n")


def visualize_routing_bipartite(results: list, plan: dict):
    """Visualize tool routing as a bipartite graph: plan steps → tools.
    
    Args:
        results: List of routing result dicts
        plan: Parsed plan dict
    """
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    
    fig, ax = plt.subplots(figsize=(12, max(6, len(results) * 0.8)))
    ax.set_xlim(-0.5, 3.5)
    ax.set_ylim(-0.5, max(len(results), 3) + 0.5)
    ax.axis('off')
    ax.set_title(f"🧭 Tool Routing: {plan['plan_id']}\n{plan['trip_summary']}", 
                 fontsize=14, fontweight='bold', pad=20)
    
    # Tool positions (right side)
    tools = ["FLIGHT_BOOKER", "HOTEL_BOOKER", "CAR_RENTAL_BOOKER"]
    tool_y = {tool: i * (len(results) / 2.5) + 0.5 for i, tool in enumerate(tools)}
    
    # Draw tool nodes (right side)
    for tool in tools:
        y = tool_y[tool]
        color = TOOL_COLORS[tool]
        emoji = TOOL_EMOJIS[tool]
        
        # Tool box
        rect = mpatches.FancyBboxPatch((2.5, y - 0.25), 0.8, 0.5,
                                        boxstyle="round,pad=0.05",
                                        facecolor=color, edgecolor='black',
                                        linewidth=2, alpha=0.8)
        ax.add_patch(rect)
        ax.text(2.9, y, f"{emoji}", fontsize=16, ha='center', va='center')
        ax.text(2.9, y - 0.45, tool.replace('_', '\n'), fontsize=8, 
                ha='center', va='top', fontweight='bold')
    
    # Draw step nodes (left side) and edges
    for i, r in enumerate(results):
        y_step = len(results) - i - 0.5
        tool = r["tool"]
        y_tool = tool_y[tool]
        color = TOOL_COLORS[tool]
        
        # Step box
        rect = mpatches.FancyBboxPatch((0, y_step - 0.2), 1.8, 0.4,
                                        boxstyle="round,pad=0.02",
                                        facecolor='#ecf0f1', edgecolor='#34495e',
                                        linewidth=1.5)
        ax.add_patch(rect)
        
        # Step text (truncated)
        step_text = r["step_text"][:35] + "..." if len(r["step_text"]) > 35 else r["step_text"]
        ax.text(0.05, y_step, f"{r['step_index']}. {step_text}", 
                fontsize=9, va='center', ha='left')
        
        # Edge (curved arrow)
        ax.annotate("", xy=(2.5, y_tool), xytext=(1.8, y_step),
                    arrowprops=dict(arrowstyle="->", color=color, 
                                   connectionstyle="arc3,rad=0.1",
                                   linewidth=2, alpha=0.7))
    
    plt.tight_layout()
    plt.show()


# ─────────────────────────────────────────────────────────────────────────────
# Emulated Tool Execution
# ─────────────────────────────────────────────────────────────────────────────

def emulate_tool_execution(results: list, plan: dict):
    """Print emulated tool execution logs (no actual API calls).
    
    Args:
        results: List of routing result dicts
        plan: Parsed plan dict
    """
    print(f"{'═' * 80}")
    print(f"🚀 TOOL EXECUTION LOG — Plan {plan['plan_id']}")
    print(f"📅 Timestamp: {datetime.datetime.now().isoformat()}")
    print(f"{'═' * 80}\n")
    
    for r in results:
        tool = r["tool"]
        emoji = r["emoji"]
        step_text = r["step_text"]
        
        print(f"┌{'─' * 78}┐")
        print(f"│ {emoji} CALLING: {tool:<63} │")
        print(f"├{'─' * 78}┤")
        print(f"│ Step {r['step_index']}: {step_text[:65]:<65} │")
        if len(step_text) > 65:
            print(f"│         {step_text[65:130]:<69} │")
        print(f"│ Confidence: {r['confidence']:.1f}%{' ' * 61} │")
        print(f"├{'─' * 78}┤")
        
        # Emulated API call based on tool type
        if tool == "FLIGHT_BOOKER":
            print(f"│ 📡 [EMULATED] POST /api/flights/search                                      │")
            print(f"│    Request: {{'origin': 'ZRH', 'destination': 'BCN', ...}}                   │")
            print(f"│ ✅ [EMULATED] Response: 200 OK — Found 12 flight options                    │")
        elif tool == "HOTEL_BOOKER":
            print(f"│ 📡 [EMULATED] POST /api/hotels/search                                       │")
            print(f"│    Request: {{'location': 'Barcelona', 'nights': 4, ...}}                    │")
            print(f"│ ✅ [EMULATED] Response: 200 OK — Found 8 hotel options                      │")
        elif tool == "CAR_RENTAL_BOOKER":
            print(f"│ 📡 [EMULATED] POST /api/cars/search                                         │")
            print(f"│    Request: {{'pickup': 'BCN Airport', 'days': 4, ...}}                      │")
            print(f"│ ✅ [EMULATED] Response: 200 OK — Found 5 rental options                     │")
        
        print(f"└{'─' * 78}┘\n")
    
    print(f"{'═' * 80}")
    print(f"✅ ALL TOOL CALLS COMPLETED — {len(results)} steps routed successfully")
    print(f"{'═' * 80}")


# ─────────────────────────────────────────────────────────────────────────────
# Full Pipeline
# ─────────────────────────────────────────────────────────────────────────────

def route_plan_from_file(file_path: str, model, embedder, device, show_logs: bool = True):
    """Complete pipeline: parse .txt file → embed → route → visualize.
    
    Args:
        file_path: Path to a .txt travel plan file
        model: Trained PyTorch classifier
        embedder: SentenceTransformer model
        device: torch device
        show_logs: Whether to print emulated tool execution logs
        
    Returns:
        List of routing results, or None if parsing failed
    """
    # 1. Parse
    print("📄 Parsing plan file...")
    plan = parse_plan_file(file_path)
    
    if not plan["steps"]:
        print("❌ No steps found in plan file. Check the format.")
        return None
    
    # 2. Route
    print(f"🔄 Routing {len(plan['steps'])} steps through classifier...\n")
    results = route_plan_steps(plan, model, embedder, device)
    
    # 3. Display table
    display_routing_table(results, plan)
    
    # 4. Visualize
    visualize_routing_bipartite(results, plan)
    
    # 5. Optional: emulated logs
    if show_logs:
        emulate_tool_execution(results, plan)
    
    return results


def route_plan_from_text(content: str, model, embedder, device, show_logs: bool = True):
    """Complete pipeline from raw text content.
    
    Args:
        content: Raw text content of the plan
        model: Trained PyTorch classifier
        embedder: SentenceTransformer model
        device: torch device
        show_logs: Whether to print emulated tool execution logs
        
    Returns:
        List of routing results, or None if parsing failed
    """
    # 1. Parse
    plan = parse_plan_text(content)
    
    if not plan["steps"]:
        return None, None
    
    # 2. Route
    results = route_plan_steps(plan, model, embedder, device)
    
    return results, plan
