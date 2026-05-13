import React, { useCallback, useEffect, useMemo, useState } from "react";
import { createRoot } from "react-dom/client";
import ReactFlow, { Background, Controls, Handle, Position, addEdge, useEdgesState, useNodesState } from "reactflow";
import axios from "axios";
import { Play, Save, Trash2, Upload, Plus, ChevronLeft, ChevronRight } from "lucide-react";
import "reactflow/dist/style.css";
import "./style.css";

const arcColors = ["#000000", "#0074D9", "#FF4136", "#2ECC40", "#FFDC00", "#AAAAAA", "#F012BE", "#FF851B", "#7FDBFF", "#870C25"];
const ops = ["Constant", "Cast", "Identity", "Equal", "Greater", "Less", "Not", "And", "Add", "Sub", "Mul", "Div", "ReduceSum", "ArgMax", "Where"];
const saved = ["baseline-cast-equal", "argmax-mask-v2", "reduce-sum-probe"];
const TASK_COUNT = 400;
const TASK_PAGE_SIZE = 40;

function clampTask(value) {
  const parsed = Number(value || 1);
  if (!Number.isFinite(parsed)) return 1;
  return Math.min(TASK_COUNT, Math.max(1, Math.trunc(parsed)));
}

function taskIdFor(taskNumber) {
  return `task${String(taskNumber).padStart(3, "0")}`;
}

function OpNode({ data, selected }) {
  return (
    <div className={`node ${selected ? "nodeSelected" : ""}`}>
      <Handle type="target" position={Position.Left} />
      <div className="nodeType">{data.opType}</div>
      <div className="nodeId">{data.label}</div>
      <Handle type="source" position={Position.Right} />
    </div>
  );
}

const nodeTypes = { op: OpNode };

function gridShape(grid) {
  return `${grid?.length || 0}x${grid?.[0]?.length || 0}`;
}

function ArcGrid({ title, values }) {
  const cols = values?.[0]?.length || 1;
  return (
    <div className="gridBlock">
      <div className="gridTitle">{title} {gridShape(values)}</div>
      <div className="arcGrid" style={{ gridTemplateColumns: `repeat(${cols}, 1fr)` }}>
        {(values || []).flatMap((row, rowIndex) =>
          row.map((value, colIndex) => (
            <div key={`${rowIndex}-${colIndex}`} className="arcCell" style={{ backgroundColor: arcColors[value] || "#000000" }} />
          ))
        )}
      </div>
    </div>
  );
}

function SidebarButton({ children, onClick }) {
  return <button onClick={onClick} className="btn compact">{children}</button>;
}

function ValidationPanel({ validation, taskId }) {
  if (validation.state === "loading") {
    return (
      <div className="validationBlock loading">
        <div className="validationTask">{taskId}</div>
        <div>Validation: Running</div>
        <div>Status: Compiling and testing ONNX</div>
      </div>
    );
  }
  if (validation.state === "passed") {
    return (
      <div className="validationBlock passed">
        <div className="validationTask">{validation.taskId}</div>
        <div>Train: Passed</div>
        <div>Shape: Passed</div>
        <div>Colors: Passed</div>
        <div>Status: READY FOR SUBMISSION (Artifact pushed)</div>
      </div>
    );
  }
  if (validation.state === "failed") {
    return (
      <div className="validationBlock failed">
        <div className="validationTask">{validation.taskId}</div>
        <div>Validation: Failed</div>
        <div className="validationReason">{validation.reason}</div>
      </div>
    );
  }
  return (
    <div className="validationBlock idle">
      <div className="validationTask">{taskId}</div>
      <div>Validation: Not run</div>
      <div>Status: Awaiting Export ONNX</div>
    </div>
  );
}

function App() {
  const [task, setTask] = useState(10);
  const [projectName, setProjectName] = useState("neurogolf-task010");
  const [selectedId, setSelectedId] = useState(null);
  const [status, setStatus] = useState("Ready");
  const taskId = taskIdFor(task);
  const [currentTask, setCurrentTask] = useState(null);
  const [taskLoadState, setTaskLoadState] = useState("loading");
  const [taskLoadError, setTaskLoadError] = useState("");
  const example = currentTask?.train?.[0] || { input: [[0]], output: [[0]] };
  const trainingPairs = useMemo(() => currentTask?.train || [], [currentTask]);
  const stats = taskLoadState === "loaded"
    ? `${currentTask?.train?.length || 0} train, ${currentTask?.test?.length || 0} test, ${currentTask?.["arc-gen"]?.length || 0} extra`
    : taskLoadState === "failed" ? "task load failed" : "loading task";
  const [validation, setValidation] = useState({ state: "idle" });
  const taskPageStart = Math.floor((task - 1) / TASK_PAGE_SIZE) * TASK_PAGE_SIZE + 1;
  const taskPageEnd = Math.min(TASK_COUNT, taskPageStart + TASK_PAGE_SIZE - 1);
  const taskNumbers = Array.from({ length: taskPageEnd - taskPageStart + 1 }, (_, index) => taskPageStart + index);
  const initialNodes = useMemo(
    () => [
      { id: "input_1", type: "op", position: { x: 70, y: 80 }, data: { label: "input_1", opType: "Input", shape: "1,1,30,30" } },
      { id: "cast_11", type: "op", position: { x: 280, y: 80 }, data: { label: "cast_11", opType: "Cast", attrs: { to: "1" }, shape: "1,1,30,30" } },
      { id: "equal_3", type: "op", position: { x: 500, y: 80 }, data: { label: "equal_3", opType: "Equal", shape: "1,1,30,30" } },
      { id: "output_1", type: "op", position: { x: 720, y: 80 }, data: { label: "output_1", opType: "Output", shape: "1,1,30,30" } },
    ],
    []
  );
  const [nodes, setNodes, onNodesChange] = useNodesState(initialNodes);
  const [edges, setEdges, onEdgesChange] = useEdgesState([
    { id: "e1", source: "input_1", target: "cast_11" },
    { id: "e2", source: "cast_11", target: "equal_3" },
    { id: "e3", source: "equal_3", target: "output_1" },
  ]);
  const selectedNode = nodes.find((node) => node.id === selectedId);

  useEffect(() => {
    let cancelled = false;
    setTaskLoadState("loading");
    setTaskLoadError("");
    setValidation({ state: "idle" });
    fetch(`/tasks/${taskId}.json`, { cache: "no-cache" })
      .then((response) => {
        if (!response.ok) throw new Error(`${taskId}.json returned HTTP ${response.status}`);
        return response.json();
      })
      .then((data) => {
        if (cancelled) return;
        setCurrentTask(data);
        setTaskLoadState("loaded");
        setStatus(`${taskId} loaded`);
      })
      .catch((error) => {
        if (cancelled) return;
        setCurrentTask(null);
        setTaskLoadError(error.message);
        setTaskLoadState("failed");
        setStatus(`Task load failed: ${error.message}`);
      });
    return () => {
      cancelled = true;
    };
  }, [taskId]);

  const addNode = (opType) => {
    const id = `${opType.toLowerCase()}_${nodes.length + 1}`;
    setNodes((current) => [...current, { id, type: "op", position: { x: 140 + nodes.length * 35, y: 180 }, data: { label: id, opType, shape: "1,1,30,30", attrs: {} } }]);
  };

  const updateSelected = (patch) => {
    setNodes((current) => current.map((node) => (node.id === selectedId ? { ...node, data: { ...node.data, ...patch } } : node)));
  };

  const deleteSelected = () => {
    if (!selectedId) return;
    setNodes((current) => current.filter((node) => node.id !== selectedId));
    setEdges((current) => current.filter((edge) => edge.source !== selectedId && edge.target !== selectedId));
    setSelectedId(null);
  };

  const exportOnnx = async () => {
    if (taskLoadState !== "loaded" || trainingPairs.length === 0) {
      setValidation({ state: "failed", taskId, reason: taskLoadError || "Selected task has no loaded training pairs" });
      return;
    }
    setStatus("Exporting ONNX");
    setValidation({ state: "loading", taskId });
    try {
      const payload = { projectName, taskId, nodes, edges, trainingPairs };
      const { data } = await axios.post("/api/export", payload);
      setValidation({ state: "passed", taskId, artifact: data.artifact });
      setStatus(`Validation passed. Exported ${data.artifact}`);
    } catch (error) {
      const response = error.response?.data;
      const rawReason = response?.reason || response?.detail?.reason || response?.detail || error.message;
      const reason = typeof rawReason === "string" ? rawReason : JSON.stringify(rawReason);
      setValidation({ state: "failed", taskId, reason });
      setStatus("Validation failed");
    }
  };

  const onConnect = useCallback((params) => setEdges((eds) => addEdge(params, eds)), [setEdges]);

  return (
    <div className="appShell">
      <aside className="leftPanel">
        <div className="brand">
          <h1>NeuroGolf Lab</h1>
          <p>task{String(task).padStart(3, "0")} loaded</p>
        </div>
        <section>
          <h2>TASK</h2>
          <div className="taskPicker">
            <SidebarButton onClick={() => setTask((current) => Math.max(1, current - 1))}><ChevronLeft size={15} /></SidebarButton>
            <input value={task} min="1" max={TASK_COUNT} onChange={(e) => setTask(clampTask(e.target.value))} type="number" />
            <SidebarButton onClick={() => setTask((current) => Math.min(TASK_COUNT, current + 1))}><ChevronRight size={15} /></SidebarButton>
          </div>
          <div className="taskPage">TASKS {String(taskPageStart).padStart(3, "0")}-{String(taskPageEnd).padStart(3, "0")}</div>
          <div className="taskGrid">
            {taskNumbers.map((n) => <button key={n} onClick={() => setTask(n)} className={n === task ? "active" : ""}>{String(n).padStart(3, "0")}</button>)}
          </div>
        </section>
        <section>
          <h2>PROJECT</h2>
          <input value={projectName} onChange={(e) => setProjectName(e.target.value)} />
          <div className="twoCol"><button className="btn">New</button><button className="btn"><Save size={15} />Save</button></div>
          <select>{saved.map((item) => <option key={item}>{item}</option>)}</select>
          <button className="btn full">Load Selected</button>
        </section>
        <section>
          <h2>ADD NODE</h2>
          <div className="opGrid">{ops.map((op) => <button key={op} onClick={() => addNode(op)}><Plus size={14} />{op}</button>)}</div>
        </section>
      </aside>
      <main className="workspace">
        <nav className="topbar">
          <div><strong>Task {String(task).padStart(3, "0")}</strong><span>{stats}</span></div>
          <div className="actions">
            {["Train", "Test", "Extra 10", "Compile", "Run"].map((label) => <button key={label} className="btn">{label === "Run" && <Play size={15} />}{label}</button>)}
            <button className="btn primary" onClick={exportOnnx} disabled={validation.state === "loading" || taskLoadState !== "loaded"}><Upload size={15} />Export ONNX</button>
          </div>
        </nav>
        <section className="examples">
          <div className="palette">{arcColors.map((color, index) => <span key={color}><i style={{ background: color }} />{index}</span>)}</div>
          <ArcGrid title="INPUT" values={example.input} />
          <ArcGrid title="OUTPUT" values={example.output} />
        </section>
        <section className="graphPanel">
          <ReactFlow nodes={nodes} edges={edges} nodeTypes={nodeTypes} onNodesChange={onNodesChange} onEdgesChange={onEdgesChange} onConnect={onConnect} onNodeClick={(_, node) => setSelectedId(node.id)} fitView>
            <Background gap={18} size={1} color="#cbd5e1" />
            <Controls />
          </ReactFlow>
        </section>
      </main>
      <aside className="rightPanel">
        <h2>VALIDATION STATUS</h2>
        <ValidationPanel validation={validation} taskId={taskId} />
        <h2>INSPECTOR</h2>
        {selectedNode ? (
          <div className="inspector">
            <label>Node type<input value={selectedNode.data.opType} onChange={(e) => updateSelected({ opType: e.target.value })} /></label>
            <label>ID<input value={selectedNode.id} readOnly /></label>
            <label>Input source<select>{nodes.filter((node) => node.id !== selectedNode.id).map((node) => <option key={node.id}>{node.id}</option>)}</select></label>
            <label>Shape<textarea value={selectedNode.data.shape || ""} onChange={(e) => updateSelected({ shape: e.target.value })} /></label>
            <label>Axes / Keepdims<textarea value={JSON.stringify(selectedNode.data.attrs || {}, null, 2)} onChange={(e) => updateSelected({ attrsText: e.target.value })} /></label>
            <button className="danger" onClick={deleteSelected}><Trash2 size={15} />Delete Node</button>
          </div>
        ) : <p className="muted">Select a node to inspect graph bindings and ONNX attributes.</p>}
        <div className="status">{status}</div>
      </aside>
    </div>
  );
}

createRoot(document.getElementById("root")).render(<App />);
