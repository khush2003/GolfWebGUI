import React, { useCallback, useEffect, useMemo, useState } from "react";
import { createRoot } from "react-dom/client";
import ReactFlow, { Background, Controls, Handle, Position, addEdge, useEdgesState, useNodesState } from "reactflow";
import axios from "axios";
import { Play, Save, Trash2, Upload, Plus, ChevronLeft, ChevronRight } from "lucide-react";
import "reactflow/dist/style.css";
import "./style.css";

const arcColors = ["#000000", "#0074D9", "#FF4136", "#2ECC40", "#FFDC00", "#AAAAAA", "#F012BE", "#FF851B", "#7FDBFF", "#870C25"];
const ops = ["Input", "Output", "Constant", "RowIndex", "ColIndex", "Cast", "Identity", "Equal", "Greater", "Less", "Not", "And", "Add", "Sub", "Mul", "Div", "ReduceSum", "ArgMax", "Where", "Slice", "Pad", "Concat", "Transpose", "Tile", "Resize", "Conv"];
const saved = ["baseline-cast-equal", "argmax-mask-v2", "reduce-sum-probe"];
const TASK_COUNT = 400;
const TASK_PAGE_SIZE = 40;
const inputSlots = {
  Cast: ["input"],
  Identity: ["input"],
  Not: ["input"],
  ReduceSum: ["input"],
  ArgMax: ["input"],
  Slice: ["input"],
  Pad: ["input"],
  Transpose: ["input"],
  Tile: ["input"],
  Resize: ["input"],
  Conv: ["input"],
  Output: ["input"],
  Equal: ["a", "b"],
  Greater: ["a", "b"],
  Less: ["a", "b"],
  And: ["a", "b"],
  Add: ["a", "b"],
  Sub: ["a", "b"],
  Mul: ["a", "b"],
  Div: ["a", "b"],
  Where: ["condition", "true", "false"],
  Concat: ["a", "b"],
};

function defaultAttrs(opType) {
  if (opType === "Cast") return { to: "1" };
  if (opType === "ReduceSum") return { axes: [2, 3], keepdims: 1 };
  if (opType === "ArgMax") return { axis: 1, keepdims: 1 };
  if (opType === "Slice") return { starts: [0, 0, 0, 0], ends: [1, 1, 30, 30], axes: [0, 1, 2, 3], steps: [1, 1, 1, 1] };
  if (opType === "Pad") return { pads: [0, 0, 0, 0, 0, 0, 0, 0], value: 0 };
  if (opType === "Concat") return { axis: 1 };
  if (opType === "Transpose") return { perm: [0, 1, 3, 2] };
  if (opType === "Tile") return { repeats: [1, 1, 1, 1] };
  if (opType === "Resize") return { sizes: [1, 1, 30, 30], mode: "nearest" };
  if (opType === "Conv") return { weight_shape: [1, 1, 3, 3], weights: [1, 1, 1, 1, 1, 1, 1, 1, 1], pads: [1, 1, 1, 1], strides: [1, 1] };
  return {};
}

function clampTask(value) {
  const parsed = Number(value || 1);
  if (!Number.isFinite(parsed)) return 1;
  return Math.min(TASK_COUNT, Math.max(1, Math.trunc(parsed)));
}

function taskIdFor(taskNumber) {
  return `task${String(taskNumber).padStart(3, "0")}`;
}

function OpNode({ data, selected }) {
  const slots = inputSlots[data.opType] || [];
  return (
    <div className={`node ${selected ? "nodeSelected" : ""}`}>
      {slots.map((slot, index) => (
        <Handle
          key={slot}
          id={slot}
          type="target"
          position={Position.Left}
          style={{ top: `${((index + 1) / (slots.length + 1)) * 100}%` }}
        />
      ))}
      {slots.length > 0 && <div className="nodeSlots">{slots.join(" / ")}</div>}
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
  const [exampleHeight, setExampleHeight] = useState(330);
  const [isResizingExamples, setIsResizingExamples] = useState(false);
  const [quickOp, setQuickOp] = useState("Constant");
  const [selectedEdgeIds, setSelectedEdgeIds] = useState([]);
  const [exampleMode, setExampleMode] = useState("train");
  const [exampleIndex, setExampleIndex] = useState(0);
  const [runOutput, setRunOutput] = useState(null);
  const exampleSets = useMemo(() => ({
    train: currentTask?.train || [],
    test: currentTask?.test || [],
    extra: currentTask?.["arc-gen"] || [],
  }), [currentTask]);
  const visibleExamples = exampleSets[exampleMode] || [];
  const example = visibleExamples[exampleIndex] || currentTask?.train?.[0] || { input: [[0]], output: [[0]] };
  const expectedOutput = example.output || example.target || [[0]];
  const displayedOutput = runOutput || expectedOutput;
  const displayedOutputTitle = runOutput ? "RUN OUTPUT" : "OUTPUT";
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
      { id: "output_1", type: "op", position: { x: 310, y: 80 }, data: { label: "output_1", opType: "Output", shape: "1,1,30,30" } },
    ],
    []
  );
  const [nodes, setNodes, onNodesChange] = useNodesState(initialNodes);
  const [edges, setEdges, onEdgesChange] = useEdgesState([
    { id: "e1", source: "input_1", target: "output_1", targetHandle: "input" },
  ]);
  const selectedNode = nodes.find((node) => node.id === selectedId);
  const selectedEdges = edges.filter((edge) => selectedEdgeIds.includes(edge.id));

  useEffect(() => {
    let cancelled = false;
    setTaskLoadState("loading");
    setTaskLoadError("");
    setValidation({ state: "idle" });
    setExampleMode("train");
    setExampleIndex(0);
    setRunOutput(null);
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
    const count = nodes.filter((node) => node.data?.opType === opType).length + 1;
    const id = `${opType.toLowerCase()}_${count}`;
    const data = { label: id, opType, shape: "1,1,30,30", attrs: defaultAttrs(opType) };
    if (opType === "Constant") data.value = "0";
    setNodes((current) => [...current, { id, type: "op", position: { x: 160 + current.length * 28, y: 190 }, data }]);
    setSelectedId(id);
    setSelectedEdgeIds([]);
  };

  const updateSelected = (patch) => {
    setNodes((current) => current.map((node) => (node.id === selectedId ? { ...node, data: { ...node.data, ...patch } } : node)));
  };

  const deleteSelected = () => {
    const nodeId = selectedId;
    const edgeIds = new Set(selectedEdgeIds);
    if (!nodeId && edgeIds.size === 0) return;
    if (nodeId) {
      setNodes((current) => current.filter((node) => node.id !== nodeId));
    }
    setEdges((current) =>
      current.filter((edge) => !edgeIds.has(edge.id) && (!nodeId || (edge.source !== nodeId && edge.target !== nodeId)))
    );
    setSelectedId(null);
    setSelectedEdgeIds([]);
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

  const selectExampleMode = (mode) => {
    const examples = exampleSets[mode] || [];
    const nextIndex = mode === exampleMode && examples.length > 0 ? (exampleIndex + 1) % examples.length : 0;
    setExampleMode(mode);
    setExampleIndex(nextIndex);
    setRunOutput(null);
    const label = mode === "extra" ? "extra" : mode;
    setStatus(examples.length ? `${taskId} ${label} ${nextIndex + 1}/${examples.length}` : `${taskId} has no ${label} examples`);
  };

  const compileGraph = async () => {
    setStatus("Compiling graph");
    setRunOutput(null);
    try {
      const payload = { projectName, taskId, nodes, edges, trainingPairs };
      const { data } = await axios.post("/api/compile", payload);
      const outputShape = data.io?.outputs?.[0]?.shape?.join("x") || "unknown shape";
      setStatus(`Compile passed: ${data.modelBytes} bytes, output ${outputShape}`);
      setValidation((current) => current.state === "failed" ? { state: "idle" } : current);
    } catch (error) {
      const response = error.response?.data;
      const rawReason = response?.reason || response?.detail?.reason || response?.detail || error.message;
      const reason = typeof rawReason === "string" ? rawReason : JSON.stringify(rawReason);
      setValidation({ state: "failed", taskId, reason });
      setStatus("Compile failed");
    }
  };

  const runGraph = async () => {
    if (!example?.input) {
      setValidation({ state: "failed", taskId, reason: "Displayed example has no input grid to run" });
      return;
    }
    setStatus("Running graph");
    try {
      const payload = { projectName, taskId, nodes, edges, trainingPairs, inputGrid: example.input };
      const { data } = await axios.post("/api/run", payload);
      setRunOutput(data.grid);
      setStatus(`Run produced ${gridShape(data.grid)} output from ${exampleMode} ${exampleIndex + 1}`);
      setValidation((current) => current.state === "failed" ? { state: "idle" } : current);
    } catch (error) {
      const response = error.response?.data;
      const rawReason = response?.reason || response?.detail?.reason || response?.detail || error.message;
      const reason = typeof rawReason === "string" ? rawReason : JSON.stringify(rawReason);
      setValidation({ state: "failed", taskId, reason });
      setStatus("Run failed");
    }
  };

  const payloadPreview = useMemo(
    () => JSON.stringify({ projectName, taskId, nodes, edges, trainingPairs }, null, 2),
    [projectName, taskId, nodes, edges, trainingPairs]
  );

  const onConnect = useCallback((params) => setEdges((eds) => addEdge({ ...params, id: `${params.source}-${params.target}-${params.targetHandle || "input"}-${eds.length + 1}` }, eds)), [setEdges]);

  const onSelectionChange = useCallback(({ nodes: selectedNodes, edges: selectedGraphEdges }) => {
    setSelectedId(selectedNodes[0]?.id || null);
    setSelectedEdgeIds(selectedGraphEdges.map((edge) => edge.id));
  }, []);

  useEffect(() => {
    const onKeyDown = (event) => {
      const tag = event.target?.tagName?.toLowerCase();
      if (tag === "input" || tag === "textarea" || tag === "select" || event.target?.isContentEditable) return;
      if (event.key !== "Delete" && event.key !== "Backspace") return;
      if (!selectedId && selectedEdgeIds.length === 0) return;
      event.preventDefault();
      deleteSelected();
    };
    window.addEventListener("keydown", onKeyDown);
    return () => window.removeEventListener("keydown", onKeyDown);
  }, [selectedId, selectedEdgeIds]);

  useEffect(() => {
    if (!isResizingExamples) return;
    const onMove = (event) => {
      const topbar = document.querySelector(".topbar")?.getBoundingClientRect();
      if (!topbar) return;
      const nextHeight = event.clientY - topbar.bottom;
      setExampleHeight(Math.min(560, Math.max(170, nextHeight)));
    };
    const onStop = () => setIsResizingExamples(false);
    window.addEventListener("mousemove", onMove);
    window.addEventListener("mouseup", onStop);
    return () => {
      window.removeEventListener("mousemove", onMove);
      window.removeEventListener("mouseup", onStop);
    };
  }, [isResizingExamples]);

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
          <div className="quickAdd">
            <select value={quickOp} onChange={(event) => setQuickOp(event.target.value)}>
              {ops.map((op) => <option key={op}>{op}</option>)}
            </select>
            <button className="btn" onClick={() => addNode(quickOp)}><Plus size={14} />Add</button>
          </div>
          <div className="opGrid">{ops.map((op) => <button key={op} onClick={() => addNode(op)}><Plus size={14} />{op}</button>)}</div>
        </section>
      </aside>
      <main className="workspace" style={{ "--example-height": `${exampleHeight}px` }}>
        <nav className="topbar">
          <div><strong>Task {String(task).padStart(3, "0")}</strong><span>{stats}</span></div>
          <div className="actions">
            <button className={`btn ${exampleMode === "train" ? "activeMode" : ""}`} onClick={() => selectExampleMode("train")}>Train</button>
            <button className={`btn ${exampleMode === "test" ? "activeMode" : ""}`} onClick={() => selectExampleMode("test")}>Test</button>
            <button className={`btn ${exampleMode === "extra" ? "activeMode" : ""}`} onClick={() => selectExampleMode("extra")}>Extra 10</button>
            <button className="btn" onClick={compileGraph} disabled={taskLoadState !== "loaded"}>Compile</button>
            <button className="btn" onClick={runGraph} disabled={taskLoadState !== "loaded"}><Play size={15} />Run</button>
            <button className="btn primary" onClick={exportOnnx} disabled={validation.state === "loading" || taskLoadState !== "loaded"}><Upload size={15} />Export ONNX</button>
          </div>
        </nav>
        <section className="examples">
          <div className="palette">{arcColors.map((color, index) => <span key={color}><i style={{ background: color }} />{index}</span>)}</div>
          <ArcGrid title="INPUT" values={example.input} />
          <ArcGrid title={displayedOutputTitle} values={displayedOutput} />
        </section>
        <div
          className={`splitter ${isResizingExamples ? "active" : ""}`}
          onMouseDown={(event) => {
            event.preventDefault();
            setIsResizingExamples(true);
          }}
          role="separator"
          aria-orientation="horizontal"
          aria-label="Resize task preview"
        />
        <section className="graphPanel">
          <ReactFlow
            nodes={nodes}
            edges={edges}
            nodeTypes={nodeTypes}
            onNodesChange={onNodesChange}
            onEdgesChange={onEdgesChange}
            onConnect={onConnect}
            onSelectionChange={onSelectionChange}
            onNodeClick={(_, node) => {
              setSelectedId(node.id);
              setSelectedEdgeIds([]);
            }}
            onEdgeClick={(_, edge) => {
              setSelectedId(null);
              setSelectedEdgeIds([edge.id]);
            }}
            fitView
          >
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
            <label>Shape<textarea value={selectedNode.data.shape || ""} onChange={(e) => updateSelected({ shape: e.target.value })} /></label>
            {selectedNode.data.opType === "Constant" && (
              <label>Constant value<textarea value={selectedNode.data.value || ""} onChange={(e) => updateSelected({ value: e.target.value })} /></label>
            )}
            <label>Attributes JSON<textarea value={selectedNode.data.attrsText || JSON.stringify(selectedNode.data.attrs || {}, null, 2)} onChange={(e) => updateSelected({ attrsText: e.target.value })} /></label>
            <button className="danger" onClick={deleteSelected}><Trash2 size={15} />Delete Node</button>
          </div>
        ) : selectedEdges.length > 0 ? (
          <div className="inspector">
            <label>Selected edge<input value={`${selectedEdges[0].source} -> ${selectedEdges[0].target}${selectedEdges[0].targetHandle ? `.${selectedEdges[0].targetHandle}` : ""}`} readOnly /></label>
            <button className="danger" onClick={deleteSelected}><Trash2 size={15} />Delete Edge</button>
          </div>
        ) : <p className="muted">Select a node or edge to inspect graph bindings and ONNX attributes.</p>}
        <section>
          <h2>EXPORT PAYLOAD</h2>
          <textarea className="payloadPreview" value={payloadPreview} readOnly />
        </section>
        <div className="status">{status}</div>
      </aside>
    </div>
  );
}

createRoot(document.getElementById("root")).render(<App />);
