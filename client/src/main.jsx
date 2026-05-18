import React, { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { createRoot } from "react-dom/client";
import ReactFlow, { Background, Controls, Handle, Position, addEdge, useEdgesState, useNodesState } from "reactflow";
import axios from "axios";
import { Play, Save, Trash2, Upload, Plus, ChevronLeft, ChevronRight } from "lucide-react";
import "reactflow/dist/style.css";
import "./style.css";

const arcColors = ["#000000", "#0074D9", "#FF4136", "#2ECC40", "#FFDC00", "#AAAAAA", "#F012BE", "#FF851B", "#7FDBFF", "#870C25"];
const baseOps = ["Input", "Output", "Constant", "RowIndex", "ColIndex", "Cast", "Identity", "Equal", "Greater", "Less", "GreaterOrEqual", "LessOrEqual", "Not", "And", "Or", "Xor", "Add", "Sub", "Mul", "Div", "Mod", "Min", "Max", "Sum", "Relu", "Abs", "Neg", "Floor", "Clip", "Sign", "Sqrt", "ReduceSum", "ReduceMax", "ReduceMin", "ArgMax", "Where", "Slice", "Pad", "Concat", "Transpose", "Tile", "Resize", "Conv"];
const PROJECT_STORAGE_KEY = "neurogolf-projects-v1";
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
  GreaterOrEqual: ["a", "b"],
  LessOrEqual: ["a", "b"],
  And: ["a", "b"],
  Or: ["a", "b"],
  Xor: ["a", "b"],
  Add: ["a", "b"],
  Sub: ["a", "b"],
  Mul: ["a", "b"],
  Div: ["a", "b"],
  Mod: ["a", "b"],
  Min: ["a", "b"],
  Max: ["a", "b"],
  Sum: ["a", "b"],
  Relu: ["input"],
  Abs: ["input"],
  Neg: ["input"],
  Floor: ["input"],
  Clip: ["input"],
  Sign: ["input"],
  Sqrt: ["input"],
  ReduceMax: ["input"],
  ReduceMin: ["input"],
  Where: ["condition", "true", "false"],
  Concat: ["a", "b"],
};

function defaultAttrs(opType) {
  if (opType === "Cast") return { to: "1" };
  if (opType === "ReduceSum") return { axes: [2, 3], keepdims: 1 };
  if (opType === "ReduceMax") return { axes: [2, 3], keepdims: 1 };
  if (opType === "ReduceMin") return { axes: [2, 3], keepdims: 1 };
  if (opType === "ArgMax") return { axis: 1, keepdims: 1 };
  if (opType === "Clip") return { min: 0, max: 9 };
  if (opType === "Slice") return { starts: [0, 0, 0, 0], ends: [1, 1, 30, 30], axes: [0, 1, 2, 3], steps: [1, 1, 1, 1] };
  if (opType === "Pad") return { pads: [0, 0, 0, 0, 0, 0, 0, 0], value: 0 };
  if (opType === "Concat") return { axis: 1 };
  if (opType === "Transpose") return { perm: [0, 1, 3, 2] };
  if (opType === "Tile") return { repeats: [1, 1, 1, 1] };
  if (opType === "Resize") return { sizes: [1, 1, 30, 30], mode: "nearest" };
  if (opType === "Conv") return { weight_shape: [1, 1, 3, 3], weights: [1, 1, 1, 1, 1, 1, 1, 1, 1], pads: [1, 1, 1, 1], strides: [1, 1] };
  return {};
}

function slotsForOp(opType, opInputs = {}) {
  if (inputSlots[opType]) return inputSlots[opType];
  if (opType === "Input" || opType === "Constant") return [];
  const count = opInputs[opType] || 1;
  return Array.from({ length: count }, (_, index) => `in${index}`);
}

function clampTask(value) {
  const parsed = Number(value || 1);
  if (!Number.isFinite(parsed)) return 1;
  return Math.min(TASK_COUNT, Math.max(1, Math.trunc(parsed)));
}

function taskIdFor(taskNumber) {
  return `task${String(taskNumber).padStart(3, "0")}`;
}

function defaultNodes() {
  return [
    { id: "input_1", type: "op", position: { x: 70, y: 80 }, data: { label: "input_1", opType: "Input", shape: "1,1,30,30" } },
    { id: "output_1", type: "op", position: { x: 310, y: 80 }, data: { label: "output_1", opType: "Output", shape: "1,1,30,30" } },
  ];
}

function defaultEdges() {
  return [{ id: "e1", source: "input_1", target: "output_1", targetHandle: "input" }];
}

function cloneGraph(value) {
  return JSON.parse(JSON.stringify(value));
}

function builtinProjects() {
  const input = { id: "input_1", type: "op", position: { x: 70, y: 80 }, data: { label: "input_1", opType: "Input", shape: "1,1,30,30" } };
  const output = { id: "output_1", type: "op", position: { x: 760, y: 95 }, data: { label: "output_1", opType: "Output", shape: "1,1,30,30" } };
  return [
    {
      name: "baseline-cast-equal",
      taskId: "task010",
      nodes: [
        input,
        { id: "const_0", type: "op", position: { x: 250, y: 165 }, data: { label: "const_0", opType: "Constant", shape: "1,1,30,30", value: "0" } },
        { id: "equal_1", type: "op", position: { x: 430, y: 85 }, data: { label: "equal_1", opType: "Equal", shape: "1,1,30,30" } },
        { id: "cast_1", type: "op", position: { x: 600, y: 95 }, data: { label: "cast_1", opType: "Cast", shape: "1,1,30,30", attrs: { to: "1" } } },
        output,
      ],
      edges: [
        { id: "eq_a", source: "input_1", target: "equal_1", targetHandle: "a" },
        { id: "eq_b", source: "const_0", target: "equal_1", targetHandle: "b" },
        { id: "cast_in", source: "equal_1", target: "cast_1", targetHandle: "input" },
        { id: "out", source: "cast_1", target: "output_1", targetHandle: "input" },
      ],
    },
    {
      name: "argmax-mask-v2",
      taskId: "task010",
      nodes: [
        input,
        { id: "argmax_1", type: "op", position: { x: 300, y: 80 }, data: { label: "argmax_1", opType: "ArgMax", shape: "1,1,30,30", attrs: { axis: 1, keepdims: 1 } } },
        { id: "cast_1", type: "op", position: { x: 520, y: 90 }, data: { label: "cast_1", opType: "Cast", shape: "1,1,30,30", attrs: { to: "1" } } },
        output,
      ],
      edges: [
        { id: "argmax_in", source: "input_1", target: "argmax_1", targetHandle: "input" },
        { id: "cast_in", source: "argmax_1", target: "cast_1", targetHandle: "input" },
        { id: "out", source: "cast_1", target: "output_1", targetHandle: "input" },
      ],
    },
    {
      name: "reduce-sum-probe",
      taskId: "task010",
      nodes: [
        input,
        { id: "reduce_1", type: "op", position: { x: 330, y: 80 }, data: { label: "reduce_1", opType: "ReduceSum", shape: "1,1,30,30", attrs: { axes: [1], keepdims: 1 } } },
        output,
      ],
      edges: [
        { id: "reduce_in", source: "input_1", target: "reduce_1", targetHandle: "input" },
        { id: "out", source: "reduce_1", target: "output_1", targetHandle: "input" },
      ],
    },
  ].map((project) => ({ ...project, nodes: cloneGraph(project.nodes), edges: cloneGraph(project.edges) }));
}

function loadStoredProjects() {
  try {
    const parsed = JSON.parse(window.localStorage.getItem(PROJECT_STORAGE_KEY) || "[]");
    if (!Array.isArray(parsed)) return [];
    return parsed.filter((project) => project?.name && Array.isArray(project.nodes) && Array.isArray(project.edges));
  } catch {
    return [];
  }
}

function persistStoredProjects(projects) {
  window.localStorage.setItem(PROJECT_STORAGE_KEY, JSON.stringify(projects));
}

function OpNode({ data, selected }) {
  const slots = data.inputSlots || inputSlots[data.opType] || [];
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

function formatBytes(value) {
  if (!Number.isFinite(value)) return "";
  if (value < 1024) return `${value} B`;
  if (value < 1024 * 1024) return `${(value / 1024).toFixed(1)} KB`;
  return `${(value / (1024 * 1024)).toFixed(1)} MB`;
}

function nowLabel() {
  return new Date().toLocaleTimeString([], { hour: "2-digit", minute: "2-digit", second: "2-digit" });
}

function buildExportPayload({ projectName, taskId, nodes, edges, trainingPairs }) {
  return {
    projectName,
    taskId,
    nodes,
    edges,
    trainingPairs,
  };
}

function validatePayload(payload) {
  const checks = [];
  const nodeIds = new Set();
  const duplicateIds = new Set();
  for (const node of payload.nodes) {
    if (nodeIds.has(node.id)) duplicateIds.add(node.id);
    nodeIds.add(node.id);
  }
  const edgeErrors = payload.edges.filter((edge) => !nodeIds.has(edge.source) || !nodeIds.has(edge.target));
  const inputNodes = payload.nodes.filter((node) => node.data?.opType === "Input");
  const outputNodes = payload.nodes.filter((node) => node.data?.opType === "Output");
  const connectedOutputs = outputNodes.filter((node) => payload.edges.some((edge) => edge.target === node.id));
  const unknownOps = payload.nodes.filter((node) => !node.data?.opType);

  checks.push({ label: "Project name", ok: Boolean(payload.projectName?.trim()), detail: payload.projectName || "missing" });
  checks.push({ label: "Task id", ok: /^task\d{3}$/.test(payload.taskId), detail: payload.taskId });
  checks.push({ label: "Nodes", ok: payload.nodes.length > 0, detail: `${payload.nodes.length}` });
  checks.push({ label: "Edges", ok: edgeErrors.length === 0, detail: edgeErrors.length ? `${edgeErrors.length} malformed` : `${payload.edges.length}` });
  checks.push({ label: "Input node", ok: inputNodes.length > 0, detail: `${inputNodes.length}` });
  checks.push({ label: "Connected output", ok: connectedOutputs.length > 0, detail: `${connectedOutputs.length}/${outputNodes.length}` });
  checks.push({ label: "Training pairs", ok: payload.trainingPairs.length > 0, detail: `${payload.trainingPairs.length}` });
  checks.push({ label: "Duplicate ids", ok: duplicateIds.size === 0, detail: duplicateIds.size ? Array.from(duplicateIds).join(", ") : "none" });
  checks.push({ label: "Op labels", ok: unknownOps.length === 0, detail: unknownOps.length ? `${unknownOps.length} missing` : "ready" });

  return {
    ok: checks.every((check) => check.ok),
    checks,
  };
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

function ExportInspector({ payload, payloadJson, payloadValidation, selectedNode, selectedEdges, lastRun, lastExport, lastDownload }) {
  const selectedEdge = selectedEdges[0];
  const selectedLabel = selectedNode
    ? `${selectedNode.id} / ${selectedNode.data?.opType || "Unknown"}`
    : selectedEdge
      ? `${selectedEdge.source} -> ${selectedEdge.target}${selectedEdge.targetHandle ? `.${selectedEdge.targetHandle}` : ""}`
      : "none";
  const lastRunLabel = lastRun.state === "idle"
    ? "not run"
    : `${lastRun.state}${lastRun.shape ? ` / ${lastRun.shape}` : ""}${lastRun.reason ? ` / ${lastRun.reason}` : ""}`;
  const lastExportLabel = lastExport.state === "idle"
    ? "not exported"
    : `${lastExport.state}${lastExport.artifact ? ` / ${lastExport.artifact}` : ""}${lastExport.reason ? ` / ${lastExport.reason}` : ""}`;
  const lastDownloadLabel = lastDownload.state === "idle"
    ? "not downloaded"
    : `${lastDownload.name} / ${lastDownload.at}`;

  return (
    <section className="payloadInspector" aria-label="Export payload inspector">
      <h2>EXPORT PAYLOAD</h2>
      <div className="payloadGrid">
        <div><span>Project</span><strong>{payload.projectName}</strong></div>
        <div><span>Task</span><strong>{payload.taskId}</strong></div>
        <div><span>Nodes</span><strong data-testid="payload-node-count">{payload.nodes.length}</strong></div>
        <div><span>Edges</span><strong data-testid="payload-edge-count">{payload.edges.length}</strong></div>
        <div><span>Train</span><strong>{payload.trainingPairs.length}</strong></div>
        <div><span>Selected</span><strong data-testid="payload-selected">{selectedLabel}</strong></div>
      </div>
      <div className="payloadChecks">
        {payloadValidation.checks.map((check) => (
          <div key={check.label} className={check.ok ? "payloadCheck ok" : "payloadCheck bad"}>
            <span>{check.ok ? "OK" : "ERR"}</span>
            <strong>{check.label}</strong>
            <em>{check.detail}</em>
          </div>
        ))}
      </div>
      <div className="payloadEvents">
        <div><span>Last run</span><strong data-testid="payload-last-run">{lastRunLabel}</strong></div>
        <div><span>Last export</span><strong data-testid="payload-last-export">{lastExportLabel}</strong></div>
        <div><span>Last download</span><strong data-testid="payload-last-download">{lastDownloadLabel}</strong></div>
      </div>
      <details className="payloadJson">
        <summary>Payload JSON</summary>
        <pre>{payloadJson}</pre>
      </details>
    </section>
  );
}

function App() {
  const [task, setTask] = useState(10);
  const [projectName, setProjectName] = useState("neurogolf-task010");
  const [storedProjects, setStoredProjects] = useState(() => loadStoredProjects());
  const [bestManifest, setBestManifest] = useState(null);
  const [bestLoadState, setBestLoadState] = useState("loading");
  const [selectedProjectName, setSelectedProjectName] = useState("baseline-cast-equal");
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
  const [reactFlowInstance, setReactFlowInstance] = useState(null);
  const [exampleMode, setExampleMode] = useState("train");
  const [exampleIndex, setExampleIndex] = useState(0);
  const [runOutput, setRunOutput] = useState(null);
  const [lastRun, setLastRun] = useState({ state: "idle" });
  const [lastExport, setLastExport] = useState({ state: "idle" });
  const [lastDownload, setLastDownload] = useState({ state: "idle" });
  const [lastImport, setLastImport] = useState({ state: "idle" });
  const [importedTasks, setImportedTasks] = useState(() => new Set());
  const importInputRef = useRef(null);
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
  const templateProjects = useMemo(() => builtinProjects(), []);
  const availableProjects = useMemo(() => {
    const byName = new Map(templateProjects.map((project) => [project.name, { ...project, source: "template" }]));
    for (const project of storedProjects) byName.set(project.name, { ...project, source: "saved" });
    return Array.from(byName.values()).sort((a, b) => a.name.localeCompare(b.name));
  }, [storedProjects, templateProjects]);
  const [nodes, setNodes, onNodesChange] = useNodesState(defaultNodes());
  const [edges, setEdges, onEdgesChange] = useEdgesState(defaultEdges());
  const selectedNode = nodes.find((node) => node.id === selectedId);
  const selectedEdges = edges.filter((edge) => selectedEdgeIds.includes(edge.id));
  const exportPayload = useMemo(
    () => buildExportPayload({ projectName, taskId, nodes, edges, trainingPairs }),
    [projectName, taskId, nodes, edges, trainingPairs]
  );
  const payloadValidation = useMemo(() => validatePayload(exportPayload), [exportPayload]);
  const payloadPreview = useMemo(() => JSON.stringify(exportPayload, null, 2), [exportPayload]);
  const opInputs = bestManifest?.opInputs || {};
  const availableOps = useMemo(
    () => Array.from(new Set([...baseOps, ...(bestManifest?.opTypes || [])])).sort((a, b) => a.localeCompare(b)),
    [bestManifest]
  );
  const bestTask = useMemo(
    () => bestManifest?.tasks?.find((item) => item.taskId === taskId),
    [bestManifest, taskId]
  );

  useEffect(() => {
    let cancelled = false;
    setTaskLoadState("loading");
    setTaskLoadError("");
    setValidation({ state: "idle" });
    setExampleMode("train");
    setExampleIndex(0);
    setRunOutput(null);
    setLastRun({ state: "idle" });
    setLastExport({ state: "idle" });
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

  useEffect(() => {
    let cancelled = false;
    fetch("/best/manifest.json", { cache: "no-cache" })
      .then((response) => {
        if (!response.ok) throw new Error(`best manifest returned HTTP ${response.status}`);
        return response.json();
      })
      .then((data) => {
        if (cancelled) return;
        setBestManifest(data);
        setBestLoadState("loaded");
      })
      .catch(() => {
        if (cancelled) return;
        setBestManifest(null);
        setBestLoadState("failed");
      });
    return () => {
      cancelled = true;
    };
  }, []);

  useEffect(() => {
    let cancelled = false;
    axios.get("/api/import/list")
      .then(({ data }) => {
        if (cancelled) return;
        setImportedTasks(new Set((data.items || []).map((item) => item.taskId)));
      })
      .catch(() => {});
    return () => {
      cancelled = true;
    };
  }, []);

  const scheduleFitView = () => {
    window.requestAnimationFrame(() => {
      window.requestAnimationFrame(() => reactFlowInstance?.fitView({ padding: 0.2 }));
    });
  };

  const addNode = (opType) => {
    const count = nodes.filter((node) => node.data?.opType === opType).length + 1;
    const id = `${opType.toLowerCase()}_${count}`;
    const data = { label: id, opType, shape: "1,1,30,30", attrs: defaultAttrs(opType), inputSlots: slotsForOp(opType, opInputs) };
    if (opType === "Constant") data.value = "0";
    setNodes((current) => [...current, { id, type: "op", position: { x: 160 + current.length * 28, y: 190 }, data }]);
    setSelectedId(id);
    setSelectedEdgeIds([]);
  };

  const resetTransientGraphState = () => {
    setSelectedId(null);
    setSelectedEdgeIds([]);
    setRunOutput(null);
    setValidation({ state: "idle" });
    setLastRun({ state: "idle" });
    setLastExport({ state: "idle" });
  };

  const newProject = () => {
    const name = `neurogolf-${taskId}`;
    setProjectName(name);
    setNodes(cloneGraph(defaultNodes()));
    setEdges(cloneGraph(defaultEdges()));
    resetTransientGraphState();
    setStatus(`New project ${name}`);
    scheduleFitView();
  };

  const saveProject = () => {
    const name = projectName.trim();
    if (!name) {
      setStatus("Project name is required before save");
      return;
    }
    const snapshot = {
      name,
      taskId,
      savedAt: new Date().toISOString(),
      nodes: cloneGraph(nodes),
      edges: cloneGraph(edges),
    };
    setStoredProjects((current) => {
      const next = [snapshot, ...current.filter((project) => project.name !== name)];
      persistStoredProjects(next);
      return next;
    });
    setSelectedProjectName(name);
    setStatus(`Saved project ${name}`);
  };

  const loadProject = () => {
    const project = availableProjects.find((item) => item.name === selectedProjectName);
    if (!project) {
      setStatus("Select a saved project to load");
      return;
    }
    const nextTask = Number(project.taskId?.replace("task", "")) || task;
    setTask(clampTask(nextTask));
    setProjectName(project.name);
    setNodes(cloneGraph(project.nodes));
    setEdges(cloneGraph(project.edges));
    resetTransientGraphState();
    setStatus(`Loaded project ${project.name}`);
    scheduleFitView();
  };

  const applyGraphFromBest = async (id) => {
    try {
      const { data } = await axios.get(`/api/best-graph/${id}`);
      setProjectName(data.projectName || `best-${id}-onnx`);
      setNodes(cloneGraph(data.nodes || []));
      setEdges(cloneGraph(data.edges || []));
      resetTransientGraphState();
      setStatus(`Loaded ONNX graph for ${id}: ${data.meta?.nodeCount || 0} raw nodes`);
      scheduleFitView();
      return true;
    } catch (error) {
      const response = error.response?.data;
      const rawReason = response?.reason || response?.detail || error.message;
      setStatus(`Graph load failed: ${typeof rawReason === "string" ? rawReason : JSON.stringify(rawReason)}`);
      return false;
    }
  };

  const importOnnx = async (event) => {
    const files = Array.from(event.target.files || []);
    event.target.value = "";
    if (files.length === 0) return;
    setLastImport({ state: "loading", count: files.length, at: nowLabel() });
    setStatus(`Importing ${files.length} ONNX file${files.length === 1 ? "" : "s"}`);
    const form = new FormData();
    for (const file of files) form.append("files", file, file.name);
    try {
      const { data } = await axios.post("/api/import", form);
      const saved = data.saved || [];
      const rejected = data.rejected || [];
      setLastImport({ state: "ok", saved, rejected, at: nowLabel() });
      setImportedTasks((current) => {
        const next = new Set(current);
        for (const id of saved) next.add(id);
        return next;
      });
      const parts = [`${saved.length} saved`];
      if (rejected.length) parts.push(`${rejected.length} rejected`);
      setStatus(`Import: ${parts.join(", ")}`);
      if (saved.includes(taskId)) {
        await applyGraphFromBest(taskId);
      }
    } catch (error) {
      const reason = error.response?.data?.reason || error.message;
      setLastImport({ state: "failed", reason, at: nowLabel() });
      setStatus(`Import failed: ${reason}`);
    }
  };

  const loadBestGraph = async () => {
    if (!bestTask && !importedTasks.has(taskId)) {
      setStatus(`No ONNX graph for ${taskId}`);
      return;
    }
    setStatus(`Loading ONNX graph for ${taskId}`);
    await applyGraphFromBest(taskId);
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
      setLastExport({ state: "failed", taskId, reason: taskLoadError || "Selected task has no loaded training pairs", at: nowLabel() });
      return;
    }
    setStatus("Exporting ONNX");
    setValidation({ state: "loading", taskId });
    setLastExport({ state: "loading", taskId, at: nowLabel() });
    try {
      const response = await axios.post("/api/export", exportPayload, { responseType: "blob" });
      const ct = response.headers["content-type"] || "";
      const fname = `${taskId}.onnx`;
      if (ct.includes("application/json")) {
        const data = JSON.parse(await response.data.text());
        setValidation({ state: "passed", taskId, artifact: data.artifact });
        setLastExport({ state: "passed", taskId, artifact: data.artifact, at: nowLabel() });
        setStatus(`Validation passed. Exported ${data.artifact}`);
      } else {
        const url = URL.createObjectURL(response.data);
        const a = document.createElement("a");
        a.href = url;
        a.download = fname;
        document.body.appendChild(a);
        a.click();
        a.remove();
        URL.revokeObjectURL(url);
        setValidation({ state: "passed", taskId, artifact: fname });
        setLastExport({ state: "passed", taskId, artifact: fname, at: nowLabel() });
        setLastDownload({ state: "downloaded", name: fname, taskId, at: nowLabel() });
        setStatus(`Validation passed. Downloaded ${fname}`);
      }
    } catch (error) {
      let reason = error.message;
      if (error.response?.data) {
        try {
          const text = await error.response.data.text();
          const parsed = JSON.parse(text);
          reason = parsed.reason || parsed.detail?.reason || parsed.detail || reason;
        } catch {
          reason = error.message;
        }
      }
      if (typeof reason !== "string") reason = JSON.stringify(reason);
      setValidation({ state: "failed", taskId, reason });
      setLastExport({ state: "failed", taskId, reason, at: nowLabel() });
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
      const { data } = await axios.post("/api/compile", exportPayload);
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
      setLastRun({ state: "failed", taskId, reason: "Displayed example has no input grid to run", at: nowLabel() });
      return;
    }
    setStatus("Running graph");
    setLastRun({ state: "loading", taskId, example: `${exampleMode} ${exampleIndex + 1}`, at: nowLabel() });
    try {
      const payload = { ...exportPayload, inputGrid: example.input };
      const { data } = await axios.post("/api/run", payload);
      setRunOutput(data.grid);
      setLastRun({ state: "passed", taskId, shape: gridShape(data.grid), example: `${exampleMode} ${exampleIndex + 1}`, at: nowLabel() });
      setStatus(`Run produced ${gridShape(data.grid)} output from ${exampleMode} ${exampleIndex + 1}`);
      setValidation((current) => current.state === "failed" ? { state: "idle" } : current);
    } catch (error) {
      const response = error.response?.data;
      const rawReason = response?.reason || response?.detail?.reason || response?.detail || error.message;
      const reason = typeof rawReason === "string" ? rawReason : JSON.stringify(rawReason);
      setValidation({ state: "failed", taskId, reason });
      setLastRun({ state: "failed", taskId, reason, example: `${exampleMode} ${exampleIndex + 1}`, at: nowLabel() });
      setStatus("Run failed");
    }
  };

  const recordDownload = (name, path) => {
    setLastDownload({ state: "downloaded", name, path, taskId, at: nowLabel() });
  };

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
          <input aria-label="Project name" value={projectName} onChange={(e) => setProjectName(e.target.value)} />
          <div className="twoCol"><button className="btn" onClick={newProject}>New</button><button className="btn" onClick={saveProject}><Save size={15} />Save</button></div>
          <select aria-label="Saved project" value={selectedProjectName} onChange={(event) => setSelectedProjectName(event.target.value)}>
            {availableProjects.map((item) => <option key={item.name} value={item.name}>{item.name}{item.source === "template" ? " (template)" : ""}</option>)}
          </select>
          <button className="btn full" onClick={loadProject} disabled={availableProjects.length === 0}>Load Selected</button>
        </section>
        <section>
          <h2>IMPORT ONNX</h2>
          <input
            ref={importInputRef}
            type="file"
            accept=".onnx"
            multiple
            style={{ display: "none" }}
            onChange={importOnnx}
            aria-label="Import ONNX files"
          />
          <button className="btn full" onClick={() => importInputRef.current?.click()} disabled={lastImport.state === "loading"}>
            <Upload size={15} />{lastImport.state === "loading" ? "Uploading..." : "Pick task001.onnx ... taskNNN.onnx"}
          </button>
          {lastImport.state === "ok" && (
            <div className="muted" style={{ marginTop: 6 }}>
              {lastImport.saved.length ? `Saved: ${lastImport.saved.join(", ")}` : "No files saved"}
              {lastImport.rejected.length ? ` · Rejected: ${lastImport.rejected.length}` : ""}
            </div>
          )}
          {lastImport.state === "failed" && (
            <div className="muted" style={{ marginTop: 6 }}>Failed: {lastImport.reason}</div>
          )}
        </section>
        <section>
          <h2>BEST TEMPLATE</h2>
          {bestLoadState === "loaded" && bestManifest ? (
            <div className="bestTemplate">
              <div className="muted">{bestManifest.fileCount} bundled · {importedTasks.size} imported</div>
              {bestTask || importedTasks.has(taskId) ? (
                <>
                  <button className="btn full" onClick={loadBestGraph}>Load Graph{importedTasks.has(taskId) && !bestTask ? " (imported)" : ""}</button>
                  {bestTask && (
                    <a className="btn full btnLink" href={bestTask.path} download onClick={() => recordDownload(`${taskId}.onnx`, bestTask.path)}>
                      Task {String(task).padStart(3, "0")} ONNX
                    </a>
                  )}
                </>
              ) : (
                <div className="muted">No ONNX file for {taskId}</div>
              )}
              <a className="btn full btnLink" href={bestManifest.zipPath} download onClick={() => recordDownload(bestManifest.zipName || "submission-best.zip", bestManifest.zipPath)}>
                Full best zip {formatBytes(bestManifest.zipSize)}
              </a>
            </div>
          ) : (
            <div className="muted">{bestLoadState === "failed" ? "Best template unavailable" : "Loading best template"}</div>
          )}
        </section>
        <section>
          <h2>ADD NODE</h2>
          <div className="quickAdd">
            <select value={quickOp} onChange={(event) => setQuickOp(event.target.value)}>
              {availableOps.map((op) => <option key={op}>{op}</option>)}
            </select>
            <button className="btn" onClick={() => addNode(quickOp)}><Plus size={14} />Add</button>
          </div>
          <div className="opGrid">{availableOps.map((op) => <button key={op} onClick={() => addNode(op)}><Plus size={14} />{op}</button>)}</div>
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
            onInit={setReactFlowInstance}
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
        <ExportInspector
          payload={exportPayload}
          payloadJson={payloadPreview}
          payloadValidation={payloadValidation}
          selectedNode={selectedNode}
          selectedEdges={selectedEdges}
          lastRun={lastRun}
          lastExport={lastExport}
          lastDownload={lastDownload}
        />
        <div className="status">{status}</div>
      </aside>
    </div>
  );
}

createRoot(document.getElementById("root")).render(<App />);
