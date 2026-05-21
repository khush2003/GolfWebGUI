import React, { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { GraphCanvas } from "./canvas.jsx";
import { Inspector } from "./inspector.jsx";
import { detectSuggestions, SuggestionsPanel } from "./suggestions.jsx";
import {
  TopBar, LeftRail, Workbench, BottomPanel, StatusBar,
  CommandPalette, ImportModal, ExportModal, WorkspacesModal,
  rightTabStyle,
} from "./panels.jsx";
import { Icon } from "./components.jsx";
import {
  TASK_COUNT, NODE_TYPES, defaultAttrsFor, defaultDesignGraph, clone, uid,
  makeTaskRoster, backendGraphToDesign,
} from "./data.js";
import {
  fetchTask, fetchBestManifest, fetchBestGraph, fetchBaselineSummary,
  compileGraph as apiCompile, runGraph as apiRun, exportGraph as apiExport,
  checkCorrectness as apiCheck, exportZip as apiExportZip,
  importOnnxFiles, listImportedTasks, extractFailureReason,
} from "./api.js";

const WORKSPACES_KEY = "ng-workspaces-v3";
const GRAPHS_KEY_PREFIX = "ng-graphs-v3-";
const SETTINGS_KEY = "ng-settings-v3";

function loadWorkspaces() {
  try {
    const parsed = JSON.parse(localStorage.getItem(WORKSPACES_KEY) || "null");
    if (Array.isArray(parsed) && parsed.length) return parsed;
  } catch {}
  return [
    { id: "ws-default", name: "default", tasks: 0, lastSaved: "—", current: true },
  ];
}
function persistWorkspaces(ws) { localStorage.setItem(WORKSPACES_KEY, JSON.stringify(ws)); }

function loadGraphs(wsId) {
  try {
    const parsed = JSON.parse(localStorage.getItem(GRAPHS_KEY_PREFIX + wsId) || "{}");
    if (parsed && typeof parsed === "object") return parsed;
  } catch {}
  return {};
}
function persistGraphs(wsId, graphsByTask) {
  localStorage.setItem(GRAPHS_KEY_PREFIX + wsId, JSON.stringify(graphsByTask));
}

function loadSettings() {
  try {
    return JSON.parse(localStorage.getItem(SETTINGS_KEY) || "{}");
  } catch { return {}; }
}
function persistSettings(s) { localStorage.setItem(SETTINGS_KEY, JSON.stringify(s)); }

const nowStamp = () => new Date().toTimeString().slice(0, 8);
const humanTime = () => {
  const d = new Date();
  return d.toLocaleString();
};

export default function App() {
  const [workspaces, setWorkspaces] = useState(() => loadWorkspaces());
  const currentWorkspace = workspaces.find((w) => w.current) || workspaces[0];
  const wsId = currentWorkspace.id;

  const [taskId, setTaskIdRaw] = useState(() => loadSettings().taskId || "010");
  const [tasks, setTasks] = useState(() => makeTaskRoster(TASK_COUNT));
  const [graphsByTask, setGraphsByTask] = useState(() => loadGraphs(wsId));
  const [currentTaskData, setCurrentTaskData] = useState(null);
  const [taskLoadState, setTaskLoadState] = useState("loading");

  const [dirty, setDirty] = useState(false);
  const [selectedExample, setSelectedExample] = useState({ kind: "train", index: 0 });
  const [actualOutputByKey, setActualOutputByKey] = useState({});
  const [nodeOutputsByKey, setNodeOutputsByKey] = useState({});
  const [logs, setLogs] = useState([]);
  const [status, setStatus] = useState("Ready");
  const [running, setRunning] = useState(false);
  const [runProgress, setRunProgress] = useState(0);
  const [runStates, setRunStates] = useState({});
  const [validation, setValidation] = useState({ state: "idle" });
  const [importState, setImportState] = useState({ state: "idle" });
  const [bestManifest, setBestManifest] = useState(null);

  const [paletteOpen, setPaletteOpen] = useState(false);
  const [importOpen, setImportOpen] = useState(false);
  const [exportOpen, setExportOpen] = useState(false);
  const [workspacesModal, setWorkspacesModal] = useState(null);
  const [inspectorOpen, setInspectorOpen] = useState(true);
  const [bottomCollapsed, setBottomCollapsed] = useState(true);
  const [bottomHeight, setBottomHeight] = useState(220);
  const [bottomTab, setBottomTab] = useState("logs");
  const [fullscreen, setFullscreen] = useState(false);
  const [rightTab, setRightTab] = useState("inspector");
  const [highlightedNodes, setHighlightedNodes] = useState([]);
  const [historyByTask, setHistoryByTask] = useState({});
  const [importedTasks, setImportedTasks] = useState(new Set());
  const [efficiencyByTask, setEfficiencyByTask] = useState({});
  const [efficiencyErrorByTask, setEfficiencyErrorByTask] = useState({});
  const [efficiencyRefreshing, setEfficiencyRefreshing] = useState(false);
  const [baselineByTask, setBaselineByTask] = useState(() => {
    try { return JSON.parse(localStorage.getItem("ng-baseline-eff-v1") || "{}") || {}; } catch { return {}; }
  });
  const [bestByTask, setBestByTask] = useState(() => {
    try { return JSON.parse(localStorage.getItem("ng-best-eff-v1") || "{}") || {}; } catch { return {}; }
  });
  const [bestGraphByTask, setBestGraphByTask] = useState(() => {
    try { return JSON.parse(localStorage.getItem("ng-best-graph-v1") || "{}") || {}; } catch { return {}; }
  });
  useEffect(() => { localStorage.setItem("ng-baseline-eff-v1", JSON.stringify(baselineByTask)); }, [baselineByTask]);
  useEffect(() => { localStorage.setItem("ng-best-eff-v1", JSON.stringify(bestByTask)); }, [bestByTask]);
  useEffect(() => { localStorage.setItem("ng-best-graph-v1", JSON.stringify(bestGraphByTask)); }, [bestGraphByTask]);

  const graphsRef = useRef(graphsByTask);
  useEffect(() => { graphsRef.current = graphsByTask; }, [graphsByTask]);

  const ensureGraph = useCallback((id) => {
    if (graphsRef.current[id]) return graphsRef.current[id];
    const fresh = defaultDesignGraph();
    setGraphsByTask((prev) => ({ ...prev, [id]: fresh }));
    return fresh;
  }, []);

  const graph = graphsByTask[taskId] || defaultDesignGraph();

  const setTaskId = useCallback((id) => {
    setTaskIdRaw(id);
    setSelectedExample({ kind: "train", index: 0 });
    setRunStates({});
    setValidation({ state: "idle" });
    persistSettings({ taskId: id });
  }, []);

  useEffect(() => { ensureGraph(taskId); }, [taskId, ensureGraph]);

  useEffect(() => {
    let cancelled = false;
    setTaskLoadState("loading");
    fetchTask(taskId).then((data) => {
      if (cancelled) return;
      setCurrentTaskData(data);
      setTaskLoadState("loaded");
      setTasks((prev) => prev.map((t) => t.id === taskId ? { ...t, name: data.name || t.name } : t));
    }).catch((err) => {
      if (cancelled) return;
      setCurrentTaskData(null);
      setTaskLoadState("failed");
      appendLog("error", "loader", `task${taskId} load failed: ${err.message}`);
    });
    return () => { cancelled = true; };
  }, [taskId]);

  useEffect(() => {
    fetchBestManifest().then(setBestManifest).catch(() => {});
    listImportedTasks().then((items) => {
      setImportedTasks(new Set(items.map((i) => i.taskId)));
    });
    fetchBaselineSummary().then((summary) => {
      if (!summary?.items) return;
      setBestByTask((prev) => {
        const next = { ...prev };
        let added = 0;
        for (const [fullId, eff] of Object.entries(summary.items)) {
          if (eff?.error) continue;
          const tid = fullId.replace(/^task/, "");
          if (next[tid]?.verified) continue;
          if (!next[tid] || eff.cost < next[tid].cost) {
            next[tid] = { ...eff, fromBaseline: true };
            added++;
          }
        }
        if (added > 0) {
          setTimeout(() => appendLog("info", "baselines", `Loaded baseline scores for ${added} task${added === 1 ? "" : "s"} (Σ ${(summary.totalScore || 0).toFixed(1)} / ${(25*400).toFixed(0)})`), 0);
        }
        return next;
      });
    }).catch(() => {});
  }, []);

  useEffect(() => { persistGraphs(wsId, graphsByTask); }, [wsId, graphsByTask]);
  useEffect(() => { persistWorkspaces(workspaces); }, [workspaces]);

  const pairs = useMemo(() => ({
    train: currentTaskData?.train || [],
    test: currentTaskData?.test || [],
    extra: currentTaskData?.["arc-gen"] || currentTaskData?.extra || [],
  }), [currentTaskData]);

  const currentTaskMeta = useMemo(() => tasks.find((t) => t.id === taskId) || { id: taskId, name: "Task" }, [tasks, taskId]);

  const actualKey = `${taskId}:${selectedExample.kind}:${selectedExample.index}`;
  const actualOutput = actualOutputByKey[actualKey];
  const nodeOutputs = nodeOutputsByKey[actualKey] || null;

  const updateGraph = useCallback((updater) => {
    setGraphsByTask((prev) => {
      const next = { ...prev };
      const cur = prev[taskId] || defaultDesignGraph();
      next[taskId] = updater(cur);
      return next;
    });
    setDirty(true);
  }, [taskId]);

  const appendLog = useCallback((level, src, msg) => {
    setLogs((prev) => [...prev, { t: nowStamp(), level, src, msg }]);
    setStatus(msg);
  }, []);

  const selectedNode = graph.nodes.find((n) => n.id === graph.selectedNodeId);

  const selectNode = useCallback((id) => {
    updateGraph((g) => ({ ...g, selectedNodeId: id, selectedEdgeKey: null }));
    if (id) setInspectorOpen(true);
  }, [updateGraph]);

  const selectEdge = useCallback((key) => {
    updateGraph((g) => ({ ...g, selectedEdgeKey: key, selectedNodeId: null }));
  }, [updateGraph]);

  const updateNode = useCallback((id, patch) => {
    updateGraph((g) => ({
      ...g,
      nodes: g.nodes.map((n) => {
        if (n.id !== id) return n;
        return {
          ...n,
          ...patch,
          attrs: patch.attrs !== undefined ? patch.attrs : n.attrs,
        };
      }),
    }));
  }, [updateGraph]);

  const moveNode = useCallback((id, x, y) => {
    updateGraph((g) => ({
      ...g,
      nodes: g.nodes.map((n) => n.id === id ? { ...n, x, y } : n),
    }));
  }, [updateGraph]);

  const deleteNode = useCallback((id) => {
    updateGraph((g) => ({
      ...g,
      nodes: g.nodes.filter((n) => n.id !== id),
      edges: g.edges.filter((e) => !e.from.startsWith(id + ":") && !e.to.startsWith(id + ":")),
      selectedNodeId: g.selectedNodeId === id ? null : g.selectedNodeId,
    }));
    appendLog("info", "editor", `Deleted node ${id}`);
  }, [updateGraph, appendLog]);

  const deleteEdge = useCallback((key) => {
    updateGraph((g) => ({
      ...g,
      edges: g.edges.filter((e) => `${e.from}:${e.fromPort || ""}→${e.to}:${e.toPort || ""}` !== key &&
        !(`${e.from}→${e.to}` === key) &&
        !(`${e.from.split(":")[0]}:${e.from.split(":")[1]}→${e.to.split(":")[0]}:${e.to.split(":")[1]}` === key)),
      selectedEdgeKey: null,
    }));
    appendLog("info", "editor", "Removed edge");
  }, [updateGraph, appendLog]);

  const createEdge = useCallback((from, to) => {
    updateGraph((g) => {
      const exists = g.edges.some((e) => e.from === from && e.to === to);
      if (exists) return g;
      const filtered = g.edges.filter((e) => e.to !== to);
      return { ...g, edges: [...filtered, { id: uid("e"), from, to }] };
    });
    appendLog("ok", "editor", `Connected ${from} → ${to}`);
  }, [updateGraph, appendLog]);

  const addNode = useCallback((spec) => {
    const type = typeof spec === "string" ? spec : spec.type;
    const x = typeof spec === "object" ? spec.x : 300;
    const y = typeof spec === "object" ? spec.y : 200;
    if (!NODE_TYPES[type]) return;
    const id = `${type.toLowerCase()}_${Math.floor(Math.random() * 900 + 100)}`;
    const attrs = defaultAttrsFor(type);
    updateGraph((g) => ({
      ...g,
      nodes: [...g.nodes, { id, type, x, y, attrs, label: id }],
      selectedNodeId: id,
    }));
    appendLog("info", "editor", `Added ${type} node ${id}`);
    setInspectorOpen(true);
  }, [updateGraph, appendLog]);

  const runOnExample = useCallback(async () => {
    if (running) { setRunning(false); return; }
    const pair = pairs[selectedExample.kind]?.[selectedExample.index];
    if (!pair) {
      appendLog("warn", "run", "No example selected.");
      return;
    }
    setRunning(true);
    setRunProgress(0);
    setRunStates({});
    appendLog("info", "run", `Running task${taskId} on ${selectedExample.kind} ${selectedExample.index + 1}…`);
    const order = graph.nodes.map((n) => n.id);
    let idx = 0;
    const tick = setInterval(() => {
      if (idx < order.length) {
        setRunStates((prev) => ({ ...prev, [order[idx]]: "running" }));
        setRunProgress((idx + 1) / order.length);
        idx++;
      } else {
        clearInterval(tick);
      }
    }, 60);

    try {
      const data = await apiRun({
        projectName: `${currentWorkspace.name}-task${taskId}`,
        taskId: `task${taskId}`,
        graph,
        trainingPairs: pairs.train,
        inputGrid: pair.input,
        expectedOutput: pair.output || null,
        traceIntermediates: true,
      });
      clearInterval(tick);
      setRunStates(Object.fromEntries(order.map((id) => [id, "ok"])));
      setRunProgress(1);
      setActualOutputByKey((prev) => ({ ...prev, [actualKey]: data.grid }));
      if (data.nodeOutputs) {
        setNodeOutputsByKey((prev) => ({ ...prev, [actualKey]: data.nodeOutputs }));
      }
      appendLog("ok", "run", `Run produced ${data.shape?.join("×")} output`);
      setValidation((cur) => cur.state === "failed" ? { state: "idle" } : cur);
      setTasks((prev) => prev.map((t) => t.id === taskId ? { ...t, status: "editing" } : t));
    } catch (err) {
      clearInterval(tick);
      setRunStates(Object.fromEntries(order.map((id) => [id, "err"])));
      const reason = extractFailureReason(err);
      appendLog("error", "run", `Run failed: ${reason}`);
      setValidation({ state: "failed", taskId: `task${taskId}`, reason });
      setTasks((prev) => prev.map((t) => t.id === taskId ? { ...t, status: "failing" } : t));
    } finally {
      setRunning(false);
    }
  }, [running, pairs, selectedExample, taskId, graph, currentWorkspace.name, actualKey, appendLog]);

  const exportSubmissionZip = useCallback(async () => {
    const ids = Object.keys(bestGraphByTask).sort();
    if (ids.length === 0) {
      appendLog("warn", "submission", "No verified best graphs yet — verify some tasks first.");
      return;
    }
    appendLog("info", "submission", `Bundling ${ids.length} task${ids.length === 1 ? "" : "s"} as submission zip…`);
    try {
      const response = await apiExportZip(ids.map((id) => ({
        projectName: `${currentWorkspace.name}-task${id}`,
        taskId: `task${id}`,
        graph: bestGraphByTask[id],
        trainingPairs: [],
      })));
      const url = URL.createObjectURL(response.data);
      const a = document.createElement("a");
      a.href = url;
      a.download = "neurogolf-submission.zip";
      document.body.appendChild(a);
      a.click();
      a.remove();
      URL.revokeObjectURL(url);
      appendLog("ok", "submission", `Downloaded submission with ${ids.length} task${ids.length === 1 ? "" : "s"}.`);
    } catch (err) {
      appendLog("error", "submission", `Submission build failed: ${extractFailureReason(err)}`);
    }
  }, [bestGraphByTask, currentWorkspace.name, appendLog]);

  const trySpeculative = useCallback(async (s) => {
    if (!s?.apply) return false;
    if (!pairs.train?.length) {
      appendLog("warn", "simplifier", `Cannot verify "${s.title}" — no training pairs loaded.`);
      return false;
    }
    appendLog("info", "simplifier", `Trying: ${s.title} (verifying on ${pairs.train.length} train pair${pairs.train.length === 1 ? "" : "s"})…`);
    const trial = s.apply(graph);
    try {
      const data = await apiCheck({
        projectName: `${currentWorkspace.name}-task${taskId}-trial`,
        taskId: `task${taskId}`,
        graph: trial,
        trainingPairs: pairs.train,
      });
      if (data?.correct) {
        const before = graph.nodes.length;
        updateGraph(() => trial);
        setHistoryByTask((prev) => ({
          ...prev,
          [taskId]: [
            ...(prev[taskId] || [{ nodeCount: before, label: "start" }]),
            { nodeCount: trial.nodes.length, label: `try: ${s.title}` },
          ],
        }));
        if (data.efficiency) {
          setEfficiencyByTask((p) => ({ ...p, [taskId]: data.efficiency }));
          setBestByTask((p) => {
            const cur = p[taskId];
            if (!cur || data.efficiency.cost < cur.cost) {
              setBestGraphByTask((bg) => ({ ...bg, [taskId]: trial }));
              return { ...p, [taskId]: { ...data.efficiency, verified: true, verifiedAt: Date.now() } };
            }
            return p;
          });
        }
        appendLog("ok", "simplifier", `Verified: ${s.title} (${before} → ${trial.nodes.length} nodes)`);
        return true;
      } else {
        const failed = (data?.pairs || []).filter((p) => !p.ok).length;
        const total = (data?.pairs || []).length;
        appendLog("warn", "simplifier", `Reverted: "${s.title}" failed ${failed}/${total} train pair${total === 1 ? "" : "s"}`);
        return false;
      }
    } catch (err) {
      appendLog("error", "simplifier", `Trial crashed: ${extractFailureReason(err)}`);
      return false;
    }
  }, [graph, taskId, pairs.train, currentWorkspace.name, appendLog, updateGraph]);

  const restoreBest = useCallback(() => {
    const snap = bestGraphByTask[taskId];
    if (!snap) {
      appendLog("warn", "best", "No verified best graph saved yet for this task.");
      return;
    }
    updateGraph(() => ({ ...snap, selectedNodeId: null, selectedEdgeKey: null }));
    appendLog("ok", "best", `Restored best graph for task${taskId}.`);
  }, [taskId, bestGraphByTask, updateGraph, appendLog]);

  const checkAllPairs = useCallback(async () => {
    if (!pairs.train?.length) {
      appendLog("warn", "check", "No training pairs available for check.");
      return null;
    }
    appendLog("info", "check", `Checking ${pairs.train.length} train pair${pairs.train.length === 1 ? "" : "s"}…`);
    try {
      const data = await apiCheck({
        projectName: `${currentWorkspace.name}-task${taskId}`,
        taskId: `task${taskId}`,
        graph,
        trainingPairs: pairs.train,
      });
      const ok = data.correct;
      const passed = (data.pairs || []).filter((p) => p.ok).length;
      const total = (data.pairs || []).length;
      appendLog(ok ? "ok" : "warn", "check", `Correctness: ${passed}/${total} train pair${total === 1 ? "" : "s"}`);
      if (data.efficiency) {
        setEfficiencyByTask((prev) => ({ ...prev, [taskId]: data.efficiency }));
        setEfficiencyErrorByTask((prev) => ({ ...prev, [taskId]: null }));
        if (ok) {
          setBestByTask((prev) => {
            const cur = prev[taskId];
            if (!cur || data.efficiency.cost < cur.cost) {
              setBestGraphByTask((bg) => ({ ...bg, [taskId]: graph }));
              return { ...prev, [taskId]: { ...data.efficiency, verified: true, verifiedAt: Date.now() } };
            }
            return prev;
          });
          setTasks((prev) => prev.map((t) => t.id === taskId ? { ...t, status: "passing", score: data.efficiency.score } : t));
        } else {
          setTasks((prev) => prev.map((t) => t.id === taskId ? { ...t, status: "failing" } : t));
        }
      }
      return data;
    } catch (err) {
      const reason = extractFailureReason(err);
      appendLog("error", "check", `Check failed: ${reason}`);
      return null;
    }
  }, [pairs.train, taskId, graph, currentWorkspace.name, appendLog]);

  const compileOnly = useCallback(async () => {
    appendLog("info", "compile", "Compiling…");
    try {
      const data = await apiCompile({
        projectName: `${currentWorkspace.name}-task${taskId}`,
        taskId: `task${taskId}`,
        graph,
        trainingPairs: pairs.train,
      });
      appendLog("ok", "compile", `Compiled OK: ${data.modelBytes} bytes`);
      if (data.efficiency) {
        setEfficiencyByTask((prev) => ({ ...prev, [taskId]: data.efficiency }));
        setEfficiencyErrorByTask((prev) => ({ ...prev, [taskId]: null }));
      }
    } catch (err) {
      const reason = extractFailureReason(err);
      appendLog("error", "compile", `Compile failed: ${reason}`);
      setValidation({ state: "failed", taskId: `task${taskId}`, reason });
      setEfficiencyErrorByTask((prev) => ({ ...prev, [taskId]: reason }));
    }
  }, [graph, pairs.train, taskId, currentWorkspace.name, appendLog]);

  useEffect(() => {
    if (!graph || !graph.nodes?.length) return;
    let cancelled = false;
    const timer = setTimeout(async () => {
      setEfficiencyRefreshing(true);
      try {
        const data = await apiCompile({
          projectName: `${currentWorkspace.name}-task${taskId}`,
          taskId: `task${taskId}`,
          graph,
          trainingPairs: pairs.train || [],
        });
        if (cancelled) return;
        if (data?.efficiency) {
          setEfficiencyByTask((prev) => ({ ...prev, [taskId]: data.efficiency }));
          setEfficiencyErrorByTask((prev) => ({ ...prev, [taskId]: null }));
          setBaselineByTask((prev) => prev[taskId] ? prev : { ...prev, [taskId]: data.efficiency });
          setBestByTask((prev) => {
            const cur = prev[taskId];
            if (!cur || (data.efficiency.cost < cur.cost)) {
              return { ...prev, [taskId]: data.efficiency };
            }
            return prev;
          });
        }
      } catch (err) {
        if (cancelled) return;
        const reason = extractFailureReason(err);
        setEfficiencyErrorByTask((prev) => ({ ...prev, [taskId]: reason }));
        setEfficiencyByTask((prev) => ({ ...prev, [taskId]: null }));
      } finally {
        if (!cancelled) setEfficiencyRefreshing(false);
      }
    }, 450);
    return () => { cancelled = true; clearTimeout(timer); };
  }, [graph, taskId, pairs.train, currentWorkspace.name]);

  const exportOnnx = useCallback(async () => {
    if (taskLoadState !== "loaded" || !pairs.train.length) {
      appendLog("warn", "export", "Cannot export — no training pairs loaded.");
      return;
    }
    setValidation({ state: "loading", taskId: `task${taskId}` });
    appendLog("info", "export", "Validating & exporting…");
    try {
      const response = await apiExport({
        projectName: `${currentWorkspace.name}-task${taskId}`,
        taskId: `task${taskId}`,
        graph,
        trainingPairs: pairs.train,
      });
      const ct = response.headers["content-type"] || "";
      if (ct.includes("application/json")) {
        const data = JSON.parse(await response.data.text());
        setValidation({ state: "passed", taskId: `task${taskId}`, artifact: data.artifact });
        appendLog("ok", "export", `Validation passed. Pushed ${data.artifact}`);
      } else {
        const url = URL.createObjectURL(response.data);
        const a = document.createElement("a");
        a.href = url;
        a.download = `task${taskId}.onnx`;
        document.body.appendChild(a);
        a.click();
        a.remove();
        URL.revokeObjectURL(url);
        setValidation({ state: "passed", taskId: `task${taskId}`, artifact: `task${taskId}.onnx` });
        appendLog("ok", "export", `Validation passed. Downloaded task${taskId}.onnx`);
      }
      setTasks((prev) => prev.map((t) => t.id === taskId ? { ...t, status: "passing", score: 1 } : t));
      setExportOpen(false);
    } catch (err) {
      let reason = err.message;
      if (err.response?.data) {
        try {
          const text = await err.response.data.text();
          const parsed = JSON.parse(text);
          reason = parsed.reason || parsed.detail?.reason || parsed.detail || reason;
        } catch {}
      }
      if (typeof reason !== "string") reason = JSON.stringify(reason);
      setValidation({ state: "failed", taskId: `task${taskId}`, reason });
      appendLog("error", "export", `Export failed: ${reason}`);
    }
  }, [taskLoadState, pairs.train, taskId, graph, currentWorkspace.name, appendLog]);

  const onImportFiles = useCallback(async (files) => {
    setImportState({ state: "loading" });
    appendLog("info", "import", `Uploading ${files.length} file${files.length === 1 ? "" : "s"}…`);
    try {
      const data = await importOnnxFiles(files);
      const saved = data.saved || [];
      const rejected = data.rejected || [];
      setImportState({ state: "ok", saved, rejected });
      appendLog("ok", "import", `Saved ${saved.length} · Rejected ${rejected.length}`);
      setImportedTasks((cur) => {
        const next = new Set(cur);
        for (const id of saved) next.add(id);
        return next;
      });
      for (const id of saved) {
        try {
          const r = await fetchBestGraph(id);
          setGraphsByTask((prev) => ({ ...prev, [id.replace(/^task/, "")]: { ...r.graph, selectedNodeId: null, selectedEdgeKey: null } }));
        } catch {}
      }
      if (saved.length) {
        const firstSaved = saved[0].replace(/^task/, "");
        setTaskId(firstSaved);
      }
    } catch (err) {
      const reason = extractFailureReason(err);
      setImportState({ state: "failed", reason });
      appendLog("error", "import", `Import failed: ${reason}`);
    }
  }, [appendLog, setTaskId]);

  const loadBestGraphForTask = useCallback(async () => {
    appendLog("info", "loader", `Loading ONNX baseline for task${taskId}…`);
    try {
      const r = await fetchBestGraph(`task${taskId}`);
      updateGraph(() => ({ ...r.graph, selectedNodeId: null, selectedEdgeKey: null }));
      appendLog("ok", "loader", `Loaded baseline (${r.graph.nodes.length} nodes)`);
    } catch (err) {
      appendLog("error", "loader", `Baseline load failed: ${extractFailureReason(err)}`);
    }
  }, [taskId, updateGraph, appendLog]);

  const handleSave = useCallback(() => {
    persistGraphs(wsId, graphsRef.current);
    setDirty(false);
    setWorkspaces((ws) => ws.map((w) => w.id === wsId ? { ...w, lastSaved: humanTime(), tasks: Object.keys(graphsRef.current).length } : w));
    setTasks((prev) => prev.map((t) => t.id === taskId ? { ...t, saved: humanTime() } : t));
    appendLog("ok", "loader", `Saved task${taskId} graph`);
  }, [wsId, taskId, appendLog]);

  const handleSaveAll = useCallback(() => {
    persistGraphs(wsId, graphsRef.current);
    const count = Object.keys(graphsRef.current).length;
    setDirty(false);
    setWorkspaces((ws) => ws.map((w) => w.id === wsId ? { ...w, lastSaved: humanTime(), tasks: count } : w));
    appendLog("ok", "loader", `Saved all ${count} task graphs to workspace '${currentWorkspace.name}'`);
  }, [wsId, currentWorkspace.name, appendLog]);

  const suggestions = useMemo(() => detectSuggestions(graph), [graph]);
  const taskHistory = historyByTask[taskId] || [];

  const applySuggestion = useCallback((s) => {
    if (!s.apply) return;
    updateGraph((g) => {
      const next = s.apply(g);
      setHistoryByTask((prev) => ({
        ...prev,
        [taskId]: [
          ...(prev[taskId] || [{ nodeCount: g.nodes.length, label: "start" }]),
          { nodeCount: next.nodes.length, label: s.title },
        ],
      }));
      return next;
    });
    appendLog("ok", "simplifier", `Applied: ${s.title} (−${s.savings})`);
    setHighlightedNodes([]);
  }, [updateGraph, taskId, appendLog]);

  const applyAllSafe = useCallback(() => {
    updateGraph((g) => {
      let cur = g;
      const startCount = g.nodes.length;
      const steps = [{ nodeCount: startCount, label: "start" }];
      let safety = 30;
      while (safety-- > 0) {
        const remaining = detectSuggestions(cur).filter((x) => x.severity === "safe" && x.apply);
        if (!remaining.length) break;
        const s = remaining[0];
        const stillValid = s.affectedNodes.every((id) => cur.nodes.some((n) => n.id === id));
        if (!stillValid) break;
        cur = s.apply(cur);
        steps.push({ nodeCount: cur.nodes.length, label: s.title });
      }
      setHistoryByTask((prev) => ({ ...prev, [taskId]: steps }));
      appendLog("ok", "simplifier", `Apply All: ${steps.length - 1} safe rewrites · ${startCount} → ${cur.nodes.length} nodes`);
      return cur;
    });
    setHighlightedNodes([]);
  }, [updateGraph, taskId, appendLog]);

  const undoLast = useCallback(() => {
    setHistoryByTask((prev) => ({
      ...prev,
      [taskId]: (prev[taskId] || []).slice(0, -1),
    }));
    appendLog("info", "simplifier", "Undo: popped last history step");
  }, [taskId, appendLog]);

  const pickWorkspace = useCallback((id) => {
    setWorkspaces((ws) => ws.map((w) => ({ ...w, current: w.id === id })));
    setGraphsByTask(loadGraphs(id));
    setActualOutputByKey({});
    setRunStates({});
    appendLog("info", "loader", `Switched to workspace`);
  }, [appendLog]);

  const createWorkspace = useCallback((name) => {
    const id = `ws-${Date.now()}`;
    setWorkspaces((ws) => [...ws.map((w) => ({ ...w, current: false })), { id, name, tasks: 0, lastSaved: "just now", current: true }]);
    setGraphsByTask({});
    appendLog("ok", "loader", `Created workspace '${name}'`);
  }, [appendLog]);

  const renameWorkspace = useCallback((id, name) => {
    setWorkspaces((ws) => ws.map((w) => w.id === id ? { ...w, name } : w));
  }, []);

  const duplicateWorkspace = useCallback((id) => {
    const src = workspaces.find((w) => w.id === id);
    if (!src) return;
    const newId = `ws-${Date.now()}`;
    const copy = { ...src, id: newId, name: src.name + "-copy", current: false, lastSaved: "just now" };
    setWorkspaces((ws) => [...ws, copy]);
    persistGraphs(newId, loadGraphs(id));
    appendLog("ok", "loader", `Duplicated '${src.name}'`);
  }, [workspaces, appendLog]);

  const deleteWorkspace = useCallback((id) => {
    localStorage.removeItem(GRAPHS_KEY_PREFIX + id);
    setWorkspaces((prev) => prev.filter((w) => w.id !== id));
    appendLog("warn", "loader", `Deleted workspace`);
  }, [appendLog]);

  useEffect(() => {
    const onKey = (e) => {
      const cmd = e.metaKey || e.ctrlKey;
      const shift = e.shiftKey;
      const tag = document.activeElement?.tagName;
      const inField = tag === "INPUT" || tag === "TEXTAREA" || document.activeElement?.isContentEditable;
      if (cmd && e.key.toLowerCase() === "k") { e.preventDefault(); setPaletteOpen(true); }
      if (e.key === "Escape") {
        if (fullscreen) setFullscreen(false);
        setPaletteOpen(false); setImportOpen(false); setExportOpen(false); setWorkspacesModal(null);
      }
      if (cmd && shift && e.key.toLowerCase() === "s") { e.preventDefault(); handleSaveAll(); return; }
      if (cmd && e.key.toLowerCase() === "s") { e.preventDefault(); handleSave(); return; }
      if (cmd && e.key.toLowerCase() === "i") { e.preventDefault(); setInspectorOpen((o) => !o); return; }
      if (cmd && e.key.toLowerCase() === "r") { e.preventDefault(); runOnExample(); return; }
      if (cmd && e.key.toLowerCase() === "j") { e.preventDefault(); setBottomCollapsed((c) => !c); return; }
      if (cmd && e.key.toLowerCase() === "f") { e.preventDefault(); setFullscreen((f) => !f); return; }
      if ((e.key === "Delete" || e.key === "Backspace") && !inField) {
        if (graph.selectedEdgeKey) { e.preventDefault(); deleteEdge(graph.selectedEdgeKey); }
        else if (graph.selectedNodeId) { e.preventDefault(); deleteNode(graph.selectedNodeId); }
      }
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [graph.selectedEdgeKey, graph.selectedNodeId, fullscreen, runOnExample, deleteEdge, deleteNode, handleSave, handleSaveAll]);

  const taskName = currentTaskMeta.name;

  return (
    <div style={{ display: "flex", flexDirection: "column", height: "100vh", overflow: "hidden" }}>
      <TopBar
        workspace={currentWorkspace}
        workspaces={workspaces}
        taskId={taskId}
        taskName={taskName}
        dirty={dirty}
        running={running}
        onSave={handleSave}
        onSaveAll={handleSaveAll}
        onRun={runOnExample}
        onCompile={compileOnly}
        onImport={() => setImportOpen(true)}
        onExport={() => setExportOpen(true)}
        onExportZip={exportSubmissionZip}
        submissionCount={Object.keys(bestGraphByTask).length}
        totalScore={Object.values(bestByTask).reduce((s, e) => s + (e?.score || 0), 0)}
        onOpenPalette={() => setPaletteOpen(true)}
        onPickWorkspace={pickWorkspace}
        onOpenWorkspaces={(mode) => setWorkspacesModal(mode)}
      />
      <div style={{ display: "flex", flex: 1, overflow: "hidden", minHeight: 0 }}>
        {!fullscreen && (
          <LeftRail
            tasks={tasks}
            currentTaskId={taskId}
            onSelectTask={setTaskId}
            workspace={currentWorkspace}
            bestByTask={bestByTask}
          />
        )}

        <div style={{ flex: 1, display: "flex", flexDirection: "column", minWidth: 0 }}>
          {!fullscreen && (
            <Workbench
              task={currentTaskMeta}
              pairs={pairs}
              selectedExample={selectedExample}
              onSelectExample={setSelectedExample}
              actualOutput={actualOutput}
              running={running}
              runProgress={runProgress}
              onRun={runOnExample}
              onClearOutput={() => {
                setActualOutputByKey((prev) => { const next = { ...prev }; delete next[actualKey]; return next; });
                setNodeOutputsByKey((prev) => { const next = { ...prev }; delete next[actualKey]; return next; });
                setRunStates({});
                setRunProgress(0);
              }}
            />
          )}

          <div style={{ flex: 1, display: "flex", minHeight: 0, position: "relative", overflow: "hidden" }}>
            <div style={{ flex: 1, position: "relative", minWidth: 0 }}>
              <GraphCanvas
                graph={graph}
                runStates={runStates}
                highlightedNodes={highlightedNodes}
                onSelectNode={selectNode}
                onSelectEdge={selectEdge}
                onUpdateNode={updateNode}
                onMoveNode={moveNode}
                onAddNode={addNode}
                onCreateEdge={createEdge}
                fullscreen={fullscreen}
                onToggleFullscreen={() => setFullscreen((f) => !f)}
              />
              {importedTasks.has(`task${taskId}`) && graph.nodes.length <= 2 && (
                <div style={{
                  position: "absolute", left: 14, top: 14, zIndex: 4,
                  background: "var(--accent-bg)", border: "1px solid var(--accent-dim)",
                  borderRadius: 6, padding: "8px 12px", display: "flex", gap: 8, alignItems: "center",
                  fontSize: 12, color: "var(--accent)",
                }}>
                  <Icon name="zap" size={13} />
                  An ONNX baseline is available for task{taskId}.
                  <button onClick={loadBestGraphForTask} style={{
                    background: "var(--accent)", border: "none", borderRadius: 4,
                    padding: "3px 8px", cursor: "pointer",
                    color: "oklch(0.18 0.02 200)", fontWeight: 600, fontSize: 11.5,
                  }}>Load baseline</button>
                </div>
              )}
            </div>

            {inspectorOpen && (
              <div style={{
                width: 360, flexShrink: 0,
                borderLeft: "1px solid var(--border)",
                background: "var(--bg-surface)",
                display: "flex", flexDirection: "column",
                ...(fullscreen ? {
                  position: "absolute",
                  top: 0, right: 0, bottom: 0,
                  zIndex: 10,
                  boxShadow: "-12px 0 32px -16px oklch(0.05 0 0 / 0.65)",
                } : null),
              }}>
                <div style={{ display: "flex", borderBottom: "1px solid var(--border)", flexShrink: 0 }}>
                  <button onClick={() => setRightTab("inspector")} style={rightTabStyle(rightTab === "inspector")}>
                    <Icon name="sliders" size={12} /> Inspector
                  </button>
                  <button onClick={() => setRightTab("suggestions")} style={rightTabStyle(rightTab === "suggestions")}>
                    <Icon name="sparkle" size={12} /> Simplifier
                    {suggestions.length > 0 && (
                      <span style={{
                        fontFamily: "var(--mono)", fontSize: 10, padding: "1px 5px", borderRadius: 8,
                        background: rightTab === "suggestions" ? "var(--accent-bg)" : "var(--bg-input)",
                        color: rightTab === "suggestions" ? "var(--accent)" : "var(--text-muted)",
                        marginLeft: 2,
                      }}>{suggestions.length}</span>
                    )}
                  </button>
                  <div style={{ flex: 1 }} />
                  <button onClick={() => setInspectorOpen(false)} style={{
                    background: "transparent", border: "none", color: "var(--text-muted)", cursor: "pointer",
                    padding: "0 12px",
                  }} title="Close drawer (⌘I)">
                    <Icon name="x" size={14} />
                  </button>
                </div>

                {rightTab === "inspector" && (
                  <Inspector
                    task={currentTaskMeta}
                    graph={graph}
                    selectedNode={selectedNode}
                    onUpdateNode={updateNode}
                    onDeleteNode={deleteNode}
                    runState={runStates[graph.selectedNodeId]}
                    inputGrid={pairs[selectedExample.kind]?.[selectedExample.index]?.input || null}
                    expectedOutput={pairs[selectedExample.kind]?.[selectedExample.index]?.output || null}
                    actualOutput={actualOutput || null}
                    nodeOutputs={nodeOutputs}
                    running={running}
                  />
                )}
                {rightTab === "suggestions" && (
                  <SuggestionsPanel
                    suggestions={suggestions}
                    history={taskHistory}
                    onHover={setHighlightedNodes}
                    onApply={applySuggestion}
                    onApplyAllSafe={applyAllSafe}
                    onUndo={undoLast}
                    efficiency={efficiencyByTask[taskId] || null}
                    baseline={baselineByTask[taskId] || null}
                    best={bestByTask[taskId] || null}
                    bestGraphAvailable={!!bestGraphByTask[taskId]}
                    efficiencyError={efficiencyErrorByTask[taskId] || null}
                    efficiencyRefreshing={efficiencyRefreshing}
                    onCheck={checkAllPairs}
                    onRestoreBest={restoreBest}
                    onTry={trySpeculative}
                  />
                )}
              </div>
            )}
          </div>

          {!fullscreen && (
            <BottomPanel
              logs={logs}
              validation={validation}
              taskId={taskId}
              graph={graph}
              height={bottomHeight}
              onHeight={setBottomHeight}
              collapsed={bottomCollapsed}
              onToggle={() => setBottomCollapsed((c) => !c)}
              tab={bottomTab}
              onTab={setBottomTab}
              inputGrid={pairs[selectedExample.kind]?.[selectedExample.index]?.input || null}
              expectedOutput={pairs[selectedExample.kind]?.[selectedExample.index]?.output || null}
              actualOutput={actualOutput || null}
              nodeOutputs={nodeOutputs}
              running={running}
            />
          )}

          <StatusBar
            taskId={taskId} graph={graph}
            bottomCollapsed={bottomCollapsed}
            onToggleBottom={() => setBottomCollapsed((c) => !c)}
            inspectorOpen={inspectorOpen}
            onToggleInspector={() => setInspectorOpen((o) => !o)}
            selectedNode={selectedNode}
            fullscreen={fullscreen}
            onToggleFullscreen={() => setFullscreen((f) => !f)}
            suggestionCount={suggestions.length}
            onOpenSimplifier={() => { setInspectorOpen(true); setRightTab("suggestions"); }}
            status={status}
            efficiency={efficiencyByTask[taskId] || null}
            efficiencyError={efficiencyErrorByTask[taskId] || null}
            efficiencyRefreshing={efficiencyRefreshing}
            bestByTask={bestByTask}
          />
        </div>
      </div>

      <CommandPalette
        open={paletteOpen} onClose={() => setPaletteOpen(false)}
        tasks={tasks}
        onSelectTask={setTaskId}
        onAddNode={(t) => addNode(t)}
        onRun={runOnExample}
        onSave={handleSave}
        onSaveAll={handleSaveAll}
        onImport={() => setImportOpen(true)}
        onExport={() => setExportOpen(true)}
        onCompile={compileOnly}
      />
      <ImportModal open={importOpen} onClose={() => setImportOpen(false)}
        onPickFiles={onImportFiles} importState={importState} />
      <ExportModal open={exportOpen} onClose={() => setExportOpen(false)} taskId={taskId} validation={validation} onExport={exportOnnx} />
      <WorkspacesModal
        open={!!workspacesModal} mode={workspacesModal}
        onClose={() => setWorkspacesModal(null)}
        workspaces={workspaces} currentId={currentWorkspace.id}
        onPick={pickWorkspace}
        onCreate={createWorkspace}
        onRename={renameWorkspace}
        onDuplicate={duplicateWorkspace}
        onDelete={deleteWorkspace}
      />
    </div>
  );
}
