import axios from "axios";
import { designGraphToBackend, backendGraphToDesign } from "./data.js";

function payloadFromGraph({ projectName, taskId, graph, trainingPairs = [], extra = {} }) {
  const { nodes, edges } = designGraphToBackend(graph);
  return {
    projectName,
    taskId,
    nodes,
    edges,
    trainingPairs,
    ...extra,
  };
}

export async function fetchTask(taskId) {
  const fname = /^task/.test(taskId) ? `${taskId}.json` : `task${taskId}.json`;
  const r = await fetch(`/tasks/${fname}`, { cache: "no-cache" });
  if (!r.ok) throw new Error(`Task ${taskId} returned HTTP ${r.status}`);
  return r.json();
}

export async function fetchBestManifest() {
  try {
    const r = await fetch("/best/manifest.json", { cache: "no-cache" });
    if (!r.ok) return null;
    return await r.json();
  } catch {
    return null;
  }
}

export async function fetchBestGraph(taskId) {
  const { data } = await axios.get(`/api/best-graph/${taskId}`);
  const graph = backendGraphToDesign(data);
  return { graph, projectName: data.projectName, meta: data.meta };
}

export async function compileGraph({ projectName, taskId, graph, trainingPairs }) {
  const payload = payloadFromGraph({ projectName, taskId, graph, trainingPairs });
  const { data } = await axios.post("/api/compile", payload);
  return data;
}

export async function checkCorrectness({ projectName, taskId, graph, trainingPairs }) {
  const payload = payloadFromGraph({ projectName, taskId, graph, trainingPairs });
  const { data } = await axios.post("/api/check", payload);
  return data;
}

export async function runGraph({ projectName, taskId, graph, trainingPairs, inputGrid, expectedOutput, traceIntermediates = true }) {
  const payload = payloadFromGraph({
    projectName,
    taskId,
    graph,
    trainingPairs,
    extra: { inputGrid, expectedOutput, traceIntermediates },
  });
  const { data } = await axios.post("/api/run", payload);
  return data;
}

export async function exportGraph({ projectName, taskId, graph, trainingPairs }) {
  const payload = payloadFromGraph({ projectName, taskId, graph, trainingPairs });
  const response = await axios.post("/api/export", payload, { responseType: "blob" });
  return response;
}

export async function exportZip(tasks) {
  const payload = {
    tasks: tasks.map(({ projectName, taskId, graph, trainingPairs }) => ({
      ...payloadFromGraph({ projectName, taskId, graph, trainingPairs }),
    })),
  };
  const response = await axios.post("/api/export-zip", payload, { responseType: "blob" });
  return response;
}

export async function fetchBaselineSummary() {
  try {
    const { data } = await axios.get("/api/baselines-summary");
    return data;
  } catch {
    return { items: {}, count: 0, totalScore: 0 };
  }
}

export async function listImportedTasks() {
  try {
    const { data } = await axios.get("/api/import/list");
    return data.items || [];
  } catch {
    return [];
  }
}

export async function importOnnxFiles(files) {
  const form = new FormData();
  for (const f of files) form.append("files", f, f.name);
  const { data } = await axios.post("/api/import", form);
  return data;
}

export function extractFailureReason(err) {
  const resp = err?.response?.data;
  const reason = resp?.reason || resp?.detail?.reason || resp?.detail || err?.message;
  return typeof reason === "string" ? reason : JSON.stringify(reason);
}
