export const ARC_COLORS = [
  "#0b0b0d", "#3677e8", "#e84a3e", "#3ec872", "#f0d040",
  "#8a8a8a", "#d44ec7", "#f08732", "#7dd0f7", "#8a2a2a",
];

export const ARC_COLOR_NAMES = [
  "black", "blue", "red", "green", "yellow",
  "gray", "magenta", "orange", "sky", "maroon",
];

export const TASK_COUNT = 400;

export const NODE_GROUP_COLORS = {
  accent:  { bar: "var(--accent)",          bg: "oklch(0.27 0.06 200)" },
  math:    { bar: "oklch(0.78 0.13 75)",    bg: "oklch(0.27 0.05 75)"  },
  logic:   { bar: "oklch(0.75 0.14 330)",   bg: "oklch(0.27 0.05 330)" },
  reduce:  { bar: "oklch(0.78 0.13 155)",   bg: "oklch(0.27 0.05 155)" },
  shape:   { bar: "oklch(0.78 0.13 260)",   bg: "oklch(0.27 0.05 260)" },
  neutral: { bar: "oklch(0.55 0.012 250)",  bg: "oklch(0.245 0.008 250)" },
};

const intAttr = (def) => ({ type: "int", default: def });
const enumAttr = (options, def) => ({ type: "enum", options, default: def });
const boolAttr = (def) => ({ type: "bool", default: def });
const shapeAttr = (def) => ({ type: "shape", default: def });
const tensorAttr = (def) => ({ type: "tensor", default: def });
const strAttr = (def) => ({ type: "string", default: def });

export const NODE_TYPES = {
  Input:        { color: "accent",  ins: [],                          outs: ["x"],     attrs: { shape: shapeAttr([1, 1, 30, 30]) } },
  Output:       { color: "accent",  ins: ["input"],                   outs: [],        attrs: { shape: shapeAttr([1, 1, 30, 30]) } },
  Constant:     { color: "neutral", ins: [],                          outs: ["y"],     attrs: { value: tensorAttr("0") } },
  RowIndex:     { color: "neutral", ins: [],                          outs: ["y"],     attrs: {} },
  ColIndex:     { color: "neutral", ins: [],                          outs: ["y"],     attrs: {} },
  Cast:         { color: "neutral", ins: ["input"],                   outs: ["y"],     attrs: { to: strAttr("1") } },
  Identity:     { color: "neutral", ins: ["input"],                   outs: ["y"],     attrs: {} },
  Reshape:      { color: "neutral", ins: ["data", "shape"],           outs: ["y"],     attrs: {} },
  Squeeze:      { color: "neutral", ins: ["input"],                   outs: ["y"],     attrs: {} },
  Unsqueeze:    { color: "neutral", ins: ["input"],                   outs: ["y"],     attrs: { axes: shapeAttr([1]) } },
  Expand:       { color: "neutral", ins: ["input", "shape"],          outs: ["y"],     attrs: {} },
  Transpose:    { color: "neutral", ins: ["input"],                   outs: ["y"],     attrs: { perm: shapeAttr([0, 1, 3, 2]) } },
  Slice:        { color: "neutral", ins: ["input"],                   outs: ["y"],     attrs: { starts: shapeAttr([0,0,0,0]), ends: shapeAttr([1,1,30,30]), axes: shapeAttr([0,1,2,3]), steps: shapeAttr([1,1,1,1]) } },
  Pad:          { color: "neutral", ins: ["input"],                   outs: ["y"],     attrs: { pads: shapeAttr([0,0,0,0,0,0,0,0]), value: intAttr(0) } },
  Concat:       { color: "neutral", ins: ["a", "b"],                  outs: ["y"],     attrs: { axis: intAttr(1) } },
  Gather:       { color: "neutral", ins: ["data", "indices"],         outs: ["y"],     attrs: { axis: intAttr(0) } },
  OneHot:       { color: "neutral", ins: ["indices","depth","values"],outs: ["y"],     attrs: { axis: intAttr(-1) } },
  Tile:         { color: "shape",   ins: ["input"],                   outs: ["y"],     attrs: { repeats: shapeAttr([1,1,1,1]) } },
  Resize:       { color: "shape",   ins: ["input"],                   outs: ["y"],     attrs: { sizes: shapeAttr([1,1,30,30]), mode: enumAttr(["nearest","linear","cubic"], "nearest") } },
  Add:          { color: "math",    ins: ["a", "b"],                  outs: ["y"],     attrs: {} },
  Sub:          { color: "math",    ins: ["a", "b"],                  outs: ["y"],     attrs: {} },
  Mul:          { color: "math",    ins: ["a", "b"],                  outs: ["y"],     attrs: {} },
  Div:          { color: "math",    ins: ["a", "b"],                  outs: ["y"],     attrs: {} },
  Mod:          { color: "math",    ins: ["a", "b"],                  outs: ["y"],     attrs: {} },
  Min:          { color: "math",    ins: ["a", "b"],                  outs: ["y"],     attrs: {} },
  Max:          { color: "math",    ins: ["a", "b"],                  outs: ["y"],     attrs: {} },
  Sum:          { color: "math",    ins: ["a", "b"],                  outs: ["y"],     attrs: {} },
  MatMul:       { color: "math",    ins: ["a", "b"],                  outs: ["y"],     attrs: {} },
  Relu:         { color: "math",    ins: ["input"],                   outs: ["y"],     attrs: {} },
  Abs:          { color: "math",    ins: ["input"],                   outs: ["y"],     attrs: {} },
  Neg:          { color: "math",    ins: ["input"],                   outs: ["y"],     attrs: {} },
  Floor:        { color: "math",    ins: ["input"],                   outs: ["y"],     attrs: {} },
  Sign:         { color: "math",    ins: ["input"],                   outs: ["y"],     attrs: {} },
  Sqrt:         { color: "math",    ins: ["input"],                   outs: ["y"],     attrs: {} },
  Clip:         { color: "math",    ins: ["input"],                   outs: ["y"],     attrs: { min: intAttr(0), max: intAttr(9) } },
  Equal:        { color: "logic",   ins: ["a", "b"],                  outs: ["y"],     attrs: {} },
  Greater:      { color: "logic",   ins: ["a", "b"],                  outs: ["y"],     attrs: {} },
  Less:         { color: "logic",   ins: ["a", "b"],                  outs: ["y"],     attrs: {} },
  GreaterOrEqual:{color: "logic",   ins: ["a", "b"],                  outs: ["y"],     attrs: {} },
  LessOrEqual:  { color: "logic",   ins: ["a", "b"],                  outs: ["y"],     attrs: {} },
  Not:          { color: "logic",   ins: ["input"],                   outs: ["y"],     attrs: {} },
  And:          { color: "logic",   ins: ["a", "b"],                  outs: ["y"],     attrs: {} },
  Or:           { color: "logic",   ins: ["a", "b"],                  outs: ["y"],     attrs: {} },
  Xor:          { color: "logic",   ins: ["a", "b"],                  outs: ["y"],     attrs: {} },
  Where:        { color: "logic",   ins: ["condition","true","false"],outs: ["y"],     attrs: {} },
  ReduceSum:    { color: "reduce",  ins: ["input"],                   outs: ["y"],     attrs: { axes: shapeAttr([2,3]), keepdims: intAttr(1) } },
  ReduceMax:    { color: "reduce",  ins: ["input"],                   outs: ["y"],     attrs: { axes: shapeAttr([2,3]), keepdims: intAttr(1) } },
  ReduceMin:    { color: "reduce",  ins: ["input"],                   outs: ["y"],     attrs: { axes: shapeAttr([2,3]), keepdims: intAttr(1) } },
  ArgMax:       { color: "reduce",  ins: ["input"],                   outs: ["y"],     attrs: { axis: intAttr(1), keepdims: intAttr(1) } },
  Conv:         { color: "shape",   ins: ["input"],                   outs: ["y"],     attrs: { weight_shape: shapeAttr([1,1,3,3]), weights: tensorAttr("[1,1,1,1,1,1,1,1,1]"), pads: shapeAttr([1,1,1,1]), strides: shapeAttr([1,1]) } },
};

export const ORDER_GROUPS = [
  ["accent", "Graph I/O"],
  ["math", "Math"],
  ["logic", "Logic"],
  ["reduce", "Reduce"],
  ["shape", "Shape"],
  ["neutral", "Tensor"],
];

const BACKEND_INPUT_SLOTS = {
  Cast: ["input"],
  Identity: ["input"],
  Not: ["input"],
  ReduceSum: ["input"],
  ReduceMax: ["input"],
  ReduceMin: ["input"],
  ArgMax: ["input"],
  Slice: ["input"],
  Pad: ["input"],
  Transpose: ["input"],
  Tile: ["input"],
  Resize: ["input"],
  Conv: ["input"],
  Output: ["input"],
  Relu: ["input"],
  Abs: ["input"],
  Neg: ["input"],
  Floor: ["input"],
  Clip: ["input"],
  Sign: ["input"],
  Sqrt: ["input"],
  Squeeze: ["input"],
  Unsqueeze: ["input"],
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
  Where: ["condition", "true", "false"],
  Concat: ["a", "b"],
  Gather: ["data", "indices"],
  Reshape: ["data", "shape"],
  Expand: ["input", "shape"],
  OneHot: ["indices", "depth", "values"],
  MatMul: ["a", "b"],
};

export function backendInputSlots(type) {
  if (BACKEND_INPUT_SLOTS[type]) return BACKEND_INPUT_SLOTS[type];
  if (type === "Input" || type === "Constant" || type === "RowIndex" || type === "ColIndex") return [];
  const def = NODE_TYPES[type];
  if (def) return def.ins;
  return ["input"];
}

export const STATUSES = ["passing", "failing", "editing", "untested"];

export function makeTaskRoster(count = TASK_COUNT) {
  const out = [];
  for (let i = 1; i <= count; i++) {
    const id = String(i).padStart(3, "0");
    out.push({
      id,
      name: `Task ${id}`,
      status: "untested",
      score: null,
      saved: null,
    });
  }
  return out;
}

export function defaultAttrsFor(type) {
  const def = NODE_TYPES[type];
  if (!def) return {};
  const out = {};
  for (const [k, schema] of Object.entries(def.attrs)) {
    out[k] = clone(schema.default);
  }
  return out;
}

export function clone(v) {
  return v == null ? v : JSON.parse(JSON.stringify(v));
}

export function uid(prefix) {
  return `${prefix}_${Math.random().toString(36).slice(2, 7)}`;
}

const ATTR_KEYS_GUI_INTERNAL = new Set([
  "source", "size", "valuesPreview",
]);

export function designNodeToBackend(node) {
  const def = NODE_TYPES[node.type] || { ins: [] };
  const attrs = {};
  let value;
  let values;
  let dataType;
  let shape;
  for (const [k, v] of Object.entries(node.attrs || {})) {
    if (node.type === "Constant") {
      if (k === "value") { value = typeof v === "string" ? v : JSON.stringify(v); continue; }
      if (k === "values") { values = v; continue; }
      if (k === "dataType") { dataType = v; continue; }
      if (k === "shape") { shape = Array.isArray(v) ? v.join(",") : String(v); continue; }
    }
    if (k === "shape" && (node.type === "Input" || node.type === "Output")) {
      shape = Array.isArray(v) ? v.join(",") : String(v);
      continue;
    }
    if (ATTR_KEYS_GUI_INTERNAL.has(k)) continue;
    attrs[k] = v;
  }
  const data = {
    label: node.label || node.id,
    opType: node.type,
    value,
    values,
    dataType,
    attrs,
    outputCount: node.outputCount,
    inputSlots: backendInputSlots(node.type),
  };
  if (shape !== undefined) data.shape = shape;
  else if (node.type === "Input" || node.type === "Output") data.shape = "1,1,30,30";
  return {
    id: node.id,
    type: "op",
    position: { x: node.x, y: node.y },
    data,
  };
}

export function designEdgeToBackend(edge, index) {
  const [from, fromPort] = edge.from.split(":");
  const [to, toPort] = edge.to.split(":");
  return {
    id: edge.id || `e_${from}_${to}_${toPort}_${index}`,
    source: from,
    sourceHandle: fromPort || undefined,
    target: to,
    targetHandle: toPort || "input",
  };
}

function parseShape(value) {
  if (Array.isArray(value)) return value.map((v) => Number(v) || 0);
  if (typeof value === "string" && value.trim().length) {
    return value.split(",").map((v) => Number(v.trim()) || 0);
  }
  return [1, 1, 30, 30];
}

function parseAttrs(value) {
  if (!value) return {};
  if (typeof value === "object") return value;
  try {
    const parsed = JSON.parse(value);
    return parsed && typeof parsed === "object" ? parsed : {};
  } catch {
    return {};
  }
}

export function backendNodeToDesign(node) {
  const d = node.data || {};
  const type = d.opType || node.type || "Identity";
  const attrs = parseAttrs(d.attrs || d.attrsText || {});
  if (type === "Input" || type === "Output") {
    attrs.shape = parseShape(d.shape);
  }
  if (type === "Constant") {
    if (d.values !== undefined) {
      attrs.values = d.values;
    } else if (d.value !== undefined) {
      attrs.value = typeof d.value === "string" ? d.value : JSON.stringify(d.value);
    }
    if (d.dataType !== undefined) attrs.dataType = d.dataType;
    if (d.shape !== undefined) attrs.shape = d.shape;
  }
  return {
    id: node.id,
    type,
    x: node.position?.x ?? 0,
    y: node.position?.y ?? 0,
    label: d.label || node.id,
    attrs,
    outputCount: d.outputCount,
    inputSlots: Array.isArray(d.inputSlots) && d.inputSlots.length > 0 ? d.inputSlots : undefined,
  };
}

export function effectiveInputPorts(node) {
  if (node?.inputSlots && node.inputSlots.length > 0) return node.inputSlots;
  const def = NODE_TYPES[node?.type];
  return def?.ins || [];
}

export function effectiveOutputPorts(node) {
  if (node?.outputCount && node.outputCount > 1) {
    return Array.from({ length: node.outputCount }, (_, i) => `out${i}`);
  }
  const def = NODE_TYPES[node?.type];
  return def?.outs || [];
}

export function backendEdgeToDesign(edge) {
  const fromPort = edge.sourceHandle || "y";
  const toPort = edge.targetHandle || "input";
  return {
    id: edge.id,
    from: `${edge.source}:${fromPort}`,
    to: `${edge.target}:${toPort}`,
  };
}

export function designGraphToBackend(graph) {
  return {
    nodes: graph.nodes.map(designNodeToBackend),
    edges: graph.edges.map(designEdgeToBackend),
  };
}

export function backendGraphToDesign(payload) {
  return {
    nodes: (payload.nodes || []).map(backendNodeToDesign),
    edges: (payload.edges || []).map(backendEdgeToDesign),
    selectedNodeId: null,
    selectedEdgeKey: null,
  };
}

export function defaultDesignGraph() {
  return {
    nodes: [
      { id: "input_1", type: "Input", x: 80, y: 200, label: "input_1", attrs: { shape: [1, 1, 30, 30] } },
      { id: "output_1", type: "Output", x: 420, y: 200, label: "output_1", attrs: { shape: [1, 1, 30, 30] } },
    ],
    edges: [{ id: "e0", from: "input_1:x", to: "output_1:input" }],
    selectedNodeId: null,
    selectedEdgeKey: null,
  };
}
