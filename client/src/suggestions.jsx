import React from "react";
import { Icon, Button, Pill, SectionLabel } from "./components.jsx";
import { NODE_TYPES } from "./data.js";

function parseMaybeJson(v) {
  if (v == null) return v;
  if (typeof v !== "string") return v;
  try { return JSON.parse(v); } catch { return v; }
}

function constArray(node) {
  if (!node || node.type !== "Constant") return null;
  if (Array.isArray(node.attrs?.values)) return node.attrs.values.slice();
  const v = parseMaybeJson(node.attrs?.value);
  if (Array.isArray(v)) {
    const flat = v.flat ? v.flat(10) : v;
    return Array.isArray(flat) ? flat.slice() : [Number(v)];
  }
  if (typeof v === "number" || typeof v === "boolean") return [Number(v)];
  return null;
}

function constShape(node) {
  if (!node || node.type !== "Constant") return null;
  const raw = node.attrs?.shape;
  if (Array.isArray(raw)) return raw.map(Number);
  if (typeof raw === "string" && raw.trim().length) {
    return raw.split(",").map((x) => Number(x.trim())).filter((x) => Number.isFinite(x));
  }
  const flat = constArray(node);
  if (flat) return [flat.length];
  return null;
}

function constScalarValue(node) {
  const arr = constArray(node);
  if (!arr || arr.length === 0) return null;
  const first = Number(arr[0]);
  if (!arr.every((x) => Number(x) === first)) return null;
  return first;
}

function isConstantValueEqualTo(rawValue, target) {
  let v = rawValue;
  if (typeof v === "string") {
    try { v = JSON.parse(v); } catch { return false; }
  }
  if (v === target) return true;
  if (Array.isArray(v)) {
    const flat = v.flat ? v.flat(10) : v;
    return Array.isArray(flat) && flat.length > 0 && flat.every((x) => Number(x) === target);
  }
  return false;
}

function isConstantAllEqualTo(node, target) {
  const arr = constArray(node);
  if (!arr || arr.length === 0) return false;
  return arr.every((x) => Number(x) === target);
}

function arraysShallowEqual(a, b) {
  if (!Array.isArray(a) || !Array.isArray(b)) return false;
  if (a.length !== b.length) return false;
  return a.every((v, i) => Number(v) === Number(b[i]));
}

function isIdentityPerm(perm) {
  if (!Array.isArray(perm) || perm.length === 0) return false;
  for (let i = 0; i < perm.length; i++) if (Number(perm[i]) !== i) return false;
  return true;
}

function composePerm(outer, inner) {
  if (!Array.isArray(outer) || !Array.isArray(inner)) return null;
  if (outer.length !== inner.length) return null;
  return outer.map((idx) => Number(inner[Number(idx)]));
}

function graphAdjacency(graph) {
  const incoming = {}, outgoing = {};
  graph.edges.forEach((e) => {
    const fromId = e.from.split(":")[0];
    const toId = e.to.split(":")[0];
    (outgoing[fromId] = outgoing[fromId] || []).push(e);
    (incoming[toId] = incoming[toId] || []).push(e);
  });
  return { incoming, outgoing };
}

function inputEdgeOnPort(incoming, nodeId, suffix) {
  return (incoming[nodeId] || []).find((e) => e.to.endsWith(`:${suffix}`));
}

function inputSourceOnPort(byId, incoming, nodeId, suffix) {
  const e = inputEdgeOnPort(incoming, nodeId, suffix);
  if (!e) return null;
  return { edge: e, node: byId[e.from.split(":")[0]] };
}

function sourceUsedElsewhere(graph, sourceId, currentEdge) {
  return graph.edges.some((e2) =>
    e2.from.startsWith(sourceId + ":") && e2 !== currentEdge
  );
}

function applyRewireThrough(g, removeId, replacementSource) {
  const newEdges = [];
  g.edges.forEach((e) => {
    const [fromId] = e.from.split(":");
    const [toId] = e.to.split(":");
    if (fromId === removeId) newEdges.push({ ...e, from: replacementSource });
    else if (toId === removeId) { /* drop */ }
    else newEdges.push(e);
  });
  return {
    ...g,
    nodes: g.nodes.filter((n) => n.id !== removeId),
    edges: newEdges,
    selectedNodeId: g.selectedNodeId === removeId ? null : g.selectedNodeId,
  };
}

function applyRemoveNodes(g, idsToRemove) {
  const idSet = new Set(idsToRemove);
  return {
    ...g,
    nodes: g.nodes.filter((n) => !idSet.has(n.id)),
    edges: g.edges.filter((e) => {
      const f = e.from.split(":")[0];
      const t = e.to.split(":")[0];
      return !idSet.has(f) && !idSet.has(t);
    }),
    selectedNodeId: idSet.has(g.selectedNodeId) ? null : g.selectedNodeId,
  };
}

function applyMergeConstants(g, survivorId, duplicateIds) {
  const dup = new Set(duplicateIds);
  const survivor = g.nodes.find((n) => n.id === survivorId);
  const def = NODE_TYPES[survivor?.type] || { outs: ["y"] };
  const port = def.outs[0] || "y";
  const newEdges = g.edges.map((e) => {
    const [fromId] = e.from.split(":");
    if (dup.has(fromId)) return { ...e, from: `${survivorId}:${port}` };
    return e;
  });
  return {
    ...g,
    nodes: g.nodes.filter((n) => !dup.has(n.id)),
    edges: newEdges,
  };
}

function applyReplaceWithConstant(g, removeId, newConstNode) {
  const target = g.nodes.find((n) => n.id === removeId);
  const newId = newConstNode.id;
  const port = "y";
  const newEdges = g.edges
    .filter((e) => e.to.split(":")[0] !== removeId)
    .map((e) => {
      const [fromId] = e.from.split(":");
      if (fromId === removeId) return { ...e, from: `${newId}:${port}` };
      return e;
    });
  return {
    ...g,
    nodes: g.nodes.filter((n) => n.id !== removeId).concat([newConstNode]),
    edges: newEdges,
    selectedNodeId: g.selectedNodeId === removeId ? null : g.selectedNodeId,
  };
}

function applyReplaceNode(g, oldNode, newNode, keepInputsFromPort = null) {
  const newId = newNode.id;
  const port = "y";
  const newEdges = g.edges
    .map((e) => {
      const [fromId] = e.from.split(":");
      const [toId] = e.to.split(":");
      if (fromId === oldNode.id) return { ...e, from: `${newId}:${port}` };
      if (toId === oldNode.id) {
        if (keepInputsFromPort && !e.to.endsWith(`:${keepInputsFromPort}`)) return null;
        return { ...e, to: `${newId}:${e.to.split(":")[1]}` };
      }
      return e;
    })
    .filter(Boolean);
  return {
    ...g,
    nodes: g.nodes.map((n) => n.id === oldNode.id ? newNode : n),
    edges: newEdges,
  };
}

function makeConstant(idHint, valuesFlat, shape, dataType = 1) {
  const id = `${idHint}_const_${Math.random().toString(36).slice(2, 7)}`;
  return {
    id,
    type: "Constant",
    x: 0,
    y: 0,
    label: id,
    attrs: {
      values: valuesFlat.slice(),
      shape: shape.slice(),
      dataType,
    },
  };
}

function isAllZeros(arr) {
  return Array.isArray(arr) && arr.length > 0 && arr.every((x) => Number(x) === 0);
}

function isAllOnes(arr) {
  return Array.isArray(arr) && arr.length > 0 && arr.every((x) => Number(x) === 1);
}

function shapesEqual(a, b) {
  if (!Array.isArray(a) || !Array.isArray(b)) return false;
  if (a.length !== b.length) return false;
  return a.every((v, i) => Number(v) === Number(b[i]));
}

function inputShapeOf(node, byId, incoming) {
  const e = (incoming[node.id] || [])[0];
  if (!e) return null;
  const src = byId[e.from.split(":")[0]];
  if (src?.type === "Input" || src?.type === "Output") {
    const raw = src.attrs?.shape;
    if (Array.isArray(raw)) return raw.map(Number);
    if (typeof raw === "string") return raw.split(",").map((x) => Number(x.trim()));
  }
  return null;
}

function broadcastShape(a, b) {
  if (!Array.isArray(a) || !Array.isArray(b)) return null;
  const n = Math.max(a.length, b.length);
  const out = [];
  for (let i = 0; i < n; i++) {
    const ax = a[a.length - 1 - i];
    const bx = b[b.length - 1 - i];
    const da = ax == null ? 1 : Number(ax);
    const db = bx == null ? 1 : Number(bx);
    if (da === db) out.unshift(da);
    else if (da === 1) out.unshift(db);
    else if (db === 1) out.unshift(da);
    else return null;
  }
  return out;
}

// Sinks that propagate dtype unchanged (input data passes through). A Constant
// feeding only these can safely change dtype. Derived empirically from 7912
// verify attempts: every commit-positive bucket of constant-downcast had
// consumers exclusively in this allowlist (Gather.in0, Reshape.in0).
const DTYPE_PASSTHROUGH_PORTS = new Set([
  "Gather.in0",
  "Reshape.in0",
  "Transpose.in0",
  "Squeeze.in0",
  "Unsqueeze.in0",
  "Tile.in0",
  "Identity.input",
  "Cast.input",
  "Output.input",
]);

// Ops that produce bool output. A Cast from one of these to a numeric type
// cannot be dropped because the downstream op rejects bool inputs.
const BOOL_PRODUCING_OPS = new Set([
  "Equal", "Greater", "Less", "GreaterOrEqual", "LessOrEqual",
  "And", "Or", "Not", "Xor",
]);

// Sources from which Cast has never been droppable across 1941 measured
// attempts (Conv/ConvTranspose output precise floats; Unsqueeze/CumSum/Sum/
// GatherND/ReduceSum mismatch downstream dtype expectations).
const CAST_UNDROPPABLE_SOURCES = new Set([
  "Conv", "ConvTranspose",
  "CumSum", "Sum",
  "GatherND",
  "ReduceSum", "ReduceMax", "ReduceMin",
  "Unsqueeze",
]);

// Sinks that require a tensor of a specific shape/rank — a scalar broadcast
// collapse there breaks the operator. From 301 verify attempts: every 0%
// bucket of scalar-broadcast-collapse fell here.
const SHAPE_RIGID_PORTS = new Set([
  "Conv.in1", "Conv.in2",
  "ScatterElements.in1", "ScatterElements.in2",
  "MatMul.in0", "MatMul.in1",
]);

function consumerPortSig(graph, nodeId) {
  const sigs = [];
  for (const e of graph.edges) {
    if (e.from.split(":")[0] !== nodeId) continue;
    const toId = e.to.split(":")[0];
    const toPort = e.to.split(":")[1] || "in0";
    const toNode = graph.nodes.find((x) => x.id === toId);
    if (!toNode) continue;
    sigs.push(`${toNode.type}.${toPort}`);
  }
  return sigs;
}

function consumerTypes(graph, nodeId) {
  const types = [];
  for (const e of graph.edges) {
    if (e.from.split(":")[0] !== nodeId) continue;
    const toNode = graph.nodes.find((x) => x.id === e.to.split(":")[0]);
    if (toNode) types.push(toNode.type);
  }
  return types;
}

export function detectSuggestions(graph) {
  const out = [];
  const byId = Object.fromEntries(graph.nodes.map((n) => [n.id, n]));
  const { incoming, outgoing } = graphAdjacency(graph);

  const outputs = graph.nodes.filter((n) => n.type === "Output").map((n) => n.id);
  const reachable = new Set();
  const visit = (id) => {
    if (reachable.has(id)) return;
    reachable.add(id);
    (incoming[id] || []).forEach((e) => visit(e.from.split(":")[0]));
  };
  outputs.forEach(visit);
  const dead = graph.nodes.filter((n) => n.type !== "Output" && !reachable.has(n.id));
  if (dead.length > 0) {
    out.push({
      id: `dead-${dead.map((d) => d.id).join("_")}`,
      rule: "dead-nodes",
      title: `${dead.length} dead node${dead.length > 1 ? "s" : ""} not connected to output`,
      detail: `Drops ${dead.slice(0, 3).map((d) => d.id).join(", ")}${dead.length > 3 ? `… and ${dead.length - 3} more` : ""}`,
      category: "dead-code",
      severity: "safe",
      savings: dead.length,
      affectedNodes: dead.map((n) => n.id),
      removedNodes: dead.map((n) => n.id),
      apply: (g) => applyRemoveNodes(g, dead.map((n) => n.id)),
    });
  }

  graph.nodes.forEach((n) => {
    if (!["Mul", "Add", "Sub", "Div"].includes(n.type)) return;
    const inEdges = incoming[n.id] || [];
    inEdges.forEach((e) => {
      const srcId = e.from.split(":")[0];
      const src = byId[srcId];
      if (src?.type !== "Constant") return;
      const onB = e.to.endsWith(":b");
      const isIdentity =
        (n.type === "Mul" && isConstantAllEqualTo(src, 1)) ||
        (n.type === "Add" && isConstantAllEqualTo(src, 0)) ||
        (n.type === "Sub" && onB && isConstantAllEqualTo(src, 0)) ||
        (n.type === "Div" && onB && isConstantAllEqualTo(src, 1));
      if (!isIdentity) return;
      const otherEdge = inEdges.find((e2) => e2 !== e);
      if (!otherEdge) return;
      const constHasOtherUsers = sourceUsedElsewhere(graph, srcId, e);
      out.push({
        id: `id-${n.type}-${n.id}`,
        rule: "identity-arith",
        title: `${n.type} by ${n.type === "Mul" || n.type === "Div" ? "1" : "0"} (${n.id})`,
        detail: `Passthrough from ${otherEdge.from}${constHasOtherUsers ? "" : `; drops ${srcId}`}`,
        category: "algebraic",
        severity: "safe",
        savings: constHasOtherUsers ? 1 : 2,
        affectedNodes: constHasOtherUsers ? [n.id] : [n.id, srcId],
        removedNodes: constHasOtherUsers ? [n.id] : [n.id, srcId],
        apply: (g) => {
          let g2 = applyRewireThrough(g, n.id, otherEdge.from);
          if (!constHasOtherUsers) g2 = applyRemoveNodes(g2, [srcId]);
          return g2;
        },
      });
    });
  });

  graph.nodes.forEach((n) => {
    if (n.type !== "Mul") return;
    const inEdges = incoming[n.id] || [];
    for (const e of inEdges) {
      const src = byId[e.from.split(":")[0]];
      if (src?.type !== "Constant") continue;
      if (!isConstantAllEqualTo(src, 0)) continue;
      const constShapeArr = constShape(src) || [1];
      const newConst = makeConstant(n.id, [0], constShapeArr, src.attrs?.dataType ?? 1);
      out.push({
        id: `mul0-${n.id}`,
        rule: "mul-by-zero",
        title: `Mul by 0 collapses to zero (${n.id})`,
        detail: `Replaces with single zero Constant`,
        category: "algebraic",
        severity: "safe",
        savings: 1,
        affectedNodes: [n.id],
        removedNodes: [n.id],
        apply: (g) => applyReplaceWithConstant(g, n.id, newConst),
      });
      break;
    }
  });

  graph.nodes.forEach((n) => {
    if (!["Sub", "Min", "Max"].includes(n.type)) return;
    const inEdges = incoming[n.id] || [];
    const aE = inEdges.find((e) => e.to.endsWith(":a"));
    const bE = inEdges.find((e) => e.to.endsWith(":b"));
    if (!aE || !bE) return;
    if (aE.from !== bE.from) return;
    if (n.type === "Sub") {
      const sourceShape = inputShapeOf(n, byId, incoming) || [1, 1, 30, 30];
      const newConst = makeConstant(n.id, [0], sourceShape, 1);
      out.push({
        id: `sub-self-${n.id}`,
        rule: "self-sub",
        title: `Sub(x, x) is zero (${n.id})`,
        detail: `Replaces with zero Constant`,
        category: "algebraic",
        severity: "safe",
        savings: 1,
        affectedNodes: [n.id],
        removedNodes: [n.id],
        apply: (g) => applyReplaceWithConstant(g, n.id, newConst),
      });
    } else {
      out.push({
        id: `idem-${n.type}-${n.id}`,
        rule: "idempotent-binary",
        title: `${n.type}(x, x) is x (${n.id})`,
        detail: `Passthrough from ${aE.from}`,
        category: "algebraic",
        severity: "safe",
        savings: 1,
        affectedNodes: [n.id],
        removedNodes: [n.id],
        apply: (g) => applyRewireThrough(g, n.id, aE.from),
      });
    }
  });

  graph.nodes.forEach((n) => {
    if (!["Neg", "Not", "Abs", "Identity"].includes(n.type)) return;
    if (n.type === "Identity") {
      const inE = (incoming[n.id] || [])[0];
      if (!inE) return;
      out.push({
        id: `id-identity-${n.id}`,
        rule: "identity-op",
        title: `Identity is a no-op (${n.id})`,
        detail: `Passthrough from ${inE.from}`,
        category: "algebraic",
        severity: "safe",
        savings: 1,
        affectedNodes: [n.id],
        removedNodes: [n.id],
        apply: (g) => applyRewireThrough(g, n.id, inE.from),
      });
      return;
    }
    const inE = (incoming[n.id] || [])[0];
    if (!inE) return;
    const src = byId[inE.from.split(":")[0]];
    if (!src) return;
    if (n.type === "Abs" && src.type === "Abs") {
      const grand = (incoming[src.id] || [])[0];
      if (!grand) return;
      out.push({
        id: `abs-abs-${n.id}`,
        rule: "double-abs",
        title: `Abs(Abs(x)) = Abs(x) (${n.id})`,
        detail: `Drops outer Abs`,
        category: "algebraic",
        severity: "safe",
        savings: 1,
        affectedNodes: [n.id, src.id],
        removedNodes: [n.id],
        apply: (g) => applyRewireThrough(g, n.id, inE.from),
      });
      return;
    }
    if ((n.type === "Neg" && src.type === "Neg") || (n.type === "Not" && src.type === "Not")) {
      const grand = (incoming[src.id] || [])[0];
      if (!grand) return;
      const srcUsedElsewhere = sourceUsedElsewhere(graph, src.id, inE);
      out.push({
        id: `dbl-${n.type}-${n.id}`,
        rule: `double-${n.type.toLowerCase()}`,
        title: `${n.type}(${n.type}(x)) = x (${n.id})`,
        detail: srcUsedElsewhere ? `Drops outer ${n.type}` : `Drops both ${n.type} nodes`,
        category: "algebraic",
        severity: "safe",
        savings: srcUsedElsewhere ? 1 : 2,
        affectedNodes: srcUsedElsewhere ? [n.id] : [n.id, src.id],
        removedNodes: srcUsedElsewhere ? [n.id] : [n.id, src.id],
        apply: (g) => {
          let g2 = applyRewireThrough(g, n.id, inE.from);
          if (!srcUsedElsewhere) g2 = applyRewireThrough(g2, src.id, grand.from);
          return g2;
        },
      });
    }
  });

  graph.nodes.forEach((n) => {
    if (n.type !== "Cast") return;
    const inEdge = (incoming[n.id] || [])[0];
    if (!inEdge) return;
    const src = byId[inEdge.from.split(":")[0]];
    if (!src) return;
    const srcDtype = src?.attrs?.dataType ?? src?.attrs?.dtype ?? src?.attrs?.to;
    const targetDtype = n.attrs?.to;
    if (srcDtype != null && targetDtype != null && String(srcDtype) === String(targetDtype)) {
      out.push({
        id: `id-cast-${n.id}`,
        rule: "identity-cast",
        title: `Cast to ${targetDtype} is identity (${n.id})`,
        detail: `Source already dtype ${srcDtype}`,
        category: "algebraic",
        severity: "safe",
        savings: 1,
        affectedNodes: [n.id],
        removedNodes: [n.id],
        apply: (g) => applyRewireThrough(g, n.id, inEdge.from),
      });
      return;
    }
    if (src.type === "Cast") {
      const grand = (incoming[src.id] || [])[0];
      if (!grand) return;
      const srcUsedElsewhere = sourceUsedElsewhere(graph, src.id, inEdge);
      out.push({
        id: `cast-cast-${n.id}`,
        rule: "cast-fusion",
        title: `Cast(Cast(x)) = Cast(x) (${n.id})`,
        detail: `Fold inner Cast into outer`,
        category: "fusion",
        severity: srcUsedElsewhere ? "safe" : "safe",
        savings: srcUsedElsewhere ? 1 : 1,
        affectedNodes: [n.id, src.id],
        removedNodes: srcUsedElsewhere ? [] : [src.id],
        apply: srcUsedElsewhere
          ? null
          : (g) => applyRewireThrough(g, src.id, grand.from),
      });
    }
  });

  graph.nodes.forEach((n) => {
    if (!["Concat", "Sum"].includes(n.type)) return;
    const inEdges = incoming[n.id] || [];
    const uniquePortsConnected = new Set(inEdges.map((e) => e.to)).size;
    if (uniquePortsConnected !== 1) return;
    const inEdge = inEdges[0];
    if (!inEdge) return;
    out.push({
      id: `single-${n.type}-${n.id}`,
      rule: "single-input",
      title: `${n.type} with only one input is identity (${n.id})`,
      detail: `Passthrough from ${inEdge.from}`,
      category: "algebraic",
      severity: "safe",
      savings: 1,
      affectedNodes: [n.id],
      removedNodes: [n.id],
      apply: (g) => applyRewireThrough(g, n.id, inEdge.from),
    });
  });

  graph.nodes.forEach((n) => {
    if (n.type !== "Transpose") return;
    const inE = (incoming[n.id] || [])[0];
    if (!inE) return;
    const perm = n.attrs?.perm;
    if (isIdentityPerm(perm)) {
      out.push({
        id: `id-transpose-${n.id}`,
        rule: "identity-transpose",
        title: `Identity Transpose (${n.id})`,
        detail: `perm = [${(perm || []).join(",")}]`,
        category: "shape",
        severity: "safe",
        savings: 1,
        affectedNodes: [n.id],
        removedNodes: [n.id],
        apply: (g) => applyRewireThrough(g, n.id, inE.from),
      });
      return;
    }
    const src = byId[inE.from.split(":")[0]];
    if (src?.type === "Transpose" && Array.isArray(src.attrs?.perm) && Array.isArray(perm)) {
      const composed = composePerm(perm, src.attrs.perm);
      if (!composed) return;
      const grand = (incoming[src.id] || [])[0];
      if (!grand) return;
      const srcUsed = sourceUsedElsewhere(graph, src.id, inE);
      if (isIdentityPerm(composed)) {
        out.push({
          id: `tt-id-${n.id}`,
          rule: "transpose-fusion-identity",
          title: `Transpose ∘ Transpose collapses to identity (${n.id})`,
          detail: srcUsed ? `Drops outer Transpose` : `Drops both Transpose nodes`,
          category: "fusion",
          severity: "safe",
          savings: srcUsed ? 1 : 2,
          affectedNodes: [n.id, src.id],
          removedNodes: srcUsed ? [n.id] : [n.id, src.id],
          apply: (g) => {
            let g2 = applyRewireThrough(g, n.id, inE.from);
            if (!srcUsed) g2 = applyRewireThrough(g2, src.id, grand.from);
            return g2;
          },
        });
      } else if (!srcUsed) {
        out.push({
          id: `tt-${n.id}`,
          rule: "transpose-fusion",
          title: `Transpose ∘ Transpose fuses into single Transpose (${n.id})`,
          detail: `perm = [${composed.join(",")}]`,
          category: "fusion",
          severity: "safe",
          savings: 1,
          affectedNodes: [n.id, src.id],
          removedNodes: [src.id],
          apply: (g) => {
            const updated = { ...n, attrs: { ...(n.attrs || {}), perm: composed } };
            const replaced = { ...g, nodes: g.nodes.map((x) => x.id === n.id ? updated : x) };
            return applyRewireThrough(replaced, src.id, grand.from);
          },
        });
      }
    }
  });

  graph.nodes.forEach((n) => {
    if (n.type !== "Pad") return;
    const inE = (incoming[n.id] || [])[0];
    if (!inE) return;
    const pads = n.attrs?.pads;
    if (Array.isArray(pads) && isAllZeros(pads)) {
      out.push({
        id: `id-pad-${n.id}`,
        rule: "identity-pad",
        title: `Identity Pad (all zeros) (${n.id})`,
        detail: `pads = [${pads.join(",")}]`,
        category: "shape",
        severity: "safe",
        savings: 1,
        affectedNodes: [n.id],
        removedNodes: [n.id],
        apply: (g) => applyRewireThrough(g, n.id, inE.from),
      });
    }
  });

  graph.nodes.forEach((n) => {
    if (n.type !== "Tile") return;
    const inE = (incoming[n.id] || [])[0];
    if (!inE) return;
    const repeats = n.attrs?.repeats;
    if (Array.isArray(repeats) && isAllOnes(repeats)) {
      out.push({
        id: `id-tile-${n.id}`,
        rule: "identity-tile",
        title: `Identity Tile (repeats all 1) (${n.id})`,
        detail: `repeats = [${repeats.join(",")}]`,
        category: "shape",
        severity: "safe",
        savings: 1,
        affectedNodes: [n.id],
        removedNodes: [n.id],
        apply: (g) => applyRewireThrough(g, n.id, inE.from),
      });
    }
  });

  graph.nodes.forEach((n) => {
    if (n.type !== "Slice") return;
    const axes = n.attrs?.axes;
    const starts = n.attrs?.starts;
    const ends = n.attrs?.ends;
    const steps = n.attrs?.steps;
    if (!Array.isArray(axes) || axes.length === 0) return;
    const targetsBatchOrChannel = axes.some((a) => Number(a) === 0 || Number(a) === 1);
    const nonTrivialStart = Array.isArray(starts) && starts.some((s) => Number(s) > 0);
    const nonTrivialEnd = Array.isArray(ends) && ends.some((e) => Number(e) < 30 && Number(e) !== 1);
    if (!targetsBatchOrChannel || (!nonTrivialStart && !nonTrivialEnd)) return;
    const padTo2 = (arr, fill) => {
      const a = Array.isArray(arr) ? arr.slice(0, 2) : [];
      while (a.length < 2) a.push(fill);
      return a;
    };
    const newAxes = [2, 3];
    const newStarts = padTo2(starts, 0);
    const newEnds = padTo2(ends, 30);
    const newSteps = padTo2(steps, 1);
    out.push({
      id: `slice-axes-nchw-${n.id}`,
      rule: "slice-axes-mismatch",
      title: `Slice axes target batch/channel (${n.id})`,
      detail: `axes:[${axes.join(",")}] hits dim 0 or 1 (size 1 in ARC NCHW); suggest axes:[2,3]`,
      category: "shape",
      severity: "verify",
      savings: 0,
      affectedNodes: [n.id],
      removedNodes: [],
      apply: (g) => {
        const updated = {
          ...n,
          attrs: {
            ...(n.attrs || {}),
            axes: newAxes,
            starts: newStarts,
            ends: newEnds,
            steps: newSteps,
          },
        };
        return { ...g, nodes: g.nodes.map((x) => x.id === n.id ? updated : x) };
      },
    });
  });

  graph.nodes.forEach((n) => {
    if (n.type !== "Slice") return;
    const inE = (incoming[n.id] || [])[0];
    if (!inE) return;
    const steps = n.attrs?.steps;
    const starts = n.attrs?.starts;
    const ends = n.attrs?.ends;
    const axes = n.attrs?.axes;
    const srcShape = inputShapeOf(n, byId, incoming);
    const fullRangeOnAllAxes =
      Array.isArray(steps) && Array.isArray(starts) && Array.isArray(ends) &&
      isAllOnes(steps) && isAllZeros(starts) &&
      Array.isArray(srcShape) &&
      Array.isArray(axes) &&
      axes.length === srcShape.length &&
      ends.every((e, i) => Number(e) >= Number(srcShape[Number(axes[i])] ?? 0));
    if (fullRangeOnAllAxes) {
      out.push({
        id: `id-slice-${n.id}`,
        rule: "identity-slice",
        title: `Identity Slice (full range, step 1) (${n.id})`,
        detail: `Drops slice over [${srcShape.join(",")}]`,
        category: "shape",
        severity: "safe",
        savings: 1,
        affectedNodes: [n.id],
        removedNodes: [n.id],
        apply: (g) => applyRewireThrough(g, n.id, inE.from),
      });
    }
  });

  graph.nodes.forEach((n) => {
    if (n.type !== "Resize") return;
    const inE = (incoming[n.id] || [])[0];
    if (!inE) return;
    const sizes = n.attrs?.sizes;
    const srcShape = inputShapeOf(n, byId, incoming);
    if (Array.isArray(sizes) && Array.isArray(srcShape) && shapesEqual(sizes, srcShape)) {
      out.push({
        id: `id-resize-${n.id}`,
        rule: "identity-resize",
        title: `Identity Resize (sizes match input) (${n.id})`,
        detail: `sizes = [${sizes.join(",")}]`,
        category: "shape",
        severity: "safe",
        savings: 1,
        affectedNodes: [n.id],
        removedNodes: [n.id],
        apply: (g) => applyRewireThrough(g, n.id, inE.from),
      });
    }
  });

  graph.nodes.forEach((n) => {
    if (n.type !== "Clip") return;
    const inE = (incoming[n.id] || [])[0];
    if (!inE) return;
    const mn = Number(n.attrs?.min);
    const mx = Number(n.attrs?.max);
    if (Number.isFinite(mn) && Number.isFinite(mx) && mn <= -1e30 && mx >= 1e30) {
      out.push({
        id: `id-clip-${n.id}`,
        rule: "identity-clip",
        title: `Identity Clip (no-op range) (${n.id})`,
        detail: `[${mn}, ${mx}]`,
        category: "shape",
        severity: "safe",
        savings: 1,
        affectedNodes: [n.id],
        removedNodes: [n.id],
        apply: (g) => applyRewireThrough(g, n.id, inE.from),
      });
    }
  });

  graph.nodes.forEach((n) => {
    if (n.type !== "Where") return;
    const inEdges = incoming[n.id] || [];
    const condE = inEdges.find((e) => e.to.endsWith(":condition"));
    const tE = inEdges.find((e) => e.to.endsWith(":true"));
    const fE = inEdges.find((e) => e.to.endsWith(":false"));
    if (!condE || !tE || !fE) return;
    if (tE.from === fE.from) {
      out.push({
        id: `where-same-${n.id}`,
        rule: "where-same-branches",
        title: `Where(c, x, x) = x (${n.id})`,
        detail: `Passthrough from ${tE.from}`,
        category: "logic",
        severity: "safe",
        savings: 1,
        affectedNodes: [n.id],
        removedNodes: [n.id],
        apply: (g) => applyRewireThrough(g, n.id, tE.from),
      });
      return;
    }
    const condSrc = byId[condE.from.split(":")[0]];
    if (condSrc?.type === "Constant") {
      const allTrue = isConstantAllEqualTo(condSrc, 1);
      const allFalse = isConstantAllEqualTo(condSrc, 0);
      if (allTrue || allFalse) {
        const pickEdge = allTrue ? tE : fE;
        out.push({
          id: `where-const-${n.id}`,
          rule: "where-const-cond",
          title: `Where(${allTrue ? "true" : "false"}, …) selects ${allTrue ? "true" : "false"} branch (${n.id})`,
          detail: `Passthrough from ${pickEdge.from}; drops cond`,
          category: "logic",
          severity: "safe",
          savings: 1,
          affectedNodes: [n.id],
          removedNodes: [n.id],
          apply: (g) => applyRewireThrough(g, n.id, pickEdge.from),
        });
      }
    }
  });

  graph.nodes.forEach((n) => {
    if (!["And", "Or"].includes(n.type)) return;
    const inEdges = incoming[n.id] || [];
    inEdges.forEach((e) => {
      const src = byId[e.from.split(":")[0]];
      if (src?.type !== "Constant") return;
      const isTrue = isConstantAllEqualTo(src, 1);
      const isFalse = isConstantAllEqualTo(src, 0);
      const otherEdge = inEdges.find((e2) => e2 !== e);
      if (!otherEdge) return;
      if (n.type === "And" && isTrue) {
        out.push({
          id: `and-true-${n.id}`,
          rule: "and-true",
          title: `And(x, true) = x (${n.id})`,
          detail: `Passthrough from ${otherEdge.from}`,
          category: "logic",
          severity: "safe",
          savings: 1,
          affectedNodes: [n.id],
          removedNodes: [n.id],
          apply: (g) => applyRewireThrough(g, n.id, otherEdge.from),
        });
      } else if (n.type === "And" && isFalse) {
        const srcShape = constShape(src) || [1, 1, 30, 30];
        const newConst = makeConstant(n.id, [0], srcShape, 9);
        out.push({
          id: `and-false-${n.id}`,
          rule: "and-false",
          title: `And(_, false) = false (${n.id})`,
          detail: `Replace with false Constant`,
          category: "logic",
          severity: "safe",
          savings: 1,
          affectedNodes: [n.id],
          removedNodes: [n.id],
          apply: (g) => applyReplaceWithConstant(g, n.id, newConst),
        });
      } else if (n.type === "Or" && isFalse) {
        out.push({
          id: `or-false-${n.id}`,
          rule: "or-false",
          title: `Or(x, false) = x (${n.id})`,
          detail: `Passthrough from ${otherEdge.from}`,
          category: "logic",
          severity: "safe",
          savings: 1,
          affectedNodes: [n.id],
          removedNodes: [n.id],
          apply: (g) => applyRewireThrough(g, n.id, otherEdge.from),
        });
      } else if (n.type === "Or" && isTrue) {
        const srcShape = constShape(src) || [1, 1, 30, 30];
        const newConst = makeConstant(n.id, [1], srcShape, 9);
        out.push({
          id: `or-true-${n.id}`,
          rule: "or-true",
          title: `Or(_, true) = true (${n.id})`,
          detail: `Replace with true Constant`,
          category: "logic",
          severity: "safe",
          savings: 1,
          affectedNodes: [n.id],
          removedNodes: [n.id],
          apply: (g) => applyReplaceWithConstant(g, n.id, newConst),
        });
      }
    });
  });

  graph.nodes.forEach((n) => {
    if (!["Equal", "Greater", "Less", "GreaterOrEqual", "LessOrEqual"].includes(n.type)) return;
    const inEdges = incoming[n.id] || [];
    const aE = inEdges.find((e) => e.to.endsWith(":a"));
    const bE = inEdges.find((e) => e.to.endsWith(":b"));
    if (!aE || !bE || aE.from !== bE.from) return;
    const srcShape = inputShapeOf(n, byId, incoming) || [1, 1, 30, 30];
    const trueVal = ["Equal", "GreaterOrEqual", "LessOrEqual"].includes(n.type);
    const newConst = makeConstant(n.id, [trueVal ? 1 : 0], srcShape, 9);
    out.push({
      id: `${n.type.toLowerCase()}-self-${n.id}`,
      rule: "self-compare",
      title: `${n.type}(x, x) = ${trueVal ? "true" : "false"} (${n.id})`,
      detail: `Replace with ${trueVal ? "true" : "false"} Constant`,
      category: "logic",
      severity: "safe",
      savings: 1,
      affectedNodes: [n.id],
      removedNodes: [n.id],
      apply: (g) => applyReplaceWithConstant(g, n.id, newConst),
    });
  });

  graph.nodes.forEach((n) => {
    if (n.type !== "Constant") return;
    const arr = constArray(n);
    if (!arr || arr.length < 2) return;
    const shapeArr = constShape(n) || [arr.length];
    if (shapeArr.length === 1 && shapeArr[0] === arr.length) return;
    const scalar = constScalarValue(n);
    if (scalar == null) return;
    const sinks = consumerPortSig(graph, n.id);
    if (sinks.length === 0) return;
    if (sinks.some((s) => SHAPE_RIGID_PORTS.has(s))) return;
    if (sinks.some((s) => s.startsWith("Concat."))) return;
    const dtype = Number(n.attrs?.dataType ?? 1);
    const idShort = (n.id || "c").slice(0, 18);
    out.push({
      id: `scalar-bcast-${n.id}`,
      rule: "scalar-broadcast-collapse",
      title: `Collapse Constant to scalar (${idShort})`,
      detail: `${arr.length} identical values → 1-element constant`,
      category: "const-fold",
      severity: "verify",
      savings: 0,
      affectedNodes: [n.id],
      removedNodes: [],
      apply: (g) => {
        const compact = { ...n, attrs: { ...(n.attrs || {}), values: [scalar], shape: [1], dataType: dtype } };
        return { ...g, nodes: g.nodes.map((x) => x.id === n.id ? compact : x) };
      },
    });
  });

  graph.nodes.forEach((n) => {
    if (n.type !== "Constant") return;
    const arr = constArray(n);
    if (!arr || arr.length === 0) return;
    const dtype = Number(n.attrs?.dataType ?? 1);
    const numericDtypes = [1, 6, 7, 10, 11];
    if (!numericDtypes.includes(dtype)) return;
    const allInts = arr.every((x) => Number.isFinite(Number(x)) && Math.trunc(Number(x)) === Number(x));
    if (!allInts) return;
    let mn = Infinity, mx = -Infinity;
    for (const v of arr) { const x = Number(v); if (x < mn) mn = x; if (x > mx) mx = x; }
    const dtypeBytes = { 1: 4, 6: 4, 7: 8, 10: 2, 11: 8 };
    const currentBytes = dtypeBytes[dtype] || 4;
    let targetDtype = null;
    let targetLabel = "int8";
    let targetBytes = 1;
    if (mn >= -128 && mx <= 127) { targetDtype = 3; targetLabel = "int8"; targetBytes = 1; }
    else if (mn >= -32768 && mx <= 32767) { targetDtype = 5; targetLabel = "int16"; targetBytes = 2; }
    else if (mn >= -(2 ** 31) && mx <= (2 ** 31 - 1)) { targetDtype = 6; targetLabel = "int32"; targetBytes = 4; }
    if (targetDtype == null) return;
    if (targetBytes >= currentBytes) return;
    const sinks = consumerPortSig(graph, n.id);
    if (sinks.length === 0) return;
    if (!sinks.every((s) => DTYPE_PASSTHROUGH_PORTS.has(s))) return;
    const idShort = (n.id || "c").slice(0, 18);
    const saveBytes = (currentBytes - targetBytes) * arr.length;
    out.push({
      id: `downcast-${n.id}`,
      rule: "constant-downcast",
      title: `Downcast Constant to ${targetLabel} (${idShort})`,
      detail: `~${saveBytes} bytes saved (${arr.length} × ${currentBytes}B → ${targetBytes}B)`,
      category: "const-fold",
      severity: "verify",
      savings: 0,
      affectedNodes: [n.id],
      removedNodes: [],
      apply: (g) => {
        const updated = { ...n, attrs: { ...(n.attrs || {}), dataType: targetDtype } };
        return { ...g, nodes: g.nodes.map((x) => x.id === n.id ? updated : x) };
      },
    });
  });

  graph.nodes.forEach((n) => {
    if (n.type !== "Cast") return;
    const inE = (incoming[n.id] || [])[0];
    if (!inE) return;
    const src = byId[inE.from.split(":")[0]];
    if (!src || src.type === "Constant") return;
    if (BOOL_PRODUCING_OPS.has(src.type)) return;
    if (CAST_UNDROPPABLE_SOURCES.has(src.type)) return;
    out.push({
      id: `try-drop-cast-${n.id}`,
      rule: "drop-cast",
      title: `Drop Cast (try) (${n.id})`,
      detail: `Removes Cast; verify downstream still works`,
      category: "algebraic",
      severity: "verify",
      savings: 1,
      affectedNodes: [n.id],
      removedNodes: [n.id],
      apply: (g) => applyRewireThrough(g, n.id, inE.from),
    });
  });

  const buckets = {};
  graph.nodes.forEach((n) => {
    if (n.type !== "Constant") return;
    const key = JSON.stringify({
      value: n.attrs?.value ?? null,
      values: n.attrs?.values ?? null,
      shape: n.attrs?.shape ?? null,
      dtype: n.attrs?.dataType ?? null,
    });
    (buckets[key] = buckets[key] || []).push(n.id);
  });
  Object.values(buckets).forEach((ids) => {
    if (ids.length < 2) return;
    const survivor = ids[0];
    const dups = ids.slice(1);
    out.push({
      id: `dup-const-${survivor}`,
      rule: "duplicate-constants",
      title: `${ids.length} identical Constants`,
      detail: `Merge ${dups.length} duplicate${dups.length > 1 ? "s" : ""} into ${survivor}`,
      category: "dead-code",
      severity: "safe",
      savings: dups.length,
      affectedNodes: ids,
      removedNodes: dups,
      apply: (g) => applyMergeConstants(g, survivor, dups),
    });
  });

  graph.nodes.forEach((n) => {
    if (n.type !== "Reshape") return;
    const inEdges = incoming[n.id] || [];
    const xE = inEdges.find((e) => e.to.endsWith(":data") || e.to.endsWith(":x") || e.to.endsWith(":input"));
    const shapeE = inEdges.find((e) => e.to.endsWith(":shape"));
    if (!xE || !shapeE) return;
    const xSrc = byId[xE.from.split(":")[0]];
    const shapeSrc = byId[shapeE.from.split(":")[0]];
    const xShape = xSrc?.attrs?.shape;
    if (shapeSrc?.type !== "Constant") return;
    const target = constArray(shapeSrc);
    if (!Array.isArray(xShape) || !Array.isArray(target)) return;
    if (xShape.length === target.length && xShape.every((v, i) => Number(v) === Number(target[i]))) {
      out.push({
        id: `id-reshape-${n.id}`,
        rule: "identity-reshape",
        title: `Reshape to same shape is identity (${n.id})`,
        detail: `Both shapes [${xShape.join(",")}]`,
        category: "shape",
        severity: "safe",
        savings: 1,
        affectedNodes: [n.id],
        removedNodes: [n.id],
        apply: (g) => applyRewireThrough(g, n.id, xE.from),
      });
    }
    if (xSrc?.type === "Reshape") {
      const grand = inputEdgeOnPort(incoming, xSrc.id, "data") || (incoming[xSrc.id] || [])[0];
      if (!grand) return;
      const innerShapeSrc = byId[(inputEdgeOnPort(incoming, xSrc.id, "shape") || {}).from?.split(":")[0]];
      const srcUsed = sourceUsedElsewhere(graph, xSrc.id, xE);
      if (!srcUsed) {
        out.push({
          id: `rr-${n.id}`,
          rule: "reshape-fusion",
          title: `Reshape ∘ Reshape fuses (${n.id})`,
          detail: `Drops intermediate Reshape ${xSrc.id}`,
          category: "fusion",
          severity: "safe",
          savings: innerShapeSrc && !sourceUsedElsewhere(graph, innerShapeSrc.id, null) ? 2 : 1,
          affectedNodes: [n.id, xSrc.id],
          removedNodes: [xSrc.id],
          apply: (g) => applyRewireThrough(g, xSrc.id, grand.from),
        });
      }
    }
  });

  graph.nodes.forEach((n) => {
    if (n.type !== "Squeeze") return;
    const inE = (incoming[n.id] || [])[0];
    if (!inE) return;
    const src = byId[inE.from.split(":")[0]];
    if (src?.type !== "Unsqueeze") return;
    const ax1 = src.attrs?.axes;
    const ax2 = n.attrs?.axes;
    if (Array.isArray(ax1) && Array.isArray(ax2) && arraysShallowEqual(ax1, ax2)) {
      const grand = (incoming[src.id] || [])[0];
      if (!grand) return;
      const srcUsed = sourceUsedElsewhere(graph, src.id, inE);
      out.push({
        id: `sq-unsq-${n.id}`,
        rule: "squeeze-unsqueeze",
        title: `Squeeze ∘ Unsqueeze cancels (${n.id})`,
        detail: srcUsed ? `Drops outer Squeeze` : `Drops both`,
        category: "fusion",
        severity: "safe",
        savings: srcUsed ? 1 : 2,
        affectedNodes: [n.id, src.id],
        removedNodes: srcUsed ? [n.id] : [n.id, src.id],
        apply: (g) => {
          let g2 = applyRewireThrough(g, n.id, inE.from);
          if (!srcUsed) g2 = applyRewireThrough(g2, src.id, grand.from);
          return g2;
        },
      });
    }
  });

  graph.nodes.forEach((n) => {
    if (n.type !== "Cast") return;
    const inE = (incoming[n.id] || [])[0];
    if (!inE) return;
    const src = byId[inE.from.split(":")[0]];
    if (src?.type !== "Constant") return;
    const arr = constArray(src);
    const shapeArr = constShape(src) || [arr?.length || 1];
    if (!arr) return;
    const srcUsed = sourceUsedElsewhere(graph, src.id, inE);
    if (srcUsed) return;
    const targetDtype = Number(n.attrs?.to ?? 1);
    let casted = arr;
    if ([6, 7, 9].includes(targetDtype)) casted = arr.map((x) => Math.trunc(Number(x)));
    else if (targetDtype === 9) casted = arr.map((x) => Number(x) !== 0 ? 1 : 0);
    else casted = arr.map((x) => Number(x));
    const newConst = makeConstant(n.id, casted, shapeArr, targetDtype);
    out.push({
      id: `fold-cast-${n.id}`,
      rule: "fold-cast",
      title: `Fold Cast over Constant (${n.id})`,
      detail: `Drops Cast + ${src.id} into one Constant`,
      category: "const-fold",
      severity: "safe",
      savings: 1,
      affectedNodes: [n.id, src.id],
      removedNodes: [n.id, src.id],
      apply: (g) => {
        const g2 = applyReplaceWithConstant(g, n.id, newConst);
        return applyRemoveNodes(g2, [src.id]);
      },
    });
  });

  graph.nodes.forEach((n) => {
    if (!["Neg", "Abs", "Floor", "Sqrt", "Sign", "Not"].includes(n.type)) return;
    const inE = (incoming[n.id] || [])[0];
    if (!inE) return;
    const src = byId[inE.from.split(":")[0]];
    if (src?.type !== "Constant") return;
    const arr = constArray(src);
    const shapeArr = constShape(src) || [arr?.length || 1];
    if (!arr) return;
    const srcUsed = sourceUsedElsewhere(graph, src.id, inE);
    if (srcUsed) return;
    let folded;
    switch (n.type) {
      case "Neg": folded = arr.map((x) => -Number(x)); break;
      case "Abs": folded = arr.map((x) => Math.abs(Number(x))); break;
      case "Floor": folded = arr.map((x) => Math.floor(Number(x))); break;
      case "Sqrt": folded = arr.map((x) => Math.sqrt(Math.max(0, Number(x)))); break;
      case "Sign": folded = arr.map((x) => Math.sign(Number(x))); break;
      case "Not": folded = arr.map((x) => Number(x) === 0 ? 1 : 0); break;
      default: return;
    }
    const dtype = n.type === "Not" ? 9 : (src.attrs?.dataType ?? 1);
    const newConst = makeConstant(n.id, folded, shapeArr, dtype);
    out.push({
      id: `fold-${n.type}-${n.id}`,
      rule: "fold-unary",
      title: `Fold ${n.type} over Constant (${n.id})`,
      detail: `Drops ${n.type} + ${src.id} into one Constant`,
      category: "const-fold",
      severity: "safe",
      savings: 1,
      affectedNodes: [n.id, src.id],
      removedNodes: [n.id, src.id],
      apply: (g) => {
        const g2 = applyReplaceWithConstant(g, n.id, newConst);
        return applyRemoveNodes(g2, [src.id]);
      },
    });
  });

  graph.nodes.forEach((n) => {
    if (!["Add", "Sub", "Mul", "Div", "Min", "Max", "Mod"].includes(n.type)) return;
    const inEdges = incoming[n.id] || [];
    const aE = inEdges.find((e) => e.to.endsWith(":a"));
    const bE = inEdges.find((e) => e.to.endsWith(":b"));
    if (!aE || !bE) return;
    const a = byId[aE.from.split(":")[0]];
    const b = byId[bE.from.split(":")[0]];
    if (a?.type !== "Constant" || b?.type !== "Constant") return;
    const av = constArray(a);
    const bv = constArray(b);
    if (!av || !bv) return;
    const aShape = constShape(a) || [av.length];
    const bShape = constShape(b) || [bv.length];
    const outShape = broadcastShape(aShape, bShape);
    if (!outShape) return;
    const sizeOut = outShape.reduce((p, x) => p * x, 1);
    if (sizeOut > 50000) return;
    const aTotal = aShape.reduce((p, x) => p * x, 1);
    const bTotal = bShape.reduce((p, x) => p * x, 1);
    if (av.length !== aTotal || bv.length !== bTotal) return;
    const aUsed = sourceUsedElsewhere(graph, a.id, aE);
    const bUsed = sourceUsedElsewhere(graph, b.id, bE);

    const broadcastIndex = (idx, srcShape, outShape) => {
      const out = [];
      let cur = idx;
      for (let i = outShape.length - 1; i >= 0; i--) {
        out.unshift(cur % outShape[i]);
        cur = Math.floor(cur / outShape[i]);
      }
      let srcIdx = 0, mul = 1;
      const offset = outShape.length - srcShape.length;
      for (let i = srcShape.length - 1; i >= 0; i--) {
        const dim = srcShape[i];
        const co = dim === 1 ? 0 : out[i + offset];
        srcIdx += co * mul;
        mul *= dim;
      }
      return srcIdx;
    };

    const folded = new Array(sizeOut);
    for (let i = 0; i < sizeOut; i++) {
      const av_ = Number(av[broadcastIndex(i, aShape, outShape)]);
      const bv_ = Number(bv[broadcastIndex(i, bShape, outShape)]);
      let v;
      switch (n.type) {
        case "Add": v = av_ + bv_; break;
        case "Sub": v = av_ - bv_; break;
        case "Mul": v = av_ * bv_; break;
        case "Div": if (bv_ === 0) return; v = av_ / bv_; break;
        case "Mod": if (bv_ === 0) return; v = av_ % bv_; break;
        case "Min": v = Math.min(av_, bv_); break;
        case "Max": v = Math.max(av_, bv_); break;
        default: return;
      }
      folded[i] = v;
    }
    const dtype = a.attrs?.dataType ?? b.attrs?.dataType ?? 1;
    const newConst = makeConstant(n.id, folded, outShape, dtype);
    const removed = [n.id];
    if (!aUsed) removed.push(a.id);
    if (!bUsed) removed.push(b.id);
    out.push({
      id: `fold-binary-${n.id}`,
      rule: "fold-binary",
      title: `Fold ${n.type}(Const, Const) (${n.id})`,
      detail: `Drops ${removed.length} node${removed.length > 1 ? "s" : ""}`,
      category: "const-fold",
      severity: "safe",
      savings: removed.length,
      affectedNodes: removed,
      removedNodes: removed,
      apply: (g) => {
        let g2 = applyReplaceWithConstant(g, n.id, newConst);
        const dropConsts = removed.filter((x) => x !== n.id);
        if (dropConsts.length) g2 = applyRemoveNodes(g2, dropConsts);
        return g2;
      },
    });
  });

  const mulFromInput = [];
  graph.nodes.forEach((n) => {
    if (n.type !== "Mul") return;
    const inEs = incoming[n.id] || [];
    const inputEdge = inEs.find((e) => {
      const src = byId[e.from.split(":")[0]];
      return src?.type === "Input";
    });
    if (inputEdge) mulFromInput.push(n.id);
  });
  if (mulFromInput.length >= 3) {
    out.push({
      id: `parallel-mul-${mulFromInput.length}`,
      rule: "parallel-clones",
      title: `${mulFromInput.length} parallel Mul branches share the same input`,
      detail: `Consider Gather lookup table or Tile + single Mul (heuristic only)`,
      category: "idiom",
      severity: "speculative",
      savings: Math.max(0, (mulFromInput.length - 1) * 3),
      affectedNodes: mulFromInput,
      removedNodes: [],
      apply: null,
    });
  }

  const constNodes = graph.nodes.filter((n) => n.type === "Constant");
  let bigConstantBytes = 0;
  const bigConsts = [];
  for (const c of constNodes) {
    const arr = constArray(c);
    const shape = constShape(c) || (arr ? [arr.length] : null);
    if (!arr || !shape) continue;
    const size = arr.length;
    const sValue = constScalarValue(c);
    if (sValue != null && size > 1) {
      bigConsts.push({ id: c.id, size, scalar: sValue });
      bigConstantBytes += size * 4;
    }
  }
  if (bigConsts.length > 0) {
    out.push({
      id: `scalar-broadcast-${bigConsts.length}`,
      rule: "scalar-broadcast",
      title: `${bigConsts.length} Constant${bigConsts.length > 1 ? "s" : ""} could be scalar broadcasts`,
      detail: `Save ~${(bigConstantBytes/1024).toFixed(1)} KB by storing 1-element + relying on broadcast`,
      category: "idiom",
      severity: "speculative",
      savings: 0,
      affectedNodes: bigConsts.map((c) => c.id),
      removedNodes: [],
      apply: null,
    });
  }

  graph.nodes.forEach((n) => {
    if (n.type !== "Conv") return;
    const wEdge = (incoming[n.id] || []).find((e) => {
      const src = byId[e.from.split(":")[0]];
      return src?.type === "Constant";
    });
    if (!wEdge) return;
    const wSrc = byId[wEdge.from.split(":")[0]];
    const arr = constArray(wSrc);
    if (!arr) return;
    const nonzero = arr.filter((x) => Math.abs(Number(x)) > 1e-12).length;
    const ratio = nonzero / arr.length;
    if (nonzero === 0 && arr.length > 0) {
      const outShape = inputShapeOf(n, byId, incoming) || [1, 1, 30, 30];
      const newConst = makeConstant(n.id, [0], outShape, 1);
      out.push({
        id: `conv-zero-${n.id}`,
        rule: "conv-zero-weights",
        title: `Conv with all-zero weights → zero (${n.id})`,
        detail: `Replace Conv with zero Constant`,
        category: "const-fold",
        severity: "verify",
        savings: 2,
        affectedNodes: [n.id, wSrc.id],
        removedNodes: [n.id, wSrc.id],
        apply: (g) => {
          const g2 = applyReplaceWithConstant(g, n.id, newConst);
          return sourceUsedElsewhere(graph, wSrc.id, wEdge) ? g2 : applyRemoveNodes(g2, [wSrc.id]);
        },
      });
      return;
    }
    if (arr.length >= 64 && ratio < 0.10) {
      out.push({
        id: `conv-sparse-${n.id}`,
        rule: "sparse-conv",
        title: `Conv weights are ${(ratio*100).toFixed(1)}% nonzero (${n.id})`,
        detail: `Consider hand-coded Conv via Gather/Mul/Add to shrink params`,
        category: "idiom",
        severity: "speculative",
        savings: 0,
        affectedNodes: [n.id, wSrc.id],
        removedNodes: [],
        apply: null,
      });
    }
  });

  graph.nodes.forEach((n) => {
    if (n.type !== "Reshape") return;
    const inEdges = incoming[n.id] || [];
    const xE = inEdges.find((e) => e.to.endsWith(":data"));
    const shapeE = inEdges.find((e) => e.to.endsWith(":shape"));
    if (!xE || !shapeE) return;
    const xSrc = byId[xE.from.split(":")[0]];
    const shapeSrc = byId[shapeE.from.split(":")[0]];
    if (xSrc?.type !== "Constant" || shapeSrc?.type !== "Constant") return;
    const xValues = constArray(xSrc);
    const targetShape = constArray(shapeSrc);
    if (!xValues || !targetShape) return;
    const size = targetShape.reduce((p, x) => p * Math.max(1, Number(x)), 1);
    if (size !== xValues.length || size > 50000) return;
    const xUsed = sourceUsedElsewhere(graph, xSrc.id, xE);
    const shapeUsed = sourceUsedElsewhere(graph, shapeSrc.id, shapeE);
    const newConst = makeConstant(n.id, xValues, targetShape.map(Number), xSrc.attrs?.dataType ?? 1);
    const dropList = [n.id];
    if (!xUsed) dropList.push(xSrc.id);
    if (!shapeUsed) dropList.push(shapeSrc.id);
    out.push({
      id: `fold-reshape-${n.id}`,
      rule: "fold-reshape",
      title: `Fold Reshape(Constant) (${n.id})`,
      detail: `Drops ${dropList.length} node${dropList.length === 1 ? "" : "s"}`,
      category: "const-fold",
      severity: "safe",
      savings: dropList.length,
      affectedNodes: dropList,
      removedNodes: dropList,
      apply: (g) => {
        let g2 = applyReplaceWithConstant(g, n.id, newConst);
        const drop = dropList.filter((x) => x !== n.id);
        return drop.length ? applyRemoveNodes(g2, drop) : g2;
      },
    });
  });

  graph.nodes.forEach((n) => {
    if (n.type !== "Tile") return;
    const inE = (incoming[n.id] || [])[0];
    if (!inE) return;
    const src = byId[inE.from.split(":")[0]];
    if (src?.type !== "Constant") return;
    const repeats = n.attrs?.repeats;
    if (!Array.isArray(repeats)) return;
    const srcArr = constArray(src);
    const srcShape = constShape(src);
    if (!srcArr || !srcShape) return;
    if (repeats.length !== srcShape.length) return;
    const totalRepeat = repeats.reduce((p, r) => p * Math.max(1, Number(r)), 1);
    if (srcArr.length * totalRepeat > 50000) return;
    const outShape = srcShape.map((d, i) => Number(d) * Number(repeats[i]));
    const outSize = outShape.reduce((p, x) => p * x, 1);
    const tiled = new Array(outSize);
    const strideOut = new Array(outShape.length);
    const strideIn = new Array(srcShape.length);
    {
      let s = 1;
      for (let i = outShape.length - 1; i >= 0; i--) { strideOut[i] = s; s *= outShape[i]; }
    }
    {
      let s = 1;
      for (let i = srcShape.length - 1; i >= 0; i--) { strideIn[i] = s; s *= Number(srcShape[i]); }
    }
    for (let idx = 0; idx < outSize; idx++) {
      let inIdx = 0;
      let remainder = idx;
      for (let dim = 0; dim < outShape.length; dim++) {
        const coord = Math.floor(remainder / strideOut[dim]) % outShape[dim];
        const srcCoord = coord % Number(srcShape[dim]);
        inIdx += srcCoord * strideIn[dim];
        remainder = remainder % strideOut[dim];
      }
      tiled[idx] = srcArr[inIdx];
    }
    const srcUsed = sourceUsedElsewhere(graph, src.id, inE);
    const newConst = makeConstant(n.id, tiled, outShape, src.attrs?.dataType ?? 1);
    const drop = [n.id, ...(srcUsed ? [] : [src.id])];
    out.push({
      id: `fold-tile-${n.id}`,
      rule: "fold-tile",
      title: `Fold Tile(Constant) (${n.id})`,
      detail: `${srcArr.length} → ${outSize} elements as a single Constant`,
      category: "const-fold",
      severity: "safe",
      savings: drop.length,
      affectedNodes: drop,
      removedNodes: drop,
      apply: (g) => {
        let g2 = applyReplaceWithConstant(g, n.id, newConst);
        const extra = drop.filter((x) => x !== n.id);
        return extra.length ? applyRemoveNodes(g2, extra) : g2;
      },
    });
  });

  graph.nodes.forEach((n) => {
    if (n.type !== "Concat") return;
    const inEdges = (incoming[n.id] || []).slice().sort((a, b) => {
      const ai = (a.data?.inputIndex ?? 0); const bi = (b.data?.inputIndex ?? 0);
      return ai - bi;
    });
    if (inEdges.length < 2) return;
    const sources = inEdges.map((e) => byId[e.from.split(":")[0]]);
    if (sources.some((s) => !s || s.type !== "Constant")) return;
    const axis = Number(n.attrs?.axis ?? 0);
    const arrs = sources.map(constArray);
    const shapes = sources.map(constShape);
    if (arrs.some((a) => !a) || shapes.some((s) => !s)) return;
    const refShape = shapes[0];
    for (const s of shapes) {
      if (s.length !== refShape.length) return;
      for (let d = 0; d < s.length; d++) if (d !== axis && Number(s[d]) !== Number(refShape[d])) return;
    }
    const outShape = refShape.slice();
    outShape[axis] = shapes.reduce((sum, s) => sum + Number(s[axis]), 0);
    const totalSize = outShape.reduce((p, x) => p * Math.max(1, x), 1);
    if (totalSize > 50000) return;
    let dtype = sources[0].attrs?.dataType ?? 1;
    const concatAlongLast = axis === outShape.length - 1;
    let combined;
    if (concatAlongLast) {
      combined = [];
      const outerSize = refShape.slice(0, axis).reduce((p, x) => p * x, 1) || 1;
      for (let outer = 0; outer < outerSize; outer++) {
        for (let s = 0; s < sources.length; s++) {
          const innerSize = Number(shapes[s][axis]);
          const arr = arrs[s];
          for (let i = 0; i < innerSize; i++) combined.push(arr[outer * innerSize + i]);
        }
      }
    } else {
      combined = arrs.reduce((acc, a) => acc.concat(a), []);
      if (combined.length !== totalSize) return;
    }
    const usedAny = sources.some((s, idx) => sourceUsedElsewhere(graph, s.id, inEdges[idx]));
    const drop = [n.id, ...(usedAny ? [] : sources.map((s) => s.id))];
    const newConst = makeConstant(n.id, combined, outShape, dtype);
    out.push({
      id: `fold-concat-${n.id}`,
      rule: "fold-concat",
      title: `Fold Concat of ${sources.length} Constants (${n.id})`,
      detail: `${combined.length} elements as single Constant${usedAny ? "" : `; drops ${sources.length} inputs`}`,
      category: "const-fold",
      severity: "safe",
      savings: drop.length,
      affectedNodes: drop,
      removedNodes: drop,
      apply: (g) => {
        let g2 = applyReplaceWithConstant(g, n.id, newConst);
        const extra = drop.filter((x) => x !== n.id);
        return extra.length ? applyRemoveNodes(g2, extra) : g2;
      },
    });
  });

  graph.nodes.forEach((n) => {
    if (n.type !== "Squeeze" && n.type !== "Unsqueeze") return;
    const inE = (incoming[n.id] || [])[0];
    if (!inE) return;
    const src = byId[inE.from.split(":")[0]];
    if (src?.type !== "Constant") return;
    const arr = constArray(src);
    if (!arr) return;
    const srcShape = constShape(src) || [arr.length];
    const axes = n.attrs?.axes;
    const axesList = Array.isArray(axes) ? axes.map(Number) : [];
    let newShape = srcShape.slice();
    if (n.type === "Squeeze") {
      if (axesList.length) {
        const norm = axesList.map((a) => (a + srcShape.length) % srcShape.length).sort();
        newShape = srcShape.filter((_, i) => !norm.includes(i));
      } else {
        newShape = srcShape.filter((d) => Number(d) !== 1);
      }
    } else {
      if (!axesList.length) return;
      const rank = srcShape.length + axesList.length;
      const norm = axesList.map((a) => (a + rank) % rank).sort();
      newShape = srcShape.slice();
      for (const ax of norm) newShape.splice(ax, 0, 1);
    }
    const srcUsed = sourceUsedElsewhere(graph, src.id, inE);
    const newConst = makeConstant(n.id, arr, newShape.length ? newShape : [arr.length], src.attrs?.dataType ?? 1);
    const drop = [n.id, ...(srcUsed ? [] : [src.id])];
    out.push({
      id: `fold-${n.type.toLowerCase()}-${n.id}`,
      rule: `fold-${n.type.toLowerCase()}`,
      title: `Fold ${n.type}(Constant) (${n.id})`,
      detail: `Shape ${srcShape.join("×") || "[]"} → ${newShape.join("×") || "[]"}`,
      category: "const-fold",
      severity: "safe",
      savings: drop.length,
      affectedNodes: drop,
      removedNodes: drop,
      apply: (g) => {
        let g2 = applyReplaceWithConstant(g, n.id, newConst);
        const extra = drop.filter((x) => x !== n.id);
        return extra.length ? applyRemoveNodes(g2, extra) : g2;
      },
    });
  });

  graph.nodes.forEach((n) => {
    if (n.type !== "Transpose") return;
    const inE = (incoming[n.id] || [])[0];
    if (!inE) return;
    const src = byId[inE.from.split(":")[0]];
    if (src?.type !== "Constant") return;
    const perm = n.attrs?.perm;
    if (!Array.isArray(perm)) return;
    const arr = constArray(src);
    const srcShape = constShape(src) || (arr ? [arr.length] : null);
    if (!arr || !srcShape || perm.length !== srcShape.length) return;
    if (arr.length > 50000) return;
    if (srcShape.reduce((p, x) => p * x, 1) !== arr.length) return;
    const outShape = perm.map((p) => Number(srcShape[Number(p)]));
    const outSize = outShape.reduce((p, x) => p * x, 1);
    const inStride = new Array(srcShape.length);
    {
      let s = 1;
      for (let i = srcShape.length - 1; i >= 0; i--) { inStride[i] = s; s *= Number(srcShape[i]); }
    }
    const outStride = new Array(outShape.length);
    {
      let s = 1;
      for (let i = outShape.length - 1; i >= 0; i--) { outStride[i] = s; s *= outShape[i]; }
    }
    const result = new Array(outSize);
    for (let idx = 0; idx < outSize; idx++) {
      let inIdx = 0;
      let remainder = idx;
      for (let dim = 0; dim < outShape.length; dim++) {
        const coord = Math.floor(remainder / outStride[dim]) % outShape[dim];
        inIdx += coord * inStride[perm[dim]];
        remainder = remainder % outStride[dim];
      }
      result[idx] = arr[inIdx];
    }
    const srcUsed = sourceUsedElsewhere(graph, src.id, inE);
    const newConst = makeConstant(n.id, result, outShape, src.attrs?.dataType ?? 1);
    const drop = [n.id, ...(srcUsed ? [] : [src.id])];
    out.push({
      id: `fold-transpose-${n.id}`,
      rule: "fold-transpose",
      title: `Fold Transpose(Constant) (${n.id})`,
      detail: `Shape ${srcShape.join("×")} → ${outShape.join("×")}`,
      category: "const-fold",
      severity: "safe",
      savings: drop.length,
      affectedNodes: drop,
      removedNodes: drop,
      apply: (g) => {
        let g2 = applyReplaceWithConstant(g, n.id, newConst);
        const extra = drop.filter((x) => x !== n.id);
        return extra.length ? applyRemoveNodes(g2, extra) : g2;
      },
    });
  });

  const seen = new Map();
  for (const s of out) {
    if (!seen.has(s.id)) seen.set(s.id, s);
  }
  const uniq = [...seen.values()];

  uniq.sort((a, b) => {
    if (a.severity !== b.severity) return a.severity === "safe" ? -1 : 1;
    return b.savings - a.savings;
  });

  return uniq;
}

export const CATEGORY_META = {
  algebraic: { label: "Algebraic", color: "oklch(0.78 0.13 75)" },
  "dead-code": { label: "Dead-code", color: "oklch(0.75 0.14 330)" },
  idiom: { label: "Idiom", color: "oklch(0.78 0.13 260)" },
  shape: { label: "Shape", color: "oklch(0.78 0.13 200)" },
  logic: { label: "Logic", color: "oklch(0.78 0.13 155)" },
  fusion: { label: "Fusion", color: "oklch(0.72 0.16 305)" },
  "const-fold": { label: "Const-fold", color: "oklch(0.78 0.13 50)" },
};

function fmtBytes(n) {
  if (!Number.isFinite(n)) return "—";
  if (n < 1024) return `${n} B`;
  if (n < 1024 * 1024) return `${(n / 1024).toFixed(1)} KB`;
  return `${(n / 1024 / 1024).toFixed(2)} MB`;
}
function fmtNum(n) {
  if (!Number.isFinite(n)) return "—";
  if (n < 1000) return String(n);
  if (n < 1_000_000) return `${(n / 1000).toFixed(n < 10000 ? 1 : 0)}K`;
  return `${(n / 1_000_000).toFixed(2)}M`;
}

function CapBar({ value, cap, dangerAt = 0.85 }) {
  const r = Math.min(1, value / cap);
  const color = r > dangerAt ? "var(--danger)" : r > 0.6 ? "var(--warning)" : "var(--success)";
  return (
    <div style={{ marginTop: 4, height: 4, background: "var(--bg-input)", borderRadius: 2, overflow: "hidden" }}>
      <div style={{ width: `${r * 100}%`, height: "100%", background: color, transition: "width 200ms" }} />
    </div>
  );
}

function MetricChip({ label, value, sub, tone = "default" }) {
  const colors = {
    default: { fg: "var(--text)", bg: "var(--bg-input)" },
    accent: { fg: "var(--accent)", bg: "var(--accent-bg)" },
    success: { fg: "var(--success)", bg: "var(--success-bg)" },
    warning: { fg: "var(--warning)", bg: "var(--warning-bg)" },
    danger: { fg: "var(--danger)", bg: "var(--danger-bg)" },
  }[tone] || { fg: "var(--text)", bg: "var(--bg-input)" };
  return (
    <div style={{
      padding: "6px 8px", background: colors.bg, borderRadius: 5,
      border: "1px solid var(--border)", minWidth: 0,
    }}>
      <div style={{
        fontSize: 9, fontWeight: 600, letterSpacing: ".08em",
        textTransform: "uppercase", color: "var(--text-dim)",
      }}>{label}</div>
      <div style={{
        fontFamily: "var(--mono)", fontSize: 14, fontWeight: 600,
        color: colors.fg, marginTop: 1,
      }}>{value}</div>
      {sub != null && (
        <div style={{ fontFamily: "var(--mono)", fontSize: 10.5, color: "var(--text-muted)", marginTop: 1 }}>
          {sub}
        </div>
      )}
    </div>
  );
}

export function EfficiencyPanel({ efficiency, baseline, best, error, refreshing, onCheck, onRestoreBest, bestGraphAvailable }) {
  if (!efficiency && !error && !refreshing) {
    return (
      <div style={{ padding: "10px 12px", fontSize: 11.5, color: "var(--text-dim)" }}>
        Edit the graph to see live metrics.
      </div>
    );
  }
  if (error) {
    return (
      <div style={{ padding: "10px 12px", fontSize: 11.5, color: "var(--warning)" }}>
        <Icon name="alert" size={11} /> Compile failed: {error}
      </div>
    );
  }
  const e = efficiency;
  const bytesDelta = baseline?.bytes != null && e?.bytes != null ? e.bytes - baseline.bytes : null;
  const paramsDelta = baseline?.parameters != null && e?.parameters != null ? e.parameters - baseline.parameters : null;
  const costDelta = baseline?.cost != null && e?.cost != null ? e.cost - baseline.cost : null;
  const scoreDelta = baseline?.score != null && e?.score != null ? e.score - baseline.score : null;

  const dlt = (v, fmt = (x) => x) => {
    if (v == null) return null;
    if (v === 0) return <span style={{ color: "var(--text-dim)" }}>±0</span>;
    const better = v < 0;
    return <span style={{ color: better ? "var(--success)" : "var(--warning)" }}>{v > 0 ? "+" : ""}{fmt(v)}</span>;
  };

  const hasForbidden = e?.forbiddenOps?.length > 0;
  const hasDynamic = e?.dynamicShapeIssues?.length > 0;
  const hasIssue = hasForbidden || hasDynamic || (e && !e.validBytes);
  const bestDelta = best?.cost != null && e?.cost != null ? e.cost - best.cost : null;
  return (
    <div style={{ padding: "10px 12px", borderBottom: "1px solid var(--border)", background: "var(--bg-surface)" }}>
      <div style={{
        display: "flex", alignItems: "baseline", justifyContent: "space-between", marginBottom: 8,
      }}>
        <div style={{ fontSize: 10.5, fontWeight: 600, letterSpacing: ".08em", textTransform: "uppercase", color: "var(--text-dim)" }}>
          Efficiency
        </div>
        {refreshing && <span style={{ fontSize: 10, color: "var(--text-dim)", fontFamily: "var(--mono)" }}>compiling…</span>}
      </div>
      {hasIssue && e && (
        <div style={{
          padding: "6px 8px", marginBottom: 8,
          background: "var(--danger-bg)", border: "1px solid var(--danger)",
          borderRadius: 4, fontSize: 11, color: "var(--danger)", fontFamily: "var(--mono)",
        }}>
          {hasForbidden && <div>⚠ disallowed ops: {e.forbiddenOps.join(", ")}</div>}
          {hasDynamic && <div>⚠ dynamic shape: {e.dynamicShapeIssues[0]}</div>}
          {!e.validBytes && <div>⚠ exceeds 1.44 MB cap ({fmtBytes(e.bytes)})</div>}
        </div>
      )}
      {e && (
        <>
          <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 6 }}>
            <MetricChip
              label="Score"
              value={e.score.toFixed(2)}
              sub={dlt(scoreDelta, (x) => x.toFixed(2))}
              tone={e.score >= 15 ? "success" : e.score >= 10 ? "default" : "warning"}
            />
            <MetricChip
              label="Cost"
              value={fmtNum(e.cost)}
              sub={dlt(costDelta, (x) => fmtNum(Math.abs(x)))}
              tone={e.cost < 50_000 ? "success" : "default"}
            />
            <MetricChip
              label="Params"
              value={fmtNum(e.parameters)}
              sub={dlt(paramsDelta, (x) => fmtNum(Math.abs(x)))}
            />
            <MetricChip
              label="Bytes"
              value={fmtBytes(e.bytes)}
              sub={dlt(bytesDelta, (x) => fmtBytes(Math.abs(x)))}
              tone={e.validBytes ? "default" : "danger"}
            />
          </div>
          <div style={{ marginTop: 8 }}>
            <div style={{
              display: "flex", alignItems: "center", justifyContent: "space-between",
              fontSize: 10.5, color: "var(--text-muted)", fontFamily: "var(--mono)",
            }}>
              <span>{fmtBytes(e.bytes)} / {fmtBytes(e.byteCap)} cap</span>
              <span>{((e.bytes / e.byteCap) * 100).toFixed(1)}%</span>
            </div>
            <CapBar value={e.bytes} cap={e.byteCap} />
          </div>
          <div style={{ marginTop: 8, display: "flex", flexWrap: "wrap", gap: 4, fontSize: 10.5, color: "var(--text-muted)", fontFamily: "var(--mono)" }}>
            <span style={{ padding: "1px 6px", background: "var(--bg-input)", border: "1px solid var(--border)", borderRadius: 3 }}>
              {e.graphNodes} ops
            </span>
            <span style={{ padding: "1px 6px", background: "var(--bg-input)", border: "1px solid var(--border)", borderRadius: 3 }}>
              {e.initializerCount} const{e.initializerCount === 1 ? "" : "s"}
            </span>
            <span style={{ padding: "1px 6px", background: "var(--bg-input)", border: "1px solid var(--border)", borderRadius: 3 }}>
              {fmtBytes(e.initializerBytes)} weights
            </span>
            {best?.cost && best.verified && (
              <span style={{
                padding: "1px 6px", background: "var(--success-bg)", color: "var(--success)",
                border: "1px solid var(--success)", borderRadius: 3,
              }} title={`Verified best: cost ${best.cost} · score ${best.score.toFixed(2)}`}>
                best ✓ {best.score.toFixed(2)} {bestDelta != null && bestDelta !== 0 ? `(${bestDelta > 0 ? "+" : ""}${fmtNum(Math.abs(bestDelta))} now)` : ""}
              </span>
            )}
          </div>
          {onCheck && (
            <div style={{ marginTop: 8, display: "flex", gap: 6 }}>
              <Button size="sm" variant="ghost" icon="check" onClick={onCheck} style={{ flex: 1, justifyContent: "center" }}>
                Verify on all train pairs
              </Button>
              {bestGraphAvailable && onRestoreBest && (
                <Button size="sm" variant="ghost" icon="refresh" onClick={onRestoreBest} title="Restore the lowest-cost verified graph" style={{ justifyContent: "center" }}>
                  Restore best
                </Button>
              )}
            </div>
          )}
          {e.topInitializers && e.topInitializers.length > 0 && (
            <details style={{ marginTop: 8 }}>
              <summary style={{
                cursor: "pointer", fontSize: 10.5, color: "var(--text-muted)",
                fontFamily: "var(--mono)", letterSpacing: ".04em",
              }}>top initializers · op histogram</summary>
              <div style={{ marginTop: 6, fontSize: 10.5, fontFamily: "var(--mono)", color: "var(--text-muted)" }}>
                {e.topInitializers.slice(0, 6).map((t) => (
                  <div key={t.name} style={{ display: "flex", justifyContent: "space-between", padding: "1px 0" }}>
                    <span style={{ overflow: "hidden", textOverflow: "ellipsis", whiteSpace: "nowrap", maxWidth: 180 }} title={t.name}>{t.name}</span>
                    <span>{fmtNum(t.elements)} · {fmtBytes(t.bytes)}</span>
                  </div>
                ))}
              </div>
              <div style={{ marginTop: 6, fontSize: 10.5, fontFamily: "var(--mono)", color: "var(--text-muted)", display: "flex", flexWrap: "wrap", gap: 4 }}>
                {Object.entries(e.opHistogram || {}).sort((a, b) => b[1] - a[1]).slice(0, 12).map(([op, c]) => (
                  <span key={op} style={{ padding: "1px 5px", background: "var(--bg-input)", border: "1px solid var(--border)", borderRadius: 3 }}>
                    {op} · {c}
                  </span>
                ))}
              </div>
            </details>
          )}
        </>
      )}
    </div>
  );
}

export function SuggestionsPanel({ suggestions, onHover, onApply, onApplyAllSafe, history, onUndo, efficiency, baseline, best, bestGraphAvailable, efficiencyError, efficiencyRefreshing, onCheck, onRestoreBest, onTry }) {
  const [trying, setTrying] = React.useState(null);
  const totalSavings = suggestions.reduce((sum, s) => sum + (s.severity === "safe" ? s.savings : 0), 0);
  const safeCount = suggestions.filter((s) => s.severity === "safe").length;
  const verifyCount = suggestions.filter((s) => s.severity === "verify" && s.apply).length;
  const handleTry = async (s) => {
    if (!onTry) return;
    setTrying(s.id);
    try { await onTry(s); } finally { setTrying(null); }
  };
  const handleTryAll = async () => {
    const queue = suggestions.filter((s) => s.severity === "verify" && s.apply);
    for (const s of queue) {
      setTrying(s.id);
      try { await onTry(s); } catch {}
    }
    setTrying(null);
  };

  return (
    <div style={{
      flex: 1, display: "flex", flexDirection: "column", minHeight: 0,
      background: "var(--bg-surface)",
    }}>
      <EfficiencyPanel
        efficiency={efficiency}
        baseline={baseline}
        best={best}
        bestGraphAvailable={bestGraphAvailable}
        error={efficiencyError}
        refreshing={efficiencyRefreshing}
        onCheck={onCheck}
        onRestoreBest={onRestoreBest}
      />
      <div style={{ padding: "14px 14px 10px", borderBottom: "1px solid var(--border)" }}>
        <div style={{ display: "flex", alignItems: "center", justifyContent: "space-between" }}>
          <div>
            <div style={{ fontSize: 10.5, fontWeight: 600, letterSpacing: ".08em", textTransform: "uppercase", color: "var(--text-dim)" }}>
              Auto-simplifier
            </div>
            <div style={{ display: "flex", alignItems: "baseline", gap: 8, marginTop: 2 }}>
              <span style={{ fontSize: 18, fontWeight: 600 }}>
                {suggestions.length} {suggestions.length === 1 ? "suggestion" : "suggestions"}
              </span>
              {totalSavings > 0 && <Pill tone="success">−{totalSavings} nodes</Pill>}
            </div>
          </div>
        </div>
        {safeCount > 0 && (
          <Button size="sm" variant="primary" icon="sparkle" onClick={onApplyAllSafe}
            style={{ marginTop: 10, width: "100%", justifyContent: "center" }}>
            Apply all {safeCount} safe rewrite{safeCount > 1 ? "s" : ""}
          </Button>
        )}
        {verifyCount > 0 && onTry && (
          <Button size="sm" variant="ghost" icon="zap" onClick={handleTryAll}
            style={{ marginTop: 6, width: "100%", justifyContent: "center" }}
            disabled={!!trying}
            title="Try each speculative rewrite one by one; commit only those that pass /api/check on every train pair.">
            Try all {verifyCount} speculative rewrite{verifyCount > 1 ? "s" : ""}
          </Button>
        )}
      </div>

      <div style={{ flex: 1, overflowY: "auto", padding: "6px" }}>
        {suggestions.length === 0 && (
          <div style={{
            padding: "40px 20px", textAlign: "center",
            color: "var(--text-dim)", fontSize: 12.5,
          }}>
            <Icon name="check" size={28} style={{ color: "var(--success)", marginBottom: 8 }} />
            <div style={{ fontSize: 14, color: "var(--text)", fontWeight: 500 }}>Graph is clean</div>
            <div style={{ marginTop: 4 }}>No safe rewrites detected.</div>
          </div>
        )}
        {suggestions.map((s) => {
          const meta = CATEGORY_META[s.category] || CATEGORY_META.algebraic;
          return (
            <div key={s.id}
              onMouseEnter={() => onHover(s.affectedNodes)}
              onMouseLeave={() => onHover([])}
              style={{
                padding: "10px 12px", margin: "3px 0", borderRadius: 6,
                background: "var(--bg-input)", border: "1px solid var(--border)",
                cursor: "default",
              }}>
              <div style={{ display: "flex", alignItems: "center", gap: 8, marginBottom: 5 }}>
                <span style={{ width: 6, height: 6, borderRadius: "50%", background: meta.color }} />
                <span style={{
                  fontSize: 9.5, fontWeight: 600, letterSpacing: ".08em", textTransform: "uppercase",
                  color: s.severity === "safe" ? "var(--success)" : s.severity === "verify" ? "var(--accent)" : "var(--warning)",
                  fontFamily: "var(--mono)",
                }}>
                  {s.severity === "safe" ? "safe" : s.severity === "verify" ? "verify" : "speculative"}
                </span>
                <span style={{ fontSize: 9.5, color: "var(--text-dim)", fontFamily: "var(--mono)", textTransform: "uppercase", letterSpacing: ".06em" }}>
                  {meta.label}
                </span>
                <div style={{ flex: 1 }} />
                <span style={{
                  fontFamily: "var(--mono)", fontSize: 11, fontWeight: 600,
                  color: s.savings > 0 ? "var(--success)" : "var(--text-dim)",
                  padding: "1px 6px", borderRadius: 3,
                  background: s.savings > 0 ? "var(--success-bg)" : "transparent",
                }}>
                  −{s.savings}
                </span>
              </div>
              <div style={{ fontSize: 12.5, color: "var(--text)", marginBottom: 3 }}>{s.title}</div>
              <div style={{ fontSize: 11, color: "var(--text-muted)", fontFamily: "var(--mono)" }}>{s.detail}</div>
              {s.apply && s.severity === "safe" && (
                <div style={{ display: "flex", gap: 6, marginTop: 8 }}>
                  <Button size="sm" variant="primary" icon="check" onClick={() => onApply(s)}>Apply</Button>
                  <Button size="sm" variant="ghost" icon="eye" onClick={() => onHover(s.affectedNodes)}>Preview</Button>
                </div>
              )}
              {s.apply && s.severity === "verify" && onTry && (
                <div style={{ display: "flex", gap: 6, marginTop: 8 }}>
                  <Button size="sm" variant="primary" icon="zap" onClick={() => handleTry(s)} disabled={!!trying}>
                    {trying === s.id ? "Trying…" : "Try (verify)"}
                  </Button>
                  <Button size="sm" variant="ghost" icon="eye" onClick={() => onHover(s.affectedNodes)}>Preview</Button>
                </div>
              )}
              {s.apply && s.severity === "verify" && !onTry && (
                <div style={{ display: "flex", gap: 6, marginTop: 8 }}>
                  <Button size="sm" variant="ghost" icon="check" onClick={() => onApply(s)}>Apply (no check)</Button>
                </div>
              )}
              {!s.apply && (
                <div style={{
                  marginTop: 6, padding: "5px 8px", fontSize: 11,
                  background: "var(--warning-bg)", color: "var(--warning)",
                  borderRadius: 4, fontFamily: "var(--mono)",
                  display: "inline-flex", alignItems: "center", gap: 5,
                }}>
                  <Icon name="zap" size={10} /> manual rewrite — see hint
                </div>
              )}
            </div>
          );
        })}
      </div>

      <div style={{
        padding: "10px 12px", borderTop: "1px solid var(--border)",
        background: "var(--bg-app)",
      }}>
        <div style={{ display: "flex", alignItems: "center", justifyContent: "space-between", marginBottom: 8 }}>
          <SectionLabel>History</SectionLabel>
          <Button size="sm" variant="ghost" icon="refresh" onClick={onUndo} disabled={!history?.length}>
            Undo
          </Button>
        </div>
        <div style={{
          display: "flex", alignItems: "center", gap: 5, flexWrap: "wrap",
          fontFamily: "var(--mono)", fontSize: 10.5,
        }}>
          {(!history || history.length === 0) && (
            <span style={{ color: "var(--text-dim)" }}>no rewrites applied yet</span>
          )}
          {history?.map((h, i) => (
            <span key={i} style={{ display: "inline-flex", alignItems: "center", gap: 4 }}>
              <span style={{
                padding: "1px 5px", borderRadius: 3,
                background: i === history.length - 1 ? "var(--accent-bg)" : "var(--bg-input)",
                color: i === history.length - 1 ? "var(--accent)" : "var(--text-muted)",
                border: "1px solid var(--border)",
              }}>{h.nodeCount}</span>
              {i < history.length - 1 && <Icon name="arrowRight" size={9} style={{ color: "var(--text-dim)" }} />}
            </span>
          ))}
        </div>
      </div>
    </div>
  );
}
