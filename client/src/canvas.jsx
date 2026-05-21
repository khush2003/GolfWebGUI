import React, { useCallback, useEffect, useMemo, useRef, useState } from "react";
import ReactFlow, {
  Background,
  MiniMap,
  Handle,
  Position,
  applyNodeChanges,
  applyEdgeChanges,
  useReactFlow,
  ReactFlowProvider,
} from "reactflow";
import "reactflow/dist/style.css";
import { Icon, Kbd } from "./components.jsx";
import { NODE_TYPES, NODE_GROUP_COLORS, ORDER_GROUPS, effectiveInputPorts, effectiveOutputPorts } from "./data.js";

const NODE_W = 168;
const PORT_H = 18;
const HEADER_H = 30;
const BASE_H = 64;

function nodeHeight(ins, outs) {
  const portRows = Math.max(ins.length, outs.length, 1);
  return BASE_H + Math.max(0, portRows - 1) * PORT_H;
}

function portTop(idx) {
  return HEADER_H + idx * PORT_H;
}

const OpNodeView = ({ data, id, selected }) => {
  const def = NODE_TYPES[data.opType] || { ins: [], outs: [], color: "neutral", attrs: {} };
  const ins = data.ins || def.ins;
  const outs = data.outs || def.outs;
  const groupColor = NODE_GROUP_COLORS[def.color];
  const h = nodeHeight(ins, outs);
  const runState = data.runState;
  const highlighted = data.highlighted;

  const ring =
    selected ? "0 0 0 2px var(--accent)" :
      highlighted ? "0 0 0 2px oklch(0.80 0.13 75 / 0.95), 0 0 24px oklch(0.80 0.13 75 / 0.45)" :
        runState === "ok" ? "0 0 0 2px oklch(0.76 0.13 155 / 0.85)" :
          runState === "err" ? "0 0 0 2px oklch(0.70 0.16 25 / 0.85)" :
            runState === "running" ? "0 0 0 2px oklch(0.78 0.13 75 / 0.9)" : "none";

  const subtitle = data.label && data.label !== id ? data.label : id;

  return (
    <div style={{
      width: NODE_W, height: h, position: "relative",
      background: "var(--bg-elevated)",
      border: "1px solid var(--border)",
      borderRadius: 6,
      boxShadow: ring + (selected ? ", 0 6px 16px -8px oklch(0.78 0.12 200 / 0.55)" : ""),
      userSelect: "none",
      transition: "box-shadow 120ms",
    }}>
      <div style={{
        position: "absolute", left: 0, top: 0, bottom: 0, width: 3,
        background: groupColor.bar, borderRadius: "6px 0 0 6px",
      }} />
      <div style={{
        padding: "6px 8px 4px 11px",
        borderBottom: "1px solid var(--border-soft)",
        display: "flex", alignItems: "center", justifyContent: "space-between",
        gap: 6, height: HEADER_H,
      }}>
        <div style={{ display: "flex", alignItems: "baseline", gap: 6, minWidth: 0 }}>
          <span style={{ fontSize: 13, fontWeight: 600, color: "var(--text)" }}>{data.opType}</span>
          <span style={{
            fontFamily: "var(--mono)", fontSize: 10.5, color: "var(--text-dim)",
            overflow: "hidden", textOverflow: "ellipsis", whiteSpace: "nowrap",
          }}>{subtitle}</span>
        </div>
        {runState === "running" && (
          <span className="ng-pulse-dot" style={{
            width: 8, height: 8, borderRadius: "50%",
            background: "var(--warning)",
          }} />
        )}
      </div>

      {ins.map((name, i) => (
        <React.Fragment key={`in-${name}-${i}`}>
          <Handle
            type="target"
            position={Position.Left}
            id={name}
            style={{
              top: portTop(i),
              width: 10, height: 10,
              background: "var(--bg-canvas)",
              border: "1.5px solid var(--text-dim)",
              borderRadius: "50%",
              left: -5,
            }}
          />
          <span style={{
            position: "absolute",
            left: 11, top: portTop(i) - 7,
            fontSize: 10.5, color: "var(--text-dim)", fontFamily: "var(--mono)",
            letterSpacing: ".02em", pointerEvents: "none",
          }}>{name}</span>
        </React.Fragment>
      ))}

      {outs.map((name, i) => (
        <React.Fragment key={`out-${name}-${i}`}>
          <Handle
            type="source"
            position={Position.Right}
            id={name}
            style={{
              top: portTop(i),
              width: 10, height: 10,
              background: "var(--accent)",
              border: "1.5px solid var(--bg-canvas)",
              boxShadow: "0 0 0 1px var(--accent-dim)",
              borderRadius: "50%",
              right: -5,
            }}
          />
          <span style={{
            position: "absolute",
            right: 11, top: portTop(i) - 7,
            fontSize: 10.5, color: "var(--text-dim)", fontFamily: "var(--mono)",
            letterSpacing: ".02em", pointerEvents: "none",
          }}>{name}</span>
        </React.Fragment>
      ))}
    </div>
  );
};

const nodeTypes = { op: OpNodeView };

function NodeLibrary({ onAdd, onClose }) {
  const [q, setQ] = useState("");
  const items = useMemo(() => {
    const all = Object.entries(NODE_TYPES).map(([type, def]) => ({
      type, color: def.color, ins: def.ins.length, outs: def.outs.length,
    }));
    const ql = q.toLowerCase();
    return ql ? all.filter((i) => i.type.toLowerCase().includes(ql)) : all;
  }, [q]);
  const groups = {};
  items.forEach((i) => { (groups[i.color] = groups[i.color] || []).push(i); });

  return (
    <div onClick={onClose} style={{
      position: "absolute", inset: 0, background: "oklch(0.10 0 0 / 0.55)", zIndex: 20,
      display: "flex", alignItems: "flex-end", justifyContent: "flex-start", padding: 14,
    }}>
      <div onClick={(e) => e.stopPropagation()} style={{
        width: 460, maxHeight: "calc(100% - 80px)", marginLeft: 110,
        background: "var(--bg-surface)", border: "1px solid var(--border)", borderRadius: 10,
        boxShadow: "0 18px 48px -12px oklch(0.05 0 0 / 0.7)", overflow: "hidden",
        display: "flex", flexDirection: "column",
      }}>
        <div style={{ padding: "10px 12px", display: "flex", alignItems: "center", gap: 8, borderBottom: "1px solid var(--border)" }}>
          <Icon name="search" size={14} />
          <input autoFocus value={q} onChange={(e) => setQ(e.target.value)}
            placeholder="Search ONNX node types…"
            style={{
              flex: 1, background: "transparent", border: "none", outline: "none",
              fontSize: 13, color: "var(--text)",
            }} />
          <Kbd>esc</Kbd>
        </div>
        <div style={{ overflowY: "auto", padding: "8px 4px", flex: 1 }}>
          {ORDER_GROUPS.filter(([c]) => groups[c]).map(([c, label]) => (
            <div key={c} style={{ padding: "6px 10px" }}>
              <div style={{
                fontSize: 10.5, fontWeight: 600, letterSpacing: ".08em", textTransform: "uppercase",
                color: "var(--text-dim)", marginBottom: 6,
              }}>{label}</div>
              <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 4 }}>
                {groups[c].map((it) => (
                  <button key={it.type} onClick={() => onAdd(it.type)} style={{
                    display: "flex", alignItems: "center", gap: 8, padding: "7px 9px", borderRadius: 5,
                    background: "transparent", border: "1px solid transparent", textAlign: "left",
                    cursor: "pointer", color: "var(--text)",
                  }}
                    onMouseEnter={(e) => { e.currentTarget.style.background = "var(--bg-elevated)"; e.currentTarget.style.borderColor = "var(--border)"; }}
                    onMouseLeave={(e) => { e.currentTarget.style.background = "transparent"; e.currentTarget.style.borderColor = "transparent"; }}>
                    <span style={{ width: 3, height: 14, borderRadius: 2, background: NODE_GROUP_COLORS[it.color].bar }} />
                    <span style={{ fontSize: 12.5, fontWeight: 500 }}>{it.type}</span>
                    <span style={{ marginLeft: "auto", fontSize: 10, fontFamily: "var(--mono)", color: "var(--text-dim)" }}>
                      {it.ins}→{it.outs}
                    </span>
                  </button>
                ))}
              </div>
            </div>
          ))}
        </div>
      </div>
    </div>
  );
}

const hudBtnStyle = {
  width: 36, height: 30,
  display: "inline-flex", alignItems: "center", justifyContent: "center",
  background: "oklch(0.16 0.008 250 / 0.85)",
  border: "1px solid var(--border)",
  borderRadius: 6, cursor: "pointer", color: "var(--text-muted)",
  backdropFilter: "blur(8px)",
};

const CONNECTION_LINE_STYLE = { stroke: "var(--accent)", strokeWidth: 1.8, strokeDasharray: "5 4" };
const DEFAULT_EDGE_OPTIONS = { type: "default" };
const SNAP_GRID = [8, 8];

function CanvasInner({
  graph, runStates, highlightedNodes,
  onSelectNode, onSelectEdge, onUpdateNode, onMoveNode,
  onCreateEdge, fullscreen, onToggleFullscreen, onAddNode,
}) {
  const rfRef = useRef(null);
  const [showLib, setShowLib] = useState(false);
  const { fitView, zoomIn, zoomOut, getViewport } = useReactFlow();
  const [zoomLevel, setZoomLevel] = useState(85);

  const derivedNodes = useMemo(() => graph.nodes.map((n) => ({
    id: n.id,
    type: "op",
    position: { x: n.x, y: n.y },
    data: {
      opType: n.type,
      label: n.label,
      ins: effectiveInputPorts(n),
      outs: effectiveOutputPorts(n),
      runState: runStates?.[n.id],
      highlighted: highlightedNodes?.includes(n.id),
    },
    selected: graph.selectedNodeId === n.id,
  })), [graph.nodes, graph.selectedNodeId, runStates, highlightedNodes]);

  const derivedEdges = useMemo(() => graph.edges.map((e) => {
    const [from, fromPort] = e.from.split(":");
    const [to, toPort] = e.to.split(":");
    const key = `${e.from}→${e.to}`;
    return {
      id: e.id || key,
      source: from,
      sourceHandle: fromPort,
      target: to,
      targetHandle: toPort,
      type: "default",
      animated: false,
      selected: graph.selectedEdgeKey === key,
      style: {
        stroke: graph.selectedEdgeKey === key ? "var(--accent)" : "var(--accent-edge)",
        strokeWidth: graph.selectedEdgeKey === key ? 2.4 : 1.4,
      },
      markerEnd: { type: "arrowclosed", color: "var(--accent-edge)" },
    };
  }), [graph.edges, graph.selectedEdgeKey]);

  const [rfNodes, setRfNodes] = useState(derivedNodes);
  const [rfEdges, setRfEdges] = useState(derivedEdges);
  const draggingRef = useRef(false);

  useEffect(() => {
    if (draggingRef.current && rfNodes.length === derivedNodes.length) return;
    setRfNodes(derivedNodes);
  }, [derivedNodes, rfNodes.length]);

  useEffect(() => {
    setRfEdges(derivedEdges);
  }, [derivedEdges]);

  const onNodesChange = useCallback((changes) => {
    setRfNodes((ns) => applyNodeChanges(changes, ns));
  }, []);

  const onEdgesChange = useCallback((changes) => {
    setRfEdges((es) => applyEdgeChanges(changes, es));
  }, []);

  const onNodeDragStart = useCallback(() => {
    draggingRef.current = true;
  }, []);

  const onNodeDragStop = useCallback((_, node) => {
    draggingRef.current = false;
    if (node?.position) onMoveNode(node.id, node.position.x, node.position.y);
  }, [onMoveNode]);

  const onConnect = useCallback((conn) => {
    if (!conn.source || !conn.target) return;
    if (conn.source === conn.target) return;
    onCreateEdge(`${conn.source}:${conn.sourceHandle || "y"}`, `${conn.target}:${conn.targetHandle || "input"}`);
  }, [onCreateEdge]);

  const onNodeClick = useCallback((_, node) => onSelectNode(node.id), [onSelectNode]);
  const onEdgeClick = useCallback((_, edge) => {
    onSelectEdge(`${edge.source}:${edge.sourceHandle}→${edge.target}:${edge.targetHandle}`);
  }, [onSelectEdge]);

  const onPaneClick = useCallback(() => {
    onSelectNode(null);
  }, [onSelectNode]);

  const onMoveEnd = useCallback(() => {
    const vp = getViewport();
    setZoomLevel(Math.round(vp.zoom * 100));
  }, [getViewport]);

  const onFit = useCallback(() => fitView({ padding: 0.2, duration: 200 }), [fitView]);

  return (
    <div style={{ position: "relative", width: "100%", height: "100%" }}>
      <ReactFlow
        ref={rfRef}
        nodes={rfNodes}
        edges={rfEdges}
        nodeTypes={nodeTypes}
        onNodesChange={onNodesChange}
        onEdgesChange={onEdgesChange}
        onNodeDragStart={onNodeDragStart}
        onNodeDragStop={onNodeDragStop}
        onConnect={onConnect}
        onNodeClick={onNodeClick}
        onEdgeClick={onEdgeClick}
        onPaneClick={onPaneClick}
        onMoveEnd={onMoveEnd}
        connectionLineStyle={CONNECTION_LINE_STYLE}
        defaultEdgeOptions={DEFAULT_EDGE_OPTIONS}
        fitView
        proOptions={{ hideAttribution: true }}
        snapToGrid
        snapGrid={SNAP_GRID}
      >
        <Background gap={24} size={1} color="var(--grid)" />
        <MiniMap
          pannable zoomable
          nodeColor={(n) => n.selected ? "oklch(0.78 0.12 200)" : "oklch(0.55 0.012 250)"}
          maskColor="oklch(0 0 0 / 0.4)"
          style={{ background: "oklch(0.16 0.008 250 / 0.92)", border: "1px solid var(--border)", borderRadius: 6, width: 160, height: 96 }}
        />
      </ReactFlow>

      <div style={{
        position: "absolute", right: 14, bottom: 14, display: "flex", flexDirection: "column", gap: 6, zIndex: 5,
      }}>
        <button onClick={onToggleFullscreen} style={hudBtnStyle} title={fullscreen ? "Exit fullscreen (Esc)" : "Fullscreen graph (⌘F)"}>
          <Icon name={fullscreen ? "fullscreenExit" : "fullscreenEnter"} />
        </button>
        <button onClick={onFit} style={hudBtnStyle} title="Fit to view">
          <Icon name="fit" />
        </button>
        <div style={{
          display: "flex", flexDirection: "column",
          background: "oklch(0.16 0.008 250 / 0.85)",
          border: "1px solid var(--border)", borderRadius: 6,
          backdropFilter: "blur(8px)", overflow: "hidden",
        }}>
          <button onClick={() => zoomIn({ duration: 150 })} style={{ ...hudBtnStyle, background: "transparent", border: "none", borderRadius: 0, height: 28 }} title="Zoom in">
            <Icon name="plus" />
          </button>
          <div style={{ height: 1, background: "var(--border)" }} />
          <div style={{
            fontFamily: "var(--mono)", fontSize: 10, textAlign: "center",
            color: "var(--text-dim)", padding: "2px 0", userSelect: "none",
          }}>{zoomLevel}%</div>
          <div style={{ height: 1, background: "var(--border)" }} />
          <button onClick={() => zoomOut({ duration: 150 })} style={{ ...hudBtnStyle, background: "transparent", border: "none", borderRadius: 0, height: 28 }} title="Zoom out">
            <Icon name="minus" />
          </button>
        </div>
      </div>

      <div style={{
        position: "absolute", left: 14, bottom: 14, display: "flex", alignItems: "flex-end", gap: 10, zIndex: 5,
      }}>
        <button
          onClick={() => setShowLib((s) => !s)}
          style={{
            ...hudBtnStyle, width: "auto", padding: "0 14px", height: 36, gap: 6,
            background: "var(--accent)", borderColor: "transparent",
            color: "oklch(0.18 0.02 200)", fontWeight: 600,
          }}>
          <Icon name="plus" size={15} strokeWidth={2.2} />
          Add Node
        </button>
      </div>

      {fullscreen && (
        <div style={{
          position: "absolute", left: 14, top: 14, display: "flex", gap: 6, zIndex: 5,
        }}>
          <div style={{
            padding: "5px 10px", background: "oklch(0.16 0.008 250 / 0.85)",
            border: "1px solid var(--border)", borderRadius: 6, fontSize: 11,
            color: "var(--text-muted)", fontFamily: "var(--mono)", backdropFilter: "blur(8px)",
          }}>
            fullscreen · esc to exit
          </div>
        </div>
      )}

      {showLib && (
        <NodeLibrary
          onClose={() => setShowLib(false)}
          onAdd={(type) => {
            const vp = getViewport();
            const cx = (-vp.x + window.innerWidth / 2) / vp.zoom;
            const cy = (-vp.y + window.innerHeight / 2) / vp.zoom;
            onAddNode({ type, x: cx - NODE_W / 2, y: cy - 30 });
            setShowLib(false);
          }}
        />
      )}
    </div>
  );
}

export function GraphCanvas(props) {
  return (
    <ReactFlowProvider>
      <CanvasInner {...props} />
    </ReactFlowProvider>
  );
}
