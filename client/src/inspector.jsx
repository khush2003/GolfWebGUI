import React, { useMemo, useState } from "react";
import { Icon, Button, Pill, SectionLabel, Kbd, Field, TextInput, Select, Toggle, NumberStepper, ShapeEditor, TensorEditor, inputStyle } from "./components.jsx";
import { NODE_TYPES, NODE_GROUP_COLORS, ARC_COLORS } from "./data.js";

function gridShape(grid) {
  if (!grid?.length) return [0, 0];
  return [grid.length, grid[0]?.length || 0];
}
function gridStats(grid) {
  if (!grid?.length) return { min: "—", max: "—", mean: "—", nnz: "—" };
  let mn = Infinity, mx = -Infinity, sum = 0, n = 0, nz = 0;
  for (const row of grid) for (const v of row) {
    const x = Number(v);
    if (!Number.isFinite(x)) continue;
    if (x < mn) mn = x;
    if (x > mx) mx = x;
    sum += x; n++;
    if (x !== 0) nz++;
  }
  return n ? {
    min: String(mn), max: String(mx),
    mean: (sum / n).toFixed(2), nnz: String(nz),
  } : { min: "—", max: "—", mean: "—", nnz: "—" };
}

const GridView = ({ grid, dtype = "int64", maxCell = 22 }) => {
  if (!grid?.length) {
    return (
      <div style={{
        padding: "18px 12px", border: "1px dashed var(--border)", borderRadius: 5,
        color: "var(--text-dim)", fontSize: 12, textAlign: "center",
      }}>No data yet.</div>
    );
  }
  const [h, w] = gridShape(grid);
  const cell = Math.max(8, Math.min(maxCell, Math.floor(220 / Math.max(h, w, 1))));
  return (
    <div>
      <div style={{ display: "flex", gap: 8, marginBottom: 6 }}>
        <Pill tone="accent">{dtype}</Pill>
        <Pill>{`[${h}, ${w}]`}</Pill>
      </div>
      <div style={{
        display: "grid",
        gridTemplateColumns: `repeat(${w}, ${cell}px)`,
        gridTemplateRows: `repeat(${h}, ${cell}px)`,
        gap: 1, background: "oklch(0.10 0 0)", padding: 2, borderRadius: 4,
        width: "fit-content",
      }}>
        {grid.flatMap((row, r) => row.map((v, c) => (
          <div key={`${r}-${c}`} style={{
            background: ARC_COLORS[Number(v)] || ARC_COLORS[0],
            borderRadius: 1,
          }} />
        )))}
      </div>
    </div>
  );
};

function constantToGrid(value) {
  try {
    const parsed = typeof value === "string" ? JSON.parse(value) : value;
    if (Array.isArray(parsed) && Array.isArray(parsed[0])) return parsed;
    if (Array.isArray(parsed)) return [parsed];
    return [[Number(parsed) || 0]];
  } catch {
    return null;
  }
}

const Stat = ({ label, value, mono }) => (
  <div style={{
    padding: "8px 10px", background: "var(--bg-input)",
    border: "1px solid var(--border)", borderRadius: 5,
  }}>
    <div style={{
      fontSize: 10, fontWeight: 600, letterSpacing: ".06em",
      textTransform: "uppercase", color: "var(--text-dim)",
    }}>{label}</div>
    <div style={{
      fontSize: 16, fontWeight: 600, color: "var(--text)",
      fontFamily: mono ? "var(--mono)" : "var(--font)",
      lineHeight: 1.2, marginTop: 2,
    }}>{value}</div>
  </div>
);

const portRow = {
  display: "flex", alignItems: "center", gap: 6, justifyContent: "space-between",
  padding: "4px 6px", borderRadius: 4,
  fontSize: 12, fontFamily: "var(--mono)", color: "var(--text)",
  background: "var(--bg-input)", border: "1px solid var(--border)",
  marginBottom: 4,
};
const portDot = {
  width: 8, height: 8, borderRadius: "50%",
  background: "var(--bg-canvas)", border: "1.5px solid var(--text-dim)",
};
const chipStyle = {
  padding: "2px 6px", borderRadius: 4,
  background: "var(--bg-input)", color: "var(--text-muted)",
  border: "1px solid var(--border)",
};
const statsGrid = { display: "grid", gridTemplateColumns: "1fr 1fr", gap: 6 };
const closeBtn = {
  background: "transparent", border: "1px solid var(--border)", borderRadius: 5,
  width: 24, height: 24, display: "inline-flex", alignItems: "center", justifyContent: "center",
  color: "var(--text-muted)", cursor: "pointer", flexShrink: 0,
};
const panelStyle = {
  height: "100%", display: "flex", flexDirection: "column",
  background: "var(--bg-surface)",
  minWidth: 0,
};
const panelHeader = { padding: "14px 14px 10px" };

function pickSource(node, inputGrid, expectedOutput, actualOutput, nodeOutputs) {
  if (!node) return { grid: null, source: null, preview: null };
  if (node.type === "Input") {
    return { grid: inputGrid, source: "Current example input", preview: null };
  }
  if (node.type === "Output") {
    if (actualOutput) return { grid: actualOutput, source: "Latest run output", preview: null };
    if (expectedOutput) return { grid: expectedOutput, source: "Expected (run not yet executed)", preview: null };
    return { grid: null, source: null, preview: null };
  }
  const trace = nodeOutputs?.[node.id];
  if (trace) {
    return {
      grid: trace.grid || null,
      source: trace.isApprox ? "Backend trace · argmax projection" : "Backend trace",
      preview: trace,
    };
  }
  if (node.type === "Constant") {
    return { grid: constantToGrid(node.attrs?.value), source: "Constant value", preview: null };
  }
  return { grid: null, source: null, preview: null };
}

function OutputTab({ node, inputGrid, expectedOutput, actualOutput, nodeOutputs, running }) {
  const { grid, source, preview } = useMemo(
    () => pickSource(node, inputGrid, expectedOutput, actualOutput, nodeOutputs),
    [node, inputGrid, expectedOutput, actualOutput, nodeOutputs],
  );

  const stats = useMemo(() => {
    if (preview?.stats) {
      const fmt = (v) => (typeof v === "number" ? (Number.isInteger(v) ? String(v) : v.toFixed(3)) : "—");
      return {
        min: fmt(preview.stats.min),
        max: fmt(preview.stats.max),
        mean: fmt(preview.stats.mean),
        nnz: String(preview.stats.nnz ?? "—"),
      };
    }
    return gridStats(grid);
  }, [grid, preview]);

  const isOutput = node?.type === "Output";
  const shapeLabel = preview?.shape ? `[${preview.shape.join(", ")}]` : null;
  const dtypeLabel = preview?.dtype
    ? preview.dtype
    : isOutput || node?.type === "Input"
      ? "int64 · ARC"
      : "int64";

  return (
    <>
      <SectionLabel>Live output preview</SectionLabel>
      {source && (
        <div style={{ fontSize: 11.5, color: "var(--text-dim)" }}>
          {source}{running ? " · running…" : ""}
        </div>
      )}
      {grid && grid.length ? (
        <GridView grid={grid} dtype={dtypeLabel} />
      ) : node ? (
        <div style={{
          padding: "14px 12px", border: "1px dashed var(--border)", borderRadius: 5,
          color: "var(--text-dim)", fontSize: 12,
        }}>
          {running
            ? "Run in progress — intermediates will populate when it finishes."
            : `No trace yet for ${node.id}. Press Run to compute every node's tensor.`}
        </div>
      ) : (
        <div style={{
          padding: "14px 12px", border: "1px dashed var(--border)", borderRadius: 5,
          color: "var(--text-dim)", fontSize: 12,
        }}>
          Select a node on the canvas to preview its tensor.
        </div>
      )}

      {preview && (
        <div style={{ display: "flex", flexWrap: "wrap", gap: 6, marginTop: 2 }}>
          {shapeLabel && <Pill>{shapeLabel}</Pill>}
          {preview.isApprox && <Pill tone="warning">argmax projection</Pill>}
          {preview.truncated && <Pill tone="warning">truncated 30×30</Pill>}
        </div>
      )}

      <SectionLabel>Stats</SectionLabel>
      <div style={statsGrid}>
        <Stat label="min" value={stats.min} mono />
        <Stat label="max" value={stats.max} mono />
        <Stat label="mean" value={stats.mean} mono />
        <Stat label="nnz" value={stats.nnz} mono />
      </div>

      {node && (
        <>
          <SectionLabel>Pipeline</SectionLabel>
          <div style={{ display: "flex", alignItems: "center", gap: 6, flexWrap: "wrap", fontSize: 11.5, fontFamily: "var(--mono)" }}>
            <span style={chipStyle}>input</span>
            <Icon name="arrowRight" size={10} />
            <span style={{ ...chipStyle, background: "var(--accent-bg)", color: "var(--accent)" }}>{node.id}</span>
            <Icon name="arrowRight" size={10} />
            <span style={chipStyle}>output</span>
          </div>
        </>
      )}
    </>
  );
}

export function Inspector({
  task, graph, selectedNode, onUpdateNode, onDeleteNode, runState, onClose,
  inputGrid = null, expectedOutput = null, actualOutput = null, nodeOutputs = null, running = false,
}) {
  const [tab, setTab] = useState("props");

  if (!selectedNode) {
    return (
      <div style={panelStyle}>
        <div style={{ ...panelHeader, display: "flex", justifyContent: "space-between", alignItems: "flex-start", gap: 8 }}>
          <div>
            <div style={{ fontSize: 10.5, fontWeight: 600, letterSpacing: ".08em", textTransform: "uppercase", color: "var(--text-dim)" }}>Inspector</div>
            <div style={{ display: "flex", alignItems: "baseline", gap: 8, marginTop: 2 }}>
              <span style={{ fontSize: 18, fontWeight: 600 }}>{task?.name || "Task"}</span>
              <span style={{ fontFamily: "var(--mono)", fontSize: 12, color: "var(--text-muted)" }}>task{task?.id}</span>
            </div>
          </div>
          {onClose && (
            <button onClick={onClose} style={closeBtn} title="Close inspector (⌘I)">
              <Icon name="x" size={14} />
            </button>
          )}
        </div>
        <div style={{ padding: "0 14px 14px 14px", display: "flex", flexDirection: "column", gap: 14, overflowY: "auto" }}>
          <div style={{
            padding: "14px 14px", borderRadius: 6, background: "var(--bg-input)",
            border: "1px dashed var(--border)", color: "var(--text-muted)",
            fontSize: 12.5, display: "flex", flexDirection: "column", gap: 8, alignItems: "flex-start",
          }}>
            <Icon name="target" size={18} style={{ color: "var(--accent)" }} />
            <div style={{ color: "var(--text)", fontWeight: 500 }}>No node selected</div>
            <div>Click a node on the canvas to view and edit its parameters, ports, and live output preview.</div>
          </div>

          <SectionLabel>This graph</SectionLabel>
          <div style={statsGrid}>
            <Stat label="Nodes" value={graph.nodes.length} />
            <Stat label="Edges" value={graph.edges.length} />
          </div>

          <div style={{ fontSize: 11.5, color: "var(--text-dim)" }}>
            Press <Kbd>⌘K</Kbd> for the full node library &amp; commands.
          </div>
        </div>
      </div>
    );
  }

  const def = NODE_TYPES[selectedNode.type] || { ins: [], outs: [], attrs: {}, color: "neutral" };
  const groupColor = NODE_GROUP_COLORS[def.color];

  return (
    <div style={panelStyle}>
      <div style={{ ...panelHeader, borderBottom: "1px solid var(--border)" }}>
        <div style={{ display: "flex", alignItems: "flex-start", gap: 10 }}>
          <div style={{
            width: 4, height: 30, borderRadius: 2, marginTop: 2,
            background: groupColor.bar,
          }} />
          <div style={{ flex: 1, minWidth: 0 }}>
            <div style={{
              fontSize: 10.5, fontWeight: 600, letterSpacing: ".08em",
              textTransform: "uppercase", color: "var(--text-dim)",
            }}>Node · {def.color}</div>
            <div style={{ display: "flex", alignItems: "baseline", gap: 8, marginTop: 2 }}>
              <span style={{ fontSize: 18, fontWeight: 600 }}>{selectedNode.type}</span>
              {runState && <Pill tone={runState === "ok" ? "success" : runState === "err" ? "danger" : "warning"}>{runState}</Pill>}
            </div>
            <div style={{ marginTop: 2, fontFamily: "var(--mono)", fontSize: 11.5, color: "var(--text-muted)" }}>
              {selectedNode.id}
            </div>
          </div>
          {onClose && (
            <button onClick={onClose} style={closeBtn} title="Close inspector (⌘I)">
              <Icon name="x" size={14} />
            </button>
          )}
        </div>
      </div>

      <div style={{ display: "flex", gap: 0, borderBottom: "1px solid var(--border)", padding: "0 8px" }}>
        {[
          { id: "props", label: "Properties" },
          { id: "output", label: "Output" },
          { id: "json", label: "JSON" },
        ].map((t) => (
          <button key={t.id} onClick={() => setTab(t.id)} style={{
            padding: "8px 10px", fontSize: 12.5, fontWeight: 500, background: "transparent",
            border: "none", borderBottom: tab === t.id ? "2px solid var(--accent)" : "2px solid transparent",
            color: tab === t.id ? "var(--text)" : "var(--text-muted)",
            cursor: "pointer", marginBottom: -1,
          }}>{t.label}</button>
        ))}
      </div>

      <div style={{ overflowY: "auto", padding: 14, display: "flex", flexDirection: "column", gap: 14, flex: 1 }}>
        {tab === "props" && (
          <>
            <Field label="Label">
              <TextInput value={selectedNode.label || ""}
                onChange={(v) => onUpdateNode(selectedNode.id, { label: v })}
                placeholder={selectedNode.id} />
            </Field>
            <Field label="ID" hint="read-only">
              <TextInput value={selectedNode.id} readOnly style={{ ...inputStyle, opacity: 0.6, cursor: "not-allowed" }} />
            </Field>

            {selectedNode.type === "Constant" && (
              <Field label="value" hint="tensor · JSON">
                <TensorEditor
                  value={selectedNode.attrs?.value ?? "0"}
                  onChange={(v) => onUpdateNode(selectedNode.id, { attrs: { ...selectedNode.attrs, value: typeof v === "string" ? v : JSON.stringify(v) } })}
                />
              </Field>
            )}

            {(def.ins.length > 0 || def.outs.length > 0) && (
              <>
                <SectionLabel>Ports</SectionLabel>
                <div style={{ display: "flex", gap: 14 }}>
                  <div style={{ flex: 1 }}>
                    <div style={{ fontSize: 10, color: "var(--text-dim)", marginBottom: 4, textTransform: "uppercase", letterSpacing: ".06em" }}>Inputs</div>
                    {def.ins.length ? def.ins.map((p) => (
                      <div key={p} style={portRow}>
                        <span style={portDot} />
                        <span>{p}</span>
                      </div>
                    )) : <div style={{ fontSize: 11.5, color: "var(--text-dim)" }}>—</div>}
                  </div>
                  <div style={{ flex: 1 }}>
                    <div style={{ fontSize: 10, color: "var(--text-dim)", marginBottom: 4, textTransform: "uppercase", letterSpacing: ".06em" }}>Outputs</div>
                    {def.outs.length ? def.outs.map((p) => (
                      <div key={p} style={portRow}>
                        <span>{p}</span>
                        <span style={{ ...portDot, background: "var(--accent)", borderColor: "var(--accent-dim)" }} />
                      </div>
                    )) : <div style={{ fontSize: 11.5, color: "var(--text-dim)" }}>—</div>}
                  </div>
                </div>
              </>
            )}

            {Object.keys(def.attrs).length > 0 && (
              <>
                <SectionLabel>Attributes</SectionLabel>
                {Object.entries(def.attrs).map(([key, schema]) => {
                  if (key === "value") return null;
                  const v = selectedNode.attrs?.[key] ?? schema.default;
                  const setAttr = (val) => onUpdateNode(selectedNode.id, { attrs: { ...selectedNode.attrs, [key]: val } });
                  if (schema.type === "enum") {
                    return (
                      <Field key={key} label={key}>
                        <Select value={v} options={schema.options} onChange={setAttr} />
                      </Field>
                    );
                  }
                  if (schema.type === "bool") {
                    return (
                      <Field key={key} label={key}>
                        <Toggle value={!!v} onChange={setAttr} />
                      </Field>
                    );
                  }
                  if (schema.type === "int") {
                    return (
                      <Field key={key} label={key} hint="int">
                        <NumberStepper value={Number(v) || 0} onChange={setAttr} step={1} />
                      </Field>
                    );
                  }
                  if (schema.type === "shape") {
                    return (
                      <Field key={key} label={key} hint={`shape · ${Array.isArray(v) ? v.length : 0}D`}>
                        <ShapeEditor value={v} onChange={setAttr} />
                      </Field>
                    );
                  }
                  if (schema.type === "tensor") {
                    return (
                      <Field key={key} label={key} hint="tensor · JSON">
                        <TensorEditor value={v} onChange={setAttr} />
                      </Field>
                    );
                  }
                  return (
                    <Field key={key} label={key} hint={schema.type}>
                      <TextInput value={String(v ?? "")} onChange={setAttr} />
                    </Field>
                  );
                })}
              </>
            )}

            <div style={{ marginTop: 6, display: "flex", gap: 8 }}>
              <Button variant="danger" icon="trash" onClick={() => onDeleteNode(selectedNode.id)}>Delete</Button>
            </div>
          </>
        )}

        {tab === "output" && (
          <OutputTab
            node={selectedNode}
            inputGrid={inputGrid}
            expectedOutput={expectedOutput}
            actualOutput={actualOutput}
            nodeOutputs={nodeOutputs}
            running={running}
          />
        )}

        {tab === "json" && (
          <pre style={{
            margin: 0, padding: 10, fontSize: 11.5, fontFamily: "var(--mono)",
            background: "var(--bg-input)", border: "1px solid var(--border)",
            borderRadius: 5, color: "var(--text-muted)",
            whiteSpace: "pre-wrap", overflow: "auto",
          }}>{JSON.stringify({
            id: selectedNode.id,
            type: selectedNode.type,
            label: selectedNode.label,
            attrs: selectedNode.attrs,
          }, null, 2)}</pre>
        )}
      </div>
    </div>
  );
}
