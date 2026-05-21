import React, { useEffect, useMemo, useRef, useState } from "react";
import { Icon, Button, Pill, StatusDot, SectionLabel, Kbd, MiniGrid, Field, TextInput, Select, Toggle, inputStyle } from "./components.jsx";
import { ARC_COLORS, ARC_COLOR_NAMES, NODE_TYPES } from "./data.js";

const breadcrumbBtn = {
  background: "transparent", border: "none", color: "var(--text-muted)",
  fontSize: 13, cursor: "pointer", display: "inline-flex", alignItems: "center", gap: 5,
  padding: "3px 6px", borderRadius: 4,
};

const menuItem = {
  display: "flex", alignItems: "center", gap: 8, width: "100%",
  padding: "7px 10px", borderRadius: 5, background: "transparent",
  border: "none", color: "var(--text-muted)", textAlign: "left", cursor: "pointer", fontSize: 12.5,
};

export const TopBar = ({
  workspace, workspaces, taskId, taskName, dirty,
  onSave, onSaveAll, onRun, running, onCompile,
  onImport, onExport, onOpenPalette, onOpenWorkspaces, onPickWorkspace,
  onExportZip, submissionCount, totalScore,
}) => {
  const [wsOpen, setWsOpen] = useState(false);
  const wsRef = useRef(null);
  useEffect(() => {
    if (!wsOpen) return;
    const off = (e) => { if (!wsRef.current?.contains(e.target)) setWsOpen(false); };
    document.addEventListener("mousedown", off);
    return () => document.removeEventListener("mousedown", off);
  }, [wsOpen]);

  return (
    <div style={{
      height: 48, display: "flex", alignItems: "center", gap: 8,
      padding: "0 12px 0 14px",
      background: "var(--bg-surface)",
      borderBottom: "1px solid var(--border)",
      flexShrink: 0, position: "relative",
    }}>
      <div style={{ display: "flex", alignItems: "center", gap: 8 }}>
        <div style={{
          width: 24, height: 24, borderRadius: 6,
          background: "linear-gradient(135deg, var(--accent), oklch(0.55 0.12 260))",
          display: "flex", alignItems: "center", justifyContent: "center",
          color: "oklch(0.13 0.02 250)", fontWeight: 700, fontSize: 11,
          fontFamily: "var(--mono)", letterSpacing: "-.02em",
        }}>ng</div>
        <span style={{ fontWeight: 600, fontSize: 14, letterSpacing: "-.005em" }}>NeuroGolf Lab</span>
      </div>

      <div style={{ width: 1, height: 18, background: "var(--border)", marginLeft: 6 }} />

      <div ref={wsRef} style={{ position: "relative" }}>
        <button onClick={() => setWsOpen((o) => !o)} style={{
          ...breadcrumbBtn,
          background: wsOpen ? "var(--bg-elevated)" : "transparent",
          borderRadius: 5,
        }}>
          <Icon name="folder" size={13} />
          {workspace?.name || "workspace"}
          <Icon name="chevronDown" size={11} />
        </button>
        {wsOpen && (
          <div style={{
            position: "absolute", top: 32, left: 0, zIndex: 50, minWidth: 260,
            background: "var(--bg-elevated)", border: "1px solid var(--border)", borderRadius: 8,
            boxShadow: "0 12px 28px -8px oklch(0.05 0 0 / 0.65)", padding: 5,
          }}>
            <div style={{
              padding: "6px 10px", fontSize: 10.5, fontWeight: 600, letterSpacing: ".08em",
              textTransform: "uppercase", color: "var(--text-dim)",
            }}>Workspaces</div>
            {workspaces.map((w) => (
              <button key={w.id} onClick={() => { onPickWorkspace(w.id); setWsOpen(false); }} style={{
                display: "flex", alignItems: "center", gap: 8, width: "100%",
                padding: "7px 10px", borderRadius: 5, background: "transparent",
                border: "none", color: "var(--text)", textAlign: "left", cursor: "pointer", fontSize: 12.5,
              }}
                onMouseEnter={(e) => e.currentTarget.style.background = "var(--bg-hover)"}
                onMouseLeave={(e) => e.currentTarget.style.background = "transparent"}>
                <Icon name="folder" size={13} style={{ color: w.current ? "var(--accent)" : "var(--text-muted)" }} />
                <div style={{ flex: 1, minWidth: 0 }}>
                  <div style={{ overflow: "hidden", textOverflow: "ellipsis", whiteSpace: "nowrap" }}>{w.name}</div>
                  <div style={{ fontFamily: "var(--mono)", fontSize: 10, color: "var(--text-dim)" }}>{w.tasks} tasks · {w.lastSaved}</div>
                </div>
                {w.current && <Icon name="check" size={13} style={{ color: "var(--accent)" }} />}
              </button>
            ))}
            <div style={{ height: 1, background: "var(--border)", margin: "4px 6px" }} />
            <button onClick={() => { onOpenWorkspaces("new"); setWsOpen(false); }} style={menuItem}>
              <Icon name="plus" size={13} /> New workspace…
            </button>
            <button onClick={() => { onOpenWorkspaces("manage"); setWsOpen(false); }} style={menuItem}>
              <Icon name="gear" size={13} /> Manage workspaces…
            </button>
          </div>
        )}
      </div>

      <span style={{ color: "var(--text-dim)" }}>/</span>
      <div style={{ display: "flex", alignItems: "baseline", gap: 6 }}>
        <span style={{ fontFamily: "var(--mono)", fontSize: 13, color: "var(--text)" }}>task{taskId}</span>
        <span style={{ fontSize: 13, color: "var(--text-muted)" }}>{taskName}</span>
      </div>

      <div style={{ flex: 1 }} />

      <button onClick={onOpenPalette} style={{
        display: "inline-flex", alignItems: "center", gap: 8,
        height: 28, padding: "0 8px 0 10px", borderRadius: 6,
        background: "var(--bg-input)", border: "1px solid var(--border)",
        color: "var(--text-dim)", fontSize: 12.5, cursor: "pointer", minWidth: 240,
      }}>
        <Icon name="search" size={13} />
        <span>Search nodes, tasks, commands…</span>
        <span style={{ marginLeft: "auto" }}><Kbd>⌘K</Kbd></span>
      </button>

      <div style={{ display: "flex", alignItems: "center", gap: 6, color: "var(--text-dim)", fontSize: 12, marginLeft: 6, padding: "0 4px" }}>
        <span style={{ width: 6, height: 6, borderRadius: "50%", background: dirty ? "var(--warning)" : "var(--success)" }} />
        <span>{dirty ? "Unsaved" : "Saved"}</span>
      </div>

      <Button variant="ghost" icon="save" onClick={onSave} title="Save current task (⌘S · ⌘⇧S = save all)">Save</Button>
      <Button variant="ghost" icon="upload" onClick={onImport}>Import</Button>
      <Button variant="ghost" icon="download" onClick={onExport}>Export</Button>
      {onExportZip && (
        <Button
          variant="ghost"
          icon="download"
          onClick={onExportZip}
          title={`Bundle ${submissionCount || 0} verified best graphs as competition submission zip${totalScore != null ? ` · Σ score ${totalScore.toFixed(1)}` : ""}`}
        >
          Submission · {submissionCount || 0}
        </Button>
      )}

      <div style={{ width: 1, height: 20, background: "var(--border)" }} />

      <Button variant="primary" icon={running ? "pause" : "play"} onClick={onRun}>
        {running ? "Stop" : "Run on example"}
      </Button>
    </div>
  );
};

export const LeftRail = ({ tasks, currentTaskId, onSelectTask, workspace, bestByTask = {} }) => {
  const [q, setQ] = useState("");
  const [filter, setFilter] = useState("all");

  const filtered = useMemo(() => tasks.filter((t) => {
    if (filter !== "all" && t.status !== filter) return false;
    if (q && !(`${t.id} ${t.name}`.toLowerCase().includes(q.toLowerCase()))) return false;
    return true;
  }), [tasks, q, filter]);

  const counts = useMemo(() => ({
    all: tasks.length,
    passing: tasks.filter((t) => t.status === "passing").length,
    failing: tasks.filter((t) => t.status === "failing").length,
    editing: tasks.filter((t) => t.status === "editing").length,
    untested: tasks.filter((t) => t.status === "untested").length,
  }), [tasks]);

  return (
    <div style={{
      width: 220, background: "var(--bg-surface)",
      borderRight: "1px solid var(--border)",
      display: "flex", flexDirection: "column", flexShrink: 0, height: "100%",
    }}>
      <div style={{ padding: "12px 12px 8px", borderBottom: "1px solid var(--border)" }}>
        <div style={{ display: "flex", alignItems: "center", justifyContent: "space-between" }}>
          <SectionLabel>Tasks</SectionLabel>
          <span style={{ fontFamily: "var(--mono)", fontSize: 10.5, color: "var(--text-dim)" }}>{tasks.length}</span>
        </div>
        <div style={{ position: "relative", marginTop: 8 }}>
          <Icon name="search" size={13} style={{ position: "absolute", left: 8, top: 7, color: "var(--text-dim)" }} />
          <input value={q} onChange={(e) => setQ(e.target.value)} placeholder="Filter…"
            style={{ ...inputStyle, paddingLeft: 28, height: 28, fontFamily: "var(--font)" }} />
        </div>
        <div style={{ display: "flex", gap: 3, marginTop: 8, flexWrap: "wrap" }}>
          {[
            { id: "all", label: "All", tone: null },
            { id: "editing", label: "WIP", tone: "editing" },
            { id: "passing", label: "Pass", tone: "passing" },
            { id: "failing", label: "Fail", tone: "failing" },
            { id: "untested", label: "New", tone: "untested" },
          ].map((f) => (
            <button key={f.id} onClick={() => setFilter(f.id)} style={{
              display: "inline-flex", alignItems: "center", gap: 4,
              padding: "2px 6px", borderRadius: 999, fontSize: 10.5, fontWeight: 500,
              background: filter === f.id ? "var(--bg-elevated)" : "transparent",
              color: filter === f.id ? "var(--text)" : "var(--text-muted)",
              border: "1px solid", borderColor: filter === f.id ? "var(--border-strong)" : "var(--border)",
              cursor: "pointer",
            }}>
              {f.tone && <StatusDot status={f.tone} size={5} />}
              {f.label}
              <span style={{ fontFamily: "var(--mono)", fontSize: 9.5, color: "var(--text-dim)" }}>{counts[f.id]}</span>
            </button>
          ))}
        </div>
      </div>

      <div style={{ flex: 1, overflowY: "auto", padding: "6px 0" }}>
        {filtered.map((t) => {
          const best = bestByTask[t.id];
          const score = best?.score;
          return (
          <button key={t.id} onClick={() => onSelectTask(t.id)} style={{
            display: "flex", alignItems: "center", gap: 9,
            width: "calc(100% - 12px)", margin: "0 6px", padding: "7px 8px",
            border: "none", background: t.id === currentTaskId ? "var(--bg-elevated)" : "transparent",
            color: "inherit", textAlign: "left", cursor: "pointer",
            borderRadius: 5, fontSize: 12.5, position: "relative",
          }}
            onMouseEnter={(e) => t.id !== currentTaskId && (e.currentTarget.style.background = "var(--bg-input)")}
            onMouseLeave={(e) => t.id !== currentTaskId && (e.currentTarget.style.background = "transparent")}>
            {t.id === currentTaskId && (
              <span style={{ position: "absolute", left: -6, top: 6, bottom: 6, width: 2, background: "var(--accent)", borderRadius: 1 }} />
            )}
            <StatusDot status={t.status} />
            <span style={{ fontFamily: "var(--mono)", fontSize: 11, color: "var(--text-muted)", minWidth: 26 }}>{t.id}</span>
            <span style={{ flex: 1, overflow: "hidden", textOverflow: "ellipsis", whiteSpace: "nowrap", color: t.id === currentTaskId ? "var(--text)" : "var(--text-muted)" }}>{t.name}</span>
            {Number.isFinite(score) ? (
              <span style={{
                fontFamily: "var(--mono)", fontSize: 10.5, fontWeight: 600,
                color: score >= 15 ? "var(--success)" : score >= 10 ? "var(--accent)" : "var(--warning)",
              }} title={`cost ${best?.cost} · ${best?.bytes} bytes · ${best?.parameters} params`}>
                {score.toFixed(1)}
              </span>
            ) : t.score !== null && t.score !== undefined ? (
              <span style={{
                fontFamily: "var(--mono)", fontSize: 10,
                color: "var(--text-dim)",
              }}>—</span>
            ) : null}
          </button>
          );
        })}
      </div>

      <div style={{
        padding: "10px 12px", borderTop: "1px solid var(--border)",
        display: "flex", alignItems: "center", gap: 8,
      }}>
        <span style={{
          display: "inline-flex", alignItems: "center", justifyContent: "center",
          width: 22, height: 22, borderRadius: "50%",
          background: "linear-gradient(135deg, oklch(0.65 0.15 200), oklch(0.55 0.15 280))",
          color: "oklch(0.15 0.02 250)", fontWeight: 600, fontSize: 11,
        }}>ng</span>
        <div style={{ flex: 1, fontSize: 12, color: "var(--text-muted)", minWidth: 0 }}>
          <div style={{ color: "var(--text)", overflow: "hidden", textOverflow: "ellipsis", whiteSpace: "nowrap" }}>{workspace?.name}</div>
          <div style={{ fontFamily: "var(--mono)", fontSize: 10, color: "var(--text-dim)" }}>{workspace?.tasks ?? 0} saved</div>
        </div>
      </div>
    </div>
  );
};

const GridPanel = ({ title, subtitle, data, ghost, running, progress, diffMask }) => {
  const h = data?.length || 0;
  const w = data?.[0]?.length || 1;
  const cell = Math.max(8, Math.min(20, Math.floor(160 / Math.max(h, w, 1))));

  return (
    <div style={{ display: "flex", flexDirection: "column", alignItems: "center", gap: 7 }}>
      <div style={{ display: "flex", alignItems: "baseline", gap: 6 }}>
        <span style={{
          fontSize: 10.5, fontWeight: 600, letterSpacing: ".08em",
          textTransform: "uppercase", color: "var(--text-dim)",
        }}>{title}</span>
        <span style={{ fontSize: 10.5, color: "var(--text-dim)", fontFamily: "var(--mono)" }}>{subtitle}</span>
      </div>
      <div style={{
        position: "relative", padding: 4,
        background: "oklch(0.08 0 0)", borderRadius: 5,
        border: `1px solid ${ghost ? "var(--border)" : "var(--border-strong)"}`,
        opacity: ghost ? 0.35 : 1,
        filter: running ? "blur(2px)" : "none",
        transition: "filter 200ms, opacity 200ms",
      }}>
        <div style={{
          display: "grid",
          gridTemplateColumns: `repeat(${w}, ${cell}px)`,
          gridTemplateRows: `repeat(${h}, ${cell}px)`,
          gap: 1,
          background: "oklch(0.55 0.005 250)",
          padding: 1,
        }}>
          {(data || []).flatMap((row, r) => row.map((v, c) => {
            const isDiff = diffMask?.[r]?.[c];
            return (
              <div key={`${r}-${c}`} style={{
                background: ARC_COLORS[v] || ARC_COLORS[0],
                position: "relative",
                boxShadow: isDiff ? "inset 0 0 0 2px var(--danger)" : "none",
              }}>
                {isDiff && (
                  <span style={{
                    position: "absolute", inset: 0,
                    background: "oklch(0.70 0.16 25 / 0.25)",
                  }} />
                )}
              </div>
            );
          }))}
        </div>
        {running && progress != null && (
          <div style={{
            position: "absolute", left: 6, right: 6, bottom: 6, height: 3,
            background: "oklch(0.15 0 0 / 0.6)", borderRadius: 2, overflow: "hidden",
          }}>
            <div style={{ height: "100%", width: `${progress * 100}%`, background: "var(--accent)", transition: "width 150ms" }} />
          </div>
        )}
        {ghost && (
          <div style={{
            position: "absolute", inset: 0, display: "flex", alignItems: "center", justifyContent: "center",
            color: "var(--text-dim)", fontSize: 10.5, fontFamily: "var(--mono)",
            textTransform: "uppercase", letterSpacing: ".1em",
          }}>press run</div>
        )}
      </div>
      <div style={{ fontFamily: "var(--mono)", fontSize: 10, color: "var(--text-dim)" }}>
        {h} × {w}
      </div>
    </div>
  );
};

const Arrow = () => (
  <Icon name="arrowRight" size={22} style={{ color: "var(--text-dim)", flexShrink: 0, marginTop: 24 }} />
);

const Equals = ({ ok, pending }) => (
  <div style={{
    display: "flex", flexDirection: "column", alignItems: "center", gap: 2,
    color: pending ? "var(--text-dim)" : ok ? "var(--success)" : "var(--danger)",
    flexShrink: 0, marginTop: 18,
  }}>
    <span style={{ fontSize: 22, fontWeight: 700, lineHeight: 1, fontFamily: "var(--mono)" }}>
      {pending ? "≟" : ok ? "=" : "≠"}
    </span>
    <span style={{ fontSize: 9.5, fontFamily: "var(--mono)", textTransform: "uppercase", letterSpacing: ".1em" }}>
      {pending ? "pending" : ok ? "match" : "diff"}
    </span>
  </div>
);

export const Workbench = ({
  task, pairs, selectedExample, onSelectExample,
  actualOutput, running, runProgress, onRun, onClearOutput,
}) => {
  const allExamples = useMemo(() => [
    ...(pairs?.train || []).map((p, i) => ({ kind: "train", index: i, label: `Train ${String(i + 1).padStart(2, "0")}`, pair: p })),
    ...(pairs?.test || []).map((p, i) => ({ kind: "test", index: i, label: `Test ${String(i + 1).padStart(2, "0")}`, pair: p })),
    ...(pairs?.extra || []).map((p, i) => ({ kind: "extra", index: i, label: `Extra ${String(i + 1).padStart(2, "0")}`, pair: p })),
  ], [pairs]);

  const current = allExamples.find((e) => e.kind === selectedExample.kind && e.index === selectedExample.index) || allExamples[0];
  const pair = current?.pair || null;

  const diffMask = useMemo(() => {
    if (!actualOutput || !pair?.output) return null;
    return pair.output.map((row, r) =>
      row.map((v, c) => actualOutput[r]?.[c] !== v));
  }, [actualOutput, pair]);

  const mismatch = useMemo(() => {
    if (!diffMask) return null;
    let total = 0, bad = 0;
    diffMask.forEach((row) => row.forEach((b) => { total++; if (b) bad++; }));
    return { total, bad, ok: bad === 0 };
  }, [diffMask]);

  if (!current) {
    return (
      <div style={{
        flexShrink: 0, padding: "20px 14px", textAlign: "center",
        background: "var(--bg-surface)", borderBottom: "1px solid var(--border)",
        color: "var(--text-dim)", fontSize: 12.5,
      }}>
        Loading task examples…
      </div>
    );
  }

  return (
    <div style={{
      flexShrink: 0, display: "flex", flexDirection: "column",
      background: "var(--bg-surface)",
      borderBottom: "1px solid var(--border)",
    }}>
      <div style={{ display: "flex", alignItems: "center", gap: 10, padding: "8px 14px", borderBottom: "1px solid var(--border-soft)" }}>
        <SectionLabel>Workbench</SectionLabel>

        <div style={{
          display: "flex", gap: 6, overflowX: "auto", flex: 1, alignItems: "center",
          paddingLeft: 6, paddingRight: 6,
        }}>
          {allExamples.map((ex) => {
            const sel = ex.kind === current.kind && ex.index === current.index;
            return (
              <button key={`${ex.kind}-${ex.index}`}
                onClick={() => onSelectExample({ kind: ex.kind, index: ex.index })}
                style={{
                  flexShrink: 0,
                  display: "flex", alignItems: "center", gap: 7, padding: "4px 9px 4px 5px",
                  borderRadius: 6, border: `1px solid ${sel ? "var(--accent)" : "var(--border)"}`,
                  background: sel ? "var(--accent-bg)" : "var(--bg-input)",
                  color: sel ? "var(--accent)" : "var(--text-muted)",
                  cursor: "pointer", fontSize: 11.5, fontWeight: 500,
                }}>
                <MiniGrid data={ex.pair.input} maxSize={24} />
                <span style={{ fontFamily: "var(--mono)", textTransform: "uppercase", letterSpacing: ".04em" }}>
                  {ex.label}
                </span>
              </button>
            );
          })}
        </div>

        <div style={{ display: "flex", alignItems: "center", gap: 8 }}>
          {actualOutput && (
            <Pill tone={mismatch?.ok ? "success" : "danger"}>
              {mismatch?.ok ? "match" : `${mismatch?.bad} px diff`}
            </Pill>
          )}
          <Button size="sm" variant="ghost" icon="refresh" onClick={onClearOutput} disabled={!actualOutput}>Clear</Button>
          <Button size="sm" variant={running ? "secondary" : "primary"} icon={running ? "pause" : "play"} onClick={onRun}>
            {running ? "Stop" : "Run"}
          </Button>
        </div>
      </div>

      <div style={{
        display: "flex", alignItems: "center", justifyContent: "center", gap: 24,
        padding: "14px 14px 10px",
      }}>
        <GridPanel title="Input" subtitle={current.label} data={pair.input} />
        <Arrow />
        <GridPanel title="Expected" subtitle="ground truth" data={pair.output} />
        <Equals ok={mismatch?.ok} pending={!actualOutput} />
        <GridPanel
          title="Actual"
          subtitle={running ? "running…" : actualOutput ? (mismatch?.ok ? "match" : "diff") : "not run"}
          data={actualOutput || pair.output}
          ghost={!actualOutput && !running}
          running={running}
          progress={runProgress}
          diffMask={diffMask}
        />
      </div>

      <div style={{
        display: "flex", alignItems: "center", justifyContent: "center", gap: 10,
        padding: "6px 14px 10px", flexWrap: "wrap",
      }}>
        <span style={{
          fontSize: 10, fontWeight: 600, letterSpacing: ".1em", textTransform: "uppercase",
          color: "var(--text-dim)", marginRight: 4,
        }}>palette</span>
        {ARC_COLORS.map((c, i) => (
          <span key={i} style={{
            display: "inline-flex", alignItems: "center", gap: 5,
            fontFamily: "var(--mono)", fontSize: 11, color: "var(--text-muted)",
          }}>
            <span style={{
              width: 14, height: 14, background: c, borderRadius: 3,
              border: "1px solid oklch(0.30 0.005 250)",
              boxShadow: "inset 0 0 0 1px oklch(0 0 0 / 0.2)",
            }} />
            {i} <span style={{ color: "var(--text-dim)" }}>{ARC_COLOR_NAMES[i]}</span>
          </span>
        ))}
      </div>
    </div>
  );
};

const LogsView = ({ logs }) => (
  <div style={{ height: "100%", overflowY: "auto", padding: "6px 14px", fontFamily: "var(--mono)", fontSize: 11.5 }}>
    {logs.length === 0 && (
      <div style={{ padding: "24px 0", textAlign: "center", color: "var(--text-dim)" }}>No logs yet.</div>
    )}
    {logs.map((l, i) => (
      <div key={i} style={{
        display: "grid", gridTemplateColumns: "68px 64px 90px 1fr", gap: 10,
        padding: "3px 0", color: "var(--text-muted)",
      }}>
        <span style={{ color: "var(--text-dim)" }}>{l.t}</span>
        <span style={{
          fontWeight: 600,
          color: l.level === "error" ? "var(--danger)"
            : l.level === "warn" ? "var(--warning)"
              : l.level === "ok" ? "var(--success)" : "var(--accent)",
        }}>{l.level.toUpperCase()}</span>
        <span style={{ color: "var(--text-dim)" }}>{l.src}</span>
        <span style={{ color: "var(--text)" }}>{l.msg}</span>
      </div>
    ))}
  </div>
);

const ValidationView = ({ validation, taskId, graph }) => {
  if (!validation || validation.state === "idle") {
    return (
      <div style={{ padding: 14, color: "var(--text-dim)", fontSize: 12.5 }}>
        Validation has not been run yet. Press Run or Export to validate this graph.
      </div>
    );
  }
  if (validation.state === "loading") {
    return (
      <div style={{ padding: 14, color: "var(--text-muted)", fontSize: 12.5 }}>
        Running validation against task{taskId}…
      </div>
    );
  }
  if (validation.state === "passed") {
    return (
      <div style={{ padding: 14, fontSize: 12.5 }}>
        <Pill tone="success" style={{ marginBottom: 8 }}>PASSED · task{validation.taskId}</Pill>
        <div style={{ fontFamily: "var(--mono)", color: "var(--text-muted)" }}>artifact: {validation.artifact}</div>
        <div style={{ marginTop: 8 }}>
          {graph.nodes.length} nodes · {graph.edges.length} edges
        </div>
      </div>
    );
  }
  return (
    <div style={{ padding: 14, fontSize: 12.5 }}>
      <Pill tone="danger" style={{ marginBottom: 8 }}>FAILED · task{validation.taskId}</Pill>
      <pre style={{
        margin: 0, padding: 10, fontSize: 11.5, fontFamily: "var(--mono)",
        background: "var(--bg-input)", border: "1px solid var(--border)",
        borderRadius: 5, color: "var(--danger)",
        whiteSpace: "pre-wrap", overflow: "auto", maxHeight: 200,
      }}>{validation.reason}</pre>
    </div>
  );
};

const SmallGrid = ({ data, ghost, label }) => {
  if (!data?.length) {
    return (
      <div style={{ display: "flex", flexDirection: "column", gap: 4, opacity: ghost ? 0.4 : 1 }}>
        <div style={{ fontSize: 10, color: "var(--text-dim)", textTransform: "uppercase", letterSpacing: ".08em" }}>{label}</div>
        <div style={{ fontSize: 11, color: "var(--text-dim)", padding: "12px 14px", border: "1px dashed var(--border)", borderRadius: 5 }}>—</div>
      </div>
    );
  }
  const h = data.length, w = data[0]?.length || 1;
  const cell = Math.max(6, Math.min(14, Math.floor(110 / Math.max(h, w, 1))));
  return (
    <div style={{ display: "flex", flexDirection: "column", gap: 4, alignItems: "flex-start", opacity: ghost ? 0.6 : 1 }}>
      <div style={{ fontSize: 10, color: "var(--text-dim)", textTransform: "uppercase", letterSpacing: ".08em" }}>{label} <span style={{ fontFamily: "var(--mono)", color: "var(--text-dim)" }}>· {h}×{w}</span></div>
      <div style={{
        display: "grid",
        gridTemplateColumns: `repeat(${w}, ${cell}px)`,
        gridTemplateRows: `repeat(${h}, ${cell}px)`,
        gap: 1, background: "oklch(0.10 0 0)", padding: 2, borderRadius: 4,
      }}>
        {data.flatMap((row, r) => row.map((v, c) => (
          <div key={`${r}-${c}`} style={{ background: ARC_COLORS[Number(v)] || ARC_COLORS[0] }} />
        )))}
      </div>
    </div>
  );
};

const NodeOutputView = ({ graph, inputGrid, expectedOutput, actualOutput, nodeOutputs, running }) => {
  const n = graph.nodes.find((x) => x.id === graph.selectedNodeId);
  const trace = n && nodeOutputs ? nodeOutputs[n.id] : null;
  const traceLabel = trace
    ? `trace · ${n.id}${trace.isApprox ? " (argmax)" : ""}`
    : n
      ? `selected · ${n.id}`
      : "selected";
  const traceGrid = trace?.grid || (n?.type === "Input" ? inputGrid : n?.type === "Output" ? (actualOutput || expectedOutput) : null);

  return (
    <div style={{ padding: 14, fontSize: 12.5, display: "flex", flexDirection: "column", gap: 12 }}>
      <SectionLabel>Run preview · current example</SectionLabel>
      <div style={{ display: "flex", gap: 18, flexWrap: "wrap" }}>
        <SmallGrid data={inputGrid} label="input" />
        <SmallGrid data={expectedOutput} label="expected" />
        <SmallGrid data={actualOutput} label={running ? "actual · running" : "actual"} ghost={!actualOutput && !running} />
        {n && (
          <SmallGrid
            data={traceGrid}
            label={traceLabel}
            ghost={!traceGrid && !running}
          />
        )}
      </div>
      {n ? (
        <div style={{ paddingTop: 8, borderTop: "1px solid var(--border-soft)" }}>
          <SectionLabel>Selected node · {n.id}</SectionLabel>
          <div style={{ marginTop: 6, fontFamily: "var(--mono)", color: "var(--text-muted)", fontSize: 11.5 }}>
            type: <span style={{ color: "var(--text)" }}>{n.type}</span><br />
            attrs: <span style={{ color: "var(--text)" }}>{JSON.stringify(n.attrs || {})}</span>
          </div>
          {trace ? (
            <div style={{ marginTop: 6, fontSize: 11.5, color: "var(--text-dim)", display: "flex", gap: 10, flexWrap: "wrap" }}>
              <span>shape: <span style={{ color: "var(--text)", fontFamily: "var(--mono)" }}>[{trace.shape.join(", ")}]</span></span>
              <span>dtype: <span style={{ color: "var(--text)", fontFamily: "var(--mono)" }}>{trace.dtype}</span></span>
              <span>min: <span style={{ color: "var(--text)", fontFamily: "var(--mono)" }}>{trace.stats?.min}</span></span>
              <span>max: <span style={{ color: "var(--text)", fontFamily: "var(--mono)" }}>{trace.stats?.max}</span></span>
              <span>nnz: <span style={{ color: "var(--text)", fontFamily: "var(--mono)" }}>{trace.stats?.nnz}</span></span>
              {trace.truncated && <span style={{ color: "var(--warning)" }}>· truncated 30×30</span>}
            </div>
          ) : (
            <div style={{ marginTop: 6, fontSize: 11.5, color: "var(--text-dim)" }}>
              {running ? "Trace populating…" : "Press Run to capture per-node tensors."}
            </div>
          )}
        </div>
      ) : (
        <div style={{ paddingTop: 6, fontSize: 11.5, color: "var(--text-dim)" }}>
          Select a node on the canvas to inspect its tensor.
        </div>
      )}
    </div>
  );
};

export const BottomPanel = ({
  logs, validation, taskId, graph, height, onHeight, collapsed, onToggle, tab, onTab,
  inputGrid = null, expectedOutput = null, actualOutput = null, nodeOutputs = null, running = false,
}) => {
  const tabs = [
    { id: "logs", label: "Logs", count: logs.length, icon: "terminal" },
    { id: "validation", label: "Validation", icon: "target" },
    { id: "output", label: "Node Output", icon: "eye" },
  ];

  const startDrag = (e) => {
    const startY = e.clientY;
    const startH = height;
    const move = (ev) => {
      const newH = Math.max(80, Math.min(window.innerHeight - 300, startH + (startY - ev.clientY)));
      onHeight(newH);
    };
    const up = () => {
      window.removeEventListener("pointermove", move);
      window.removeEventListener("pointerup", up);
    };
    window.addEventListener("pointermove", move);
    window.addEventListener("pointerup", up);
  };

  return (
    <div style={{
      flexShrink: 0, height: collapsed ? 0 : height,
      background: "var(--bg-surface)",
      borderTop: collapsed ? "none" : "1px solid var(--border)",
      display: "flex", flexDirection: "column",
      transition: "height 180ms cubic-bezier(.4,0,.2,1)",
      overflow: "hidden",
    }}>
      {!collapsed && (
        <>
          <div onPointerDown={startDrag} style={{ height: 4, cursor: "ns-resize", flexShrink: 0, marginTop: -2 }} />
          <div style={{ height: 32, display: "flex", alignItems: "center", borderBottom: "1px solid var(--border)" }}>
            {tabs.map((t) => (
              <button key={t.id} onClick={() => onTab(t.id)} style={{
                display: "inline-flex", alignItems: "center", gap: 6,
                height: "100%", padding: "0 12px", background: "transparent", border: "none",
                borderBottom: tab === t.id ? "2px solid var(--accent)" : "2px solid transparent",
                color: tab === t.id ? "var(--text)" : "var(--text-muted)",
                fontSize: 12, fontWeight: 500, cursor: "pointer", marginBottom: -1,
              }}>
                <Icon name={t.icon} size={12} />
                {t.label}
                {t.count != null && (
                  <span style={{
                    fontFamily: "var(--mono)", fontSize: 10, padding: "1px 5px", borderRadius: 8,
                    background: "var(--bg-input)", color: "var(--text-muted)",
                  }}>{t.count}</span>
                )}
              </button>
            ))}
            <div style={{ flex: 1 }} />
            <button onClick={onToggle} style={{
              height: "100%", padding: "0 12px", background: "transparent", border: "none",
              color: "var(--text-muted)", cursor: "pointer", display: "inline-flex", alignItems: "center", gap: 4,
              fontSize: 11.5,
            }}>
              <Icon name="chevronDown" size={12} />
              Hide
            </button>
          </div>
          <div style={{ flex: 1, overflow: "hidden" }}>
            {tab === "logs" && <LogsView logs={logs} />}
            {tab === "validation" && <ValidationView validation={validation} taskId={taskId} graph={graph} />}
            {tab === "output" && (
              <NodeOutputView
                graph={graph}
                inputGrid={inputGrid}
                expectedOutput={expectedOutput}
                actualOutput={actualOutput}
                nodeOutputs={nodeOutputs}
                running={running}
              />
            )}
          </div>
        </>
      )}
    </div>
  );
};

const statusBtn = (active, narrow = false) => ({
  display: "inline-flex", alignItems: "center", gap: 5,
  background: active ? "var(--bg-elevated)" : "transparent",
  border: "1px solid", borderColor: active ? "var(--border-strong)" : "var(--border)",
  borderRadius: 4, padding: narrow ? "2px 6px" : "2px 8px",
  color: active ? "var(--accent)" : "var(--text-muted)", cursor: "pointer", fontSize: 11,
  fontFamily: "var(--mono)",
});

const fmtBytesShort = (n) => {
  if (!Number.isFinite(n)) return "—";
  if (n < 1024) return `${n}B`;
  if (n < 1024 * 1024) return `${(n / 1024).toFixed(1)}K`;
  return `${(n / 1024 / 1024).toFixed(2)}M`;
};
const fmtNumShort = (n) => {
  if (!Number.isFinite(n)) return "—";
  if (n < 1000) return String(n);
  if (n < 1_000_000) return `${(n / 1000).toFixed(n < 10000 ? 1 : 0)}K`;
  return `${(n / 1_000_000).toFixed(2)}M`;
};

export const StatusBar = ({
  taskId, graph, bottomCollapsed, onToggleBottom,
  inspectorOpen, onToggleInspector, selectedNode, fullscreen, onToggleFullscreen,
  suggestionCount, onOpenSimplifier, status,
  efficiency, efficiencyError, efficiencyRefreshing, bestByTask,
}) => {
  const totalScore = Object.values(bestByTask || {}).reduce((s, e) => s + (e?.score || 0), 0);
  const solvedCount = Object.values(bestByTask || {}).filter((e) => e && e.cost && e.cost > 0).length;
  const eff = efficiency;
  const scoreTone =
    !eff ? "var(--text-dim)" :
    !eff.validBytes ? "var(--danger)" :
    eff.score >= 15 ? "var(--success)" :
    eff.score >= 10 ? "var(--accent)" :
    "var(--warning)";
  return (
  <div style={{
    height: 28, flexShrink: 0,
    background: "var(--bg-surface)",
    borderTop: "1px solid var(--border)",
    display: "flex", alignItems: "center", gap: 14, padding: "0 12px",
    fontSize: 11.5, fontFamily: "var(--mono)", color: "var(--text-muted)",
  }}>
    <span style={{ display: "inline-flex", alignItems: "center", gap: 5 }}>
      <Icon name="branch" size={11} /> task{taskId}
    </span>
    <span style={{ color: "var(--text-dim)" }}>{graph.nodes.length} nodes · {graph.edges.length} edges</span>
    {efficiencyError ? (
      <button onClick={onOpenSimplifier} style={{
        display: "inline-flex", alignItems: "center", gap: 5,
        background: "var(--danger-bg)", border: "1px solid var(--danger)", borderRadius: 4,
        padding: "1px 7px", color: "var(--danger)", fontFamily: "var(--mono)", cursor: "pointer",
        fontSize: 11,
      }} title={efficiencyError}>
        <Icon name="alert" size={10} /> compile fail
      </button>
    ) : eff ? (
      <button onClick={onOpenSimplifier} style={{
        display: "inline-flex", alignItems: "center", gap: 6,
        background: "var(--bg-input)", border: "1px solid var(--border)", borderRadius: 4,
        padding: "1px 8px", color: "var(--text)", fontFamily: "var(--mono)", cursor: "pointer",
        fontSize: 11,
      }} title={`Cost ${eff.cost} = ${eff.parameters} params + ${eff.bytes} bytes · score ${eff.score.toFixed(2)}`}>
        <Icon name="zap" size={10} style={{ color: scoreTone }} />
        <span style={{ color: scoreTone, fontWeight: 600 }}>{eff.score.toFixed(2)}</span>
        <span style={{ color: "var(--text-dim)" }}>· {fmtNumShort(eff.parameters)}p · {fmtBytesShort(eff.bytes)}</span>
        {efficiencyRefreshing && <span style={{ color: "var(--text-dim)" }}>·</span>}
      </button>
    ) : (
      efficiencyRefreshing && <span style={{ color: "var(--text-dim)" }}>compiling…</span>
    )}
    <span style={{ color: "var(--text-dim)" }} title={`Σ score across ${solvedCount} best-known graph${solvedCount === 1 ? "" : "s"} (max ${(25 * 400).toFixed(0)})`}>
      Σ {totalScore.toFixed(1)} / {(25 * 400).toFixed(0)}
    </span>
    {status && <span style={{ color: "var(--text-dim)", overflow: "hidden", textOverflow: "ellipsis", whiteSpace: "nowrap", maxWidth: 280 }}>{status}</span>}
    {suggestionCount > 0 && (
      <button onClick={onOpenSimplifier} style={{
        display: "inline-flex", alignItems: "center", gap: 5,
        background: "var(--warning-bg)", border: "1px solid var(--warning)",
        borderRadius: 4, padding: "2px 8px",
        color: "var(--warning)", cursor: "pointer", fontSize: 11,
        fontFamily: "var(--mono)",
      }} title="Open auto-simplifier">
        <Icon name="sparkle" size={11} /> {suggestionCount} suggestion{suggestionCount > 1 ? "s" : ""}
      </button>
    )}
    <div style={{ flex: 1 }} />
    {selectedNode && (
      <span style={{ color: "var(--text)" }}>
        {selectedNode.type} <span style={{ color: "var(--text-dim)" }}>{selectedNode.id}</span>
      </span>
    )}
    <button onClick={onToggleBottom} style={statusBtn(!bottomCollapsed)} title="Toggle bottom panel (⌘J)">
      <Icon name={bottomCollapsed ? "chevronUp" : "chevronDown"} size={11} /> Panel
    </button>
    <div style={{ width: 1, height: 14, background: "var(--border)", margin: "0 2px" }} />
    <button onClick={onToggleInspector} style={statusBtn(inspectorOpen)}>
      <Icon name="sliders" size={11} /> Inspector <Kbd>⌘I</Kbd>
    </button>
    <button onClick={onToggleFullscreen} style={statusBtn(fullscreen)} title="Fullscreen graph (⌘F · Esc)">
      <Icon name={fullscreen ? "fullscreenExit" : "fullscreenEnter"} size={11} /> {fullscreen ? "Exit FS" : "Fullscreen"}
    </button>
  </div>
  );
};

export const CommandPalette = ({ open, onClose, tasks, onSelectTask, onAddNode, onRun, onSave, onSaveAll, onImport, onExport, onCompile }) => {
  const [q, setQ] = useState("");
  useEffect(() => { if (open) setQ(""); }, [open]);

  const allCommands = useMemo(() => {
    const cmds = [
      { kind: "action", label: "Run on selected example", icon: "play", action: onRun, group: "Actions", kbd: "⌘R" },
      { kind: "action", label: "Save current task", icon: "save", action: onSave, group: "Actions", kbd: "⌘S" },
      { kind: "action", label: "Save all tasks in workspace", icon: "save", action: onSaveAll, group: "Actions", kbd: "⌘⇧S" },
      { kind: "action", label: "Compile current graph", icon: "zap", action: onCompile, group: "Actions" },
      { kind: "action", label: "Import ONNX…", icon: "upload", action: onImport, group: "Actions" },
      { kind: "action", label: "Export ONNX for current task", icon: "download", action: onExport, group: "Actions" },
    ];
    Object.keys(NODE_TYPES).forEach((t) => {
      cmds.push({ kind: "node", label: `Add ${t} node`, icon: "plus", action: () => onAddNode(t), group: "Nodes" });
    });
    tasks.forEach((t) => {
      cmds.push({ kind: "task", label: `Open task${t.id} · ${t.name}`, icon: "file", action: () => onSelectTask(t.id), group: "Tasks", status: t.status });
    });
    return cmds;
  }, [tasks, onAddNode, onSelectTask, onRun, onSave, onSaveAll, onImport, onExport, onCompile]);

  const filtered = useMemo(() => {
    if (!q) return allCommands.slice(0, 25);
    const ql = q.toLowerCase();
    return allCommands.filter((c) => c.label.toLowerCase().includes(ql)).slice(0, 40);
  }, [allCommands, q]);

  if (!open) return null;
  return (
    <div onClick={onClose} style={{
      position: "fixed", inset: 0, background: "oklch(0.10 0 0 / 0.55)",
      zIndex: 100, display: "flex", justifyContent: "center", paddingTop: "12vh",
    }}>
      <div onClick={(e) => e.stopPropagation()} style={{
        width: 560, maxHeight: "70vh",
        background: "var(--bg-surface)", border: "1px solid var(--border)", borderRadius: 10,
        boxShadow: "0 22px 60px -12px oklch(0.05 0 0 / 0.7)",
        display: "flex", flexDirection: "column",
      }}>
        <div style={{ padding: "12px 14px", display: "flex", alignItems: "center", gap: 10, borderBottom: "1px solid var(--border)" }}>
          <Icon name="search" size={15} />
          <input autoFocus value={q} onChange={(e) => setQ(e.target.value)}
            placeholder="Type a command, node, or task…"
            style={{ flex: 1, background: "transparent", border: "none", outline: "none", fontSize: 14, color: "var(--text)" }} />
          <Kbd>esc</Kbd>
        </div>
        <div style={{ overflowY: "auto", padding: 6, flex: 1 }}>
          {filtered.map((c, i) => (
            <button key={i} onClick={() => { c.action?.(); onClose(); }} style={{
              display: "flex", alignItems: "center", gap: 10, width: "100%",
              padding: "7px 10px", borderRadius: 5, background: "transparent",
              border: "none", color: "var(--text)", textAlign: "left", cursor: "pointer",
              fontSize: 13,
            }}
              onMouseEnter={(e) => e.currentTarget.style.background = "var(--bg-elevated)"}
              onMouseLeave={(e) => e.currentTarget.style.background = "transparent"}>
              <Icon name={c.icon} size={14} style={{ color: "var(--text-muted)" }} />
              <span style={{ flex: 1 }}>{c.label}</span>
              {c.status && <StatusDot status={c.status} size={6} />}
              {c.kbd && <Kbd>{c.kbd}</Kbd>}
              <span style={{ fontSize: 10, color: "var(--text-dim)", fontFamily: "var(--mono)", textTransform: "uppercase", letterSpacing: ".06em", minWidth: 56, textAlign: "right" }}>{c.group}</span>
            </button>
          ))}
          {!filtered.length && (
            <div style={{ padding: 24, textAlign: "center", color: "var(--text-dim)", fontSize: 13 }}>No matches</div>
          )}
        </div>
      </div>
    </div>
  );
};

const Modal = ({ open, onClose, title, children, footer, width = 480 }) => {
  if (!open) return null;
  return (
    <div onClick={onClose} style={{
      position: "fixed", inset: 0, background: "oklch(0.10 0 0 / 0.55)",
      zIndex: 100, display: "flex", justifyContent: "center", alignItems: "center",
    }}>
      <div onClick={(e) => e.stopPropagation()} style={{
        width, background: "var(--bg-surface)", border: "1px solid var(--border)", borderRadius: 10,
        boxShadow: "0 22px 60px -12px oklch(0.05 0 0 / 0.7)",
        display: "flex", flexDirection: "column", maxHeight: "86vh",
      }}>
        <div style={{ padding: "14px 16px", borderBottom: "1px solid var(--border)", display: "flex", alignItems: "center", justifyContent: "space-between" }}>
          <div style={{ fontSize: 15, fontWeight: 600 }}>{title}</div>
          <button onClick={onClose} style={{ background: "transparent", border: "none", cursor: "pointer", color: "var(--text-muted)" }}>
            <Icon name="x" size={16} />
          </button>
        </div>
        <div style={{ flex: 1, padding: 16, overflowY: "auto" }}>{children}</div>
        {footer && <div style={{ padding: "12px 16px", borderTop: "1px solid var(--border)", display: "flex", gap: 8, justifyContent: "flex-end" }}>{footer}</div>}
      </div>
    </div>
  );
};

export const ImportModal = ({ open, onClose, onPickFiles, importState }) => {
  const fileRef = useRef(null);
  return (
    <Modal open={open} onClose={onClose} title="Import ONNX" width={520}
      footer={<>
        <Button variant="ghost" onClick={onClose}>Cancel</Button>
        <Button variant="primary" icon="upload" onClick={() => fileRef.current?.click()}>Pick files</Button>
      </>}>
      <input
        ref={fileRef}
        type="file"
        accept=".onnx"
        multiple
        style={{ display: "none" }}
        onChange={(e) => {
          const files = Array.from(e.target.files || []);
          e.target.value = "";
          if (files.length) onPickFiles(files);
        }}
      />
      <div onClick={() => fileRef.current?.click()} style={{
        cursor: "pointer",
        border: "1.5px dashed var(--border-strong)", borderRadius: 8, padding: "32px 16px",
        textAlign: "center", color: "var(--text-muted)", background: "var(--bg-input)",
      }}>
        <Icon name="upload" size={28} style={{ color: "var(--accent)", marginBottom: 8 }} />
        <div style={{ fontSize: 14, fontWeight: 500, color: "var(--text)" }}>Drop ONNX files here</div>
        <div style={{ fontSize: 12, marginTop: 4 }}>or pick task001.onnx … taskNNN.onnx</div>
      </div>
      {importState?.state === "loading" && (
        <div style={{ marginTop: 12, color: "var(--text-muted)", fontSize: 12.5 }}>Uploading…</div>
      )}
      {importState?.state === "ok" && (
        <div style={{ marginTop: 12, fontSize: 12.5 }}>
          <Pill tone="success">imported</Pill>
          <div style={{ marginTop: 6, fontFamily: "var(--mono)", color: "var(--text-muted)" }}>
            saved: {importState.saved?.join(", ") || "0"} · rejected: {importState.rejected?.length || 0}
          </div>
        </div>
      )}
      {importState?.state === "failed" && (
        <div style={{ marginTop: 12, color: "var(--danger)", fontSize: 12.5 }}>Import failed: {importState.reason}</div>
      )}
    </Modal>
  );
};

export const ExportModal = ({ open, onClose, taskId, validation, onExport }) => {
  const ready = validation?.state !== "loading";
  return (
    <Modal open={open} onClose={onClose} title={`Export ONNX · task${taskId}`} width={520}
      footer={<>
        <Button variant="ghost" onClick={onClose}>Cancel</Button>
        <Button variant="primary" icon="download" onClick={onExport} disabled={!ready}>Export task{taskId}.onnx</Button>
      </>}>
      <div style={{ display: "flex", flexDirection: "column", gap: 14 }}>
        <Field label="Filename"><TextInput value={`task${taskId}.onnx`} readOnly /></Field>
        <div style={{
          padding: "10px 12px", background: "var(--bg-input)", borderRadius: 6,
          border: "1px solid var(--border)", fontSize: 12.5, color: "var(--text-muted)",
          display: "flex", flexDirection: "column", gap: 6,
        }}>
          <div style={{ color: "var(--text)" }}>The backend will:</div>
          <div>1. Compile this graph to ONNX</div>
          <div>2. Run validation against all training pairs</div>
          <div>3. If <span style={{ fontFamily: "var(--mono)" }}>HF_TOKEN</span> + <span style={{ fontFamily: "var(--mono)" }}>HF_REPO_ID</span> are set, upload to Hugging Face — otherwise download here as <span style={{ fontFamily: "var(--mono)" }}>task{taskId}.onnx</span>.</div>
        </div>
        {validation?.state === "loading" && (
          <div style={{
            padding: "10px 12px", background: "var(--warning-bg)", borderRadius: 6,
            border: "1px solid var(--warning)", color: "var(--warning)", fontSize: 12.5,
          }}>
            Compiling and validating…
          </div>
        )}
        {validation?.state === "failed" && (
          <div style={{
            padding: "10px 12px", background: "var(--danger-bg)", borderRadius: 6,
            border: "1px solid var(--danger)", color: "var(--danger)", fontSize: 12.5,
          }}>
            {validation.reason}
          </div>
        )}
        {validation?.state === "passed" && (
          <div style={{
            padding: "10px 12px", background: "var(--success-bg, var(--accent-bg))", borderRadius: 6,
            border: "1px solid var(--success)", color: "var(--success)", fontSize: 12.5,
          }}>
            Last export passed · artifact <span style={{ fontFamily: "var(--mono)" }}>{validation.artifact}</span>
          </div>
        )}
      </div>
    </Modal>
  );
};

export const WorkspacesModal = ({ open, mode, onClose, workspaces, currentId, onPick, onCreate, onRename, onDuplicate, onDelete }) => {
  const [newName, setNewName] = useState("");
  const [renameId, setRenameId] = useState(null);
  const [renameVal, setRenameVal] = useState("");
  useEffect(() => {
    if (open) {
      setNewName(mode === "new" ? "new-workspace" : "");
    }
  }, [open, mode]);

  return (
    <Modal open={open} onClose={onClose} title="Workspaces" width={560}
      footer={<><Button variant="ghost" onClick={onClose}>Close</Button></>}>
      <div style={{ display: "flex", flexDirection: "column", gap: 14 }}>
        <div>
          <SectionLabel>Create new workspace</SectionLabel>
          <div style={{ display: "flex", gap: 8, marginTop: 8 }}>
            <input value={newName} onChange={(e) => setNewName(e.target.value)}
              placeholder="workspace name…"
              style={{ ...inputStyle, flex: 1, fontFamily: "var(--font)" }} />
            <Button variant="primary" icon="plus" onClick={() => { if (newName.trim()) { onCreate(newName.trim()); setNewName(""); } }}>
              Create
            </Button>
          </div>
        </div>

        <div>
          <SectionLabel>All workspaces · {workspaces.length}</SectionLabel>
          <div style={{ marginTop: 8, display: "flex", flexDirection: "column", gap: 5 }}>
            {workspaces.map((w) => {
              const isRenaming = renameId === w.id;
              return (
                <div key={w.id} style={{
                  display: "flex", alignItems: "center", gap: 10,
                  padding: "10px 12px", borderRadius: 6,
                  background: w.id === currentId ? "var(--accent-bg)" : "var(--bg-input)",
                  border: `1px solid ${w.id === currentId ? "var(--accent-dim)" : "var(--border)"}`,
                }}>
                  <Icon name="folder" size={14} style={{ color: w.id === currentId ? "var(--accent)" : "var(--text-muted)" }} />
                  <div style={{ flex: 1, minWidth: 0 }}>
                    {isRenaming ? (
                      <input autoFocus value={renameVal} onChange={(e) => setRenameVal(e.target.value)}
                        onBlur={() => { if (renameVal.trim()) onRename(w.id, renameVal.trim()); setRenameId(null); }}
                        onKeyDown={(e) => {
                          if (e.key === "Enter") { if (renameVal.trim()) onRename(w.id, renameVal.trim()); setRenameId(null); }
                          if (e.key === "Escape") setRenameId(null);
                        }}
                        style={{ ...inputStyle, padding: "3px 6px", fontFamily: "var(--font)" }} />
                    ) : (
                      <div style={{
                        fontSize: 13, fontWeight: w.id === currentId ? 600 : 500,
                        color: w.id === currentId ? "var(--accent)" : "var(--text)",
                      }}>{w.name}</div>
                    )}
                    <div style={{ fontFamily: "var(--mono)", fontSize: 10.5, color: "var(--text-dim)" }}>
                      {w.tasks} tasks · saved {w.lastSaved}
                    </div>
                  </div>
                  {w.id !== currentId && <Button size="sm" variant="ghost" onClick={() => onPick(w.id)}>Switch</Button>}
                  <Button size="sm" variant="ghost" icon="sliders" onClick={() => { setRenameId(w.id); setRenameVal(w.name); }}>Rename</Button>
                  <Button size="sm" variant="ghost" icon="folder" onClick={() => onDuplicate(w.id)}>Duplicate</Button>
                  {workspaces.length > 1 && w.id !== currentId && (
                    <Button size="sm" variant="danger" icon="trash" onClick={() => onDelete(w.id)}>Delete</Button>
                  )}
                </div>
              );
            })}
          </div>
        </div>

        <div style={{ fontSize: 11.5, color: "var(--text-dim)" }}>
          Workspaces persist task graphs to localStorage. <Kbd>⌘S</Kbd> saves current task, <Kbd>⌘⇧S</Kbd> saves all in workspace.
        </div>
      </div>
    </Modal>
  );
};

export const rightTabStyle = (active) => ({
  display: "inline-flex", alignItems: "center", gap: 6,
  padding: "10px 14px", background: "transparent", border: "none",
  borderBottom: active ? "2px solid var(--accent)" : "2px solid transparent",
  color: active ? "var(--text)" : "var(--text-muted)",
  fontSize: 12.5, fontWeight: active ? 600 : 500, cursor: "pointer", marginBottom: -1,
});
