import React, { useState, useEffect } from "react";
import { ARC_COLORS } from "./data.js";

const PATHS = {
  chevronLeft:  (<polyline points="10 4 5 9 10 14" />),
  chevronRight: (<polyline points="6 4 11 9 6 14" />),
  chevronDown:  (<polyline points="4 6 9 11 14 6" />),
  chevronUp:    (<polyline points="4 11 9 6 14 11" />),
  play:         (<polygon points="5 3 14 9 5 15" />),
  pause:        (<><rect x="4" y="3" width="3" height="12"/><rect x="11" y="3" width="3" height="12"/></>),
  check:        (<polyline points="3 9 7 13 15 4" />),
  x:            (<><line x1="4" y1="4" x2="14" y2="14"/><line x1="14" y1="4" x2="4" y2="14"/></>),
  plus:         (<><line x1="9" y1="3" x2="9" y2="15"/><line x1="3" y1="9" x2="15" y2="9"/></>),
  minus:        (<line x1="3" y1="9" x2="15" y2="9"/>),
  search:       (<><circle cx="8" cy="8" r="5"/><line x1="12" y1="12" x2="15" y2="15"/></>),
  save:         (<><path d="M3 3 H12 L15 6 V15 H3 Z"/><rect x="5" y="3" width="8" height="4"/><rect x="6" y="10" width="6" height="5"/></>),
  upload:       (<><path d="M3 12 V15 H15 V12"/><polyline points="6 6 9 3 12 6"/><line x1="9" y1="3" x2="9" y2="11"/></>),
  download:     (<><path d="M3 12 V15 H15 V12"/><polyline points="6 9 9 12 12 9"/><line x1="9" y1="3" x2="9" y2="12"/></>),
  trash:        (<><polyline points="3 5 15 5"/><path d="M5 5 V14 H13 V5"/><line x1="7" y1="8" x2="7" y2="12"/><line x1="11" y1="8" x2="11" y2="12"/></>),
  gear:         (<><circle cx="9" cy="9" r="2.5"/><path d="M9 1 V3 M9 15 V17 M1 9 H3 M15 9 H17 M3 3 L4.5 4.5 M13.5 13.5 L15 15 M3 15 L4.5 13.5 M13.5 4.5 L15 3"/></>),
  grid:         (<><rect x="3" y="3" width="5" height="5"/><rect x="10" y="3" width="5" height="5"/><rect x="3" y="10" width="5" height="5"/><rect x="10" y="10" width="5" height="5"/></>),
  list:         (<><line x1="4" y1="5" x2="14" y2="5"/><line x1="4" y1="9" x2="14" y2="9"/><line x1="4" y1="13" x2="14" y2="13"/></>),
  folder:       (<path d="M2 5 V14 H16 V6 H9 L7 4 H2 Z"/>),
  zap:          (<polygon points="10 2 4 11 8 11 7 16 13 7 9 7"/>),
  eye:          (<><path d="M1 9 C3 5 6 3 9 3 C12 3 15 5 17 9 C15 13 12 15 9 15 C6 15 3 13 1 9 Z"/><circle cx="9" cy="9" r="2.5"/></>),
  fullscreenEnter: (<><polyline points="7 3 3 3 3 7"/><polyline points="11 3 15 3 15 7"/><polyline points="15 11 15 15 11 15"/><polyline points="3 11 3 15 7 15"/></>),
  fullscreenExit:  (<><polyline points="3 7 7 7 7 3"/><polyline points="11 3 11 7 15 7"/><polyline points="15 11 11 11 11 15"/><polyline points="3 11 7 11 7 15"/></>),
  fit:          (<><polyline points="5 3 3 3 3 5"/><polyline points="13 3 15 3 15 5"/><polyline points="15 13 15 15 13 15"/><polyline points="3 13 3 15 5 15"/><circle cx="9" cy="9" r="1.6" fill="currentColor" stroke="none"/></>),
  file:         (<><path d="M4 2 H11 L14 5 V16 H4 Z"/><polyline points="11 2 11 5 14 5"/></>),
  sparkle:      (<path d="M9 2 L10 7 L15 8 L10 9 L9 14 L8 9 L3 8 L8 7 Z"/>),
  refresh:      (<><polyline points="3 4 3 8 7 8"/><path d="M3 8 a6 6 0 0 1 10 -3"/><polyline points="15 14 15 10 11 10"/><path d="M15 10 a6 6 0 0 1 -10 3"/></>),
  target:       (<><circle cx="9" cy="9" r="6"/><circle cx="9" cy="9" r="3"/><circle cx="9" cy="9" r="1" fill="currentColor"/></>),
  sliders:      (<><line x1="3" y1="6" x2="15" y2="6"/><line x1="3" y1="12" x2="15" y2="12"/><circle cx="7" cy="6" r="1.5" fill="var(--bg-elevated)"/><circle cx="11" cy="12" r="1.5" fill="var(--bg-elevated)"/></>),
  arrowRight:   (<><line x1="3" y1="9" x2="15" y2="9"/><polyline points="11 5 15 9 11 13"/></>),
  branch:       (<><circle cx="5" cy="4" r="1.5"/><circle cx="5" cy="14" r="1.5"/><circle cx="13" cy="14" r="1.5"/><line x1="5" y1="5.5" x2="5" y2="12.5"/><path d="M5 8 a4 4 0 0 0 4 4 H11.5"/></>),
  terminal:     (<><polyline points="3 5 7 9 3 13"/><line x1="9" y1="13" x2="15" y2="13"/></>),
  dot:          (<circle cx="9" cy="9" r="2.5" fill="currentColor" stroke="none" />),
};

export const Icon = ({ name, size = 14, strokeWidth = 1.7, style = {} }) => (
  <svg width={size} height={size} viewBox="0 0 18 18"
    fill="none" stroke="currentColor" strokeWidth={strokeWidth}
    strokeLinecap="round" strokeLinejoin="round"
    style={{ flexShrink: 0, ...style }}>
    {PATHS[name] || null}
  </svg>
);

export const Button = ({ variant = "ghost", size = "md", icon, children, onClick, active, style = {}, disabled, title, ...rest }) => {
  const base = {
    display: "inline-flex", alignItems: "center", gap: 6,
    padding: size === "sm" ? "3px 8px" : size === "lg" ? "8px 14px" : "5px 10px",
    height: size === "sm" ? 24 : size === "lg" ? 34 : 28,
    fontSize: size === "sm" ? 12 : 13,
    fontWeight: 500,
    borderRadius: 6,
    border: "1px solid transparent",
    cursor: disabled ? "not-allowed" : "pointer",
    whiteSpace: "nowrap",
    transition: "background 80ms, border-color 80ms, color 80ms",
    userSelect: "none",
    opacity: disabled ? 0.55 : 1,
  };
  const variants = {
    primary:     { background: "var(--accent)",       color: "oklch(0.18 0.02 200)", borderColor: "transparent" },
    secondary:   { background: "var(--bg-elevated)",  color: "var(--text)",          borderColor: "var(--border)" },
    ghost:       { background: active ? "var(--bg-elevated)" : "transparent", color: active ? "var(--text)" : "var(--text-muted)", borderColor: active ? "var(--border)" : "transparent" },
    danger:      { background: "transparent",         color: "var(--danger)",        borderColor: "var(--border)" },
    dangerSolid: { background: "var(--danger-bg)",    color: "var(--danger)",        borderColor: "var(--danger)" },
  };
  return (
    <button onClick={onClick} disabled={disabled} title={title}
      style={{ ...base, ...variants[variant], ...style }} {...rest}>
      {icon && <Icon name={icon} size={size === "sm" ? 12 : 14} />}
      {children}
    </button>
  );
};

export const Pill = ({ children, tone = "neutral", style = {} }) => {
  const tones = {
    neutral: { bg: "var(--bg-elevated)", fg: "var(--text-muted)", bd: "var(--border)" },
    accent:  { bg: "var(--accent-bg)",   fg: "var(--accent)",     bd: "transparent" },
    success: { bg: "var(--success-bg)",  fg: "var(--success)",    bd: "transparent" },
    warning: { bg: "var(--warning-bg)",  fg: "var(--warning)",    bd: "transparent" },
    danger:  { bg: "var(--danger-bg)",   fg: "var(--danger)",     bd: "transparent" },
  }[tone];
  return (
    <span style={{
      display: "inline-flex", alignItems: "center", gap: 4,
      padding: "1px 7px", borderRadius: 999,
      background: tones.bg, color: tones.fg, border: `1px solid ${tones.bd}`,
      fontSize: 11, fontWeight: 500, lineHeight: 1.6, fontFamily: "var(--mono)",
      letterSpacing: ".02em", ...style,
    }}>
      {children}
    </span>
  );
};

export const StatusDot = ({ status, size = 8 }) => {
  const colors = {
    passing: "var(--success)",
    failing: "var(--danger)",
    editing: "var(--accent)",
    untested: "oklch(0.45 0.01 250)",
  };
  return (
    <span style={{
      display: "inline-block", width: size, height: size, borderRadius: "50%",
      background: colors[status] || "var(--text-dim)",
      boxShadow: status === "editing" ? "0 0 0 3px oklch(0.78 0.12 200 / 0.18)" : "none",
    }} />
  );
};

export const SectionLabel = ({ children, action, style = {} }) => (
  <div style={{
    display: "flex", alignItems: "center", justifyContent: "space-between",
    padding: "0 2px",
    fontSize: 11, fontWeight: 600, letterSpacing: ".08em",
    textTransform: "uppercase", color: "var(--text-dim)", ...style,
  }}>
    <span>{children}</span>
    {action}
  </div>
);

export const Kbd = ({ children }) => (
  <kbd style={{
    fontFamily: "var(--mono)", fontSize: 10.5, padding: "1px 5px",
    background: "var(--bg-input)", border: "1px solid var(--border)",
    borderRadius: 3, color: "var(--text-muted)", lineHeight: 1.4,
  }}>{children}</kbd>
);

export const ArcGrid = ({ data, cell = 12, gap = 1, palette = ARC_COLORS, style = {} }) => {
  if (!data || !data.length) return null;
  const h = data.length, w = data[0].length;
  return (
    <div style={{
      display: "grid",
      gridTemplateColumns: `repeat(${w}, ${cell}px)`,
      gridTemplateRows: `repeat(${h}, ${cell}px)`,
      gap, padding: 0,
      background: "oklch(0.10 0.005 250)",
      border: "1px solid var(--border)",
      borderRadius: 3,
      width: "fit-content",
      ...style,
    }}>
      {data.flatMap((row, r) => row.map((v, c) => (
        <div key={`${r}-${c}`} style={{ background: palette[v] || palette[0] }} />
      )))}
    </div>
  );
};

export const MiniGrid = ({ data, maxSize = 28 }) => {
  if (!data || !data.length) return null;
  const h = data.length, w = data[0].length;
  const cell = Math.max(1, Math.floor(maxSize / Math.max(h, w)));
  return (
    <div style={{
      display: "grid",
      gridTemplateColumns: `repeat(${w}, ${cell}px)`,
      gridTemplateRows: `repeat(${h}, ${cell}px)`,
      gap: 0, padding: 1, background: "oklch(0.10 0 0)", borderRadius: 2,
    }}>
      {data.flatMap((row, r) => row.map((v, c) => (
        <div key={`${r}-${c}`} style={{ background: ARC_COLORS[v] || ARC_COLORS[0] }} />
      )))}
    </div>
  );
};

export const inputStyle = {
  background: "var(--bg-input)", border: "1px solid var(--border)",
  borderRadius: 5, padding: "6px 8px", color: "var(--text)",
  fontSize: 12.5, fontFamily: "var(--mono)", outline: "none",
  width: "100%",
};

export const Field = ({ label, hint, children }) => (
  <label style={{ display: "flex", flexDirection: "column", gap: 4 }}>
    <div style={{ display: "flex", justifyContent: "space-between", alignItems: "baseline" }}>
      <span style={{
        fontSize: 10.5, fontWeight: 600, letterSpacing: ".06em",
        textTransform: "uppercase", color: "var(--text-dim)",
      }}>{label}</span>
      {hint && <span style={{ fontSize: 10, color: "var(--text-dim)", fontFamily: "var(--mono)" }}>{hint}</span>}
    </div>
    {children}
  </label>
);

export const TextInput = ({ value, onChange, ...rest }) => (
  <input value={value ?? ""} onChange={(e) => onChange?.(e.target.value)} style={inputStyle} {...rest} />
);

export const Select = ({ value, options, onChange }) => (
  <select value={value} onChange={(e) => onChange?.(e.target.value)} style={{
    ...inputStyle, appearance: "none",
    backgroundImage: `url("data:image/svg+xml;utf8,<svg xmlns='http://www.w3.org/2000/svg' width='14' height='14' viewBox='0 0 18 18' fill='none' stroke='%23888' stroke-width='1.6'><polyline points='4 7 9 12 14 7'/></svg>")`,
    backgroundRepeat: "no-repeat", backgroundPosition: "right 6px center",
    paddingRight: 24,
  }}>
    {options.map((o) => <option key={o} value={o}>{o}</option>)}
  </select>
);

export const Toggle = ({ value, onChange, label }) => (
  <button type="button" onClick={() => onChange?.(!value)} style={{
    display: "flex", alignItems: "center", justifyContent: "space-between", gap: 8,
    width: "100%", padding: "5px 8px", borderRadius: 5,
    background: "var(--bg-input)", border: "1px solid var(--border)",
    color: "var(--text)", fontSize: 12.5, cursor: "pointer", textAlign: "left",
    fontFamily: "var(--mono)",
  }}>
    <span>{label || (value ? "true" : "false")}</span>
    <div style={{
      width: 26, height: 14, borderRadius: 8, padding: 1.5,
      background: value ? "var(--accent-dim)" : "oklch(0.30 0.01 250)",
      transition: "background 120ms",
    }}>
      <div style={{
        width: 11, height: 11, borderRadius: "50%",
        background: value ? "var(--accent)" : "var(--text-dim)",
        transform: value ? "translateX(12px)" : "translateX(0)",
        transition: "transform 120ms",
      }} />
    </div>
  </button>
);

export const NumberStepper = ({ value, onChange, step = 1, min, max }) => {
  const clamp = (v) => {
    if (min != null && v < min) return min;
    if (max != null && v > max) return max;
    return v;
  };
  const stepBtnStyle = {
    width: 24, background: "transparent", border: "none", color: "var(--text-muted)",
    cursor: "pointer", fontSize: 14, lineHeight: 1, fontFamily: "var(--mono)",
  };
  return (
    <div style={{
      display: "flex", alignItems: "stretch", gap: 0,
      background: "var(--bg-input)", border: "1px solid var(--border)", borderRadius: 5,
      overflow: "hidden",
    }}>
      <button type="button" onClick={() => onChange?.(clamp(Number(value) - step))} style={stepBtnStyle}>−</button>
      <input
        type="number" value={value} step={step}
        onChange={(e) => onChange?.(clamp(Number(e.target.value) || 0))}
        style={{
          flex: 1, background: "transparent", border: "none", outline: "none",
          color: "var(--text)", fontSize: 12.5, fontFamily: "var(--mono)",
          textAlign: "center", padding: "4px 0",
        }}
      />
      <button type="button" onClick={() => onChange?.(clamp(Number(value) + step))} style={stepBtnStyle}>+</button>
    </div>
  );
};

export const ShapeEditor = ({ value = [], onChange }) => {
  const arr = Array.isArray(value) ? value : [];
  const setDim = (i, v) => {
    const next = arr.slice();
    next[i] = Number(v) || 0;
    onChange?.(next);
  };
  const miniBtnStyle = {
    width: 18, height: 18, padding: 0,
    display: "inline-flex", alignItems: "center", justifyContent: "center",
    background: "oklch(0.16 0.008 250)", border: "1px solid var(--border)", borderRadius: 3,
    color: "var(--text-muted)", cursor: "pointer",
  };
  return (
    <div style={{
      display: "flex", alignItems: "center", gap: 4, flexWrap: "wrap",
      padding: "4px 6px", background: "var(--bg-input)", border: "1px solid var(--border)", borderRadius: 5,
    }}>
      <span style={{ color: "var(--text-dim)", fontFamily: "var(--mono)", fontSize: 13 }}>[</span>
      {arr.map((v, i) => (
        <React.Fragment key={i}>
          <input value={v}
            onChange={(e) => setDim(i, e.target.value)}
            style={{
              width: 38, background: "oklch(0.16 0.008 250)",
              border: "1px solid var(--border)", borderRadius: 3,
              color: "var(--text)", fontSize: 12, fontFamily: "var(--mono)",
              textAlign: "center", padding: "2px 0", outline: "none",
            }}
          />
          {i < arr.length - 1 && <span style={{ color: "var(--text-dim)", fontFamily: "var(--mono)" }}>,</span>}
        </React.Fragment>
      ))}
      <span style={{ color: "var(--text-dim)", fontFamily: "var(--mono)", fontSize: 13 }}>]</span>
      <div style={{ flex: 1 }} />
      <button type="button" onClick={() => onChange?.([...arr, 1])} style={miniBtnStyle} title="Add dimension">
        <Icon name="plus" size={10} />
      </button>
      {arr.length > 0 && (
        <button type="button" onClick={() => onChange?.(arr.slice(0, -1))} style={miniBtnStyle} title="Remove dimension">
          <Icon name="minus" size={10} />
        </button>
      )}
    </div>
  );
};

export const TensorEditor = ({ value, onChange }) => {
  const display = typeof value === "string" ? value : JSON.stringify(value, null, 2);
  const [local, setLocal] = useState(display);
  const [err, setErr] = useState(null);
  useEffect(() => setLocal(display), [display]);
  return (
    <div>
      <textarea
        value={local}
        rows={4}
        onChange={(e) => {
          const txt = e.target.value;
          setLocal(txt);
          try {
            const parsed = JSON.parse(txt);
            onChange?.(parsed);
            setErr(null);
          } catch (ex) {
            setErr(ex.message);
          }
        }}
        style={{
          ...inputStyle, fontFamily: "var(--mono)", padding: "6px 8px",
          resize: "vertical", minHeight: 60, whiteSpace: "pre",
          borderColor: err ? "var(--danger)" : "var(--border)",
        }} />
      {err && <div style={{ marginTop: 4, fontSize: 10.5, color: "var(--danger)", fontFamily: "var(--mono)" }}>parse error: {err}</div>}
    </div>
  );
};
