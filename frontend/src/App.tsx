import { useMemo, useRef, useState } from "react";

type JobResult = {
  job?: { state?: string; case_id?: string };
  output?: {
    inference?: {
      summary?: string;
      findings?: string[];
      confidence?: number;
    };
    qc_status?: string;
    qc_issues?: { format?: string[]; completeness?: string[]; safety?: string[] } | string[];
    qc_issues_flat?: string[];
    rag?: {
      query?: string | null;
      hits?: Array<{ title?: string; score?: number; source?: string }>;
      context_used?: string | null;
    };
    observability?: {
      durations_ms?: { rag?: number; inference?: number; qc?: number; total?: number };
      inference_runtime?: {
        run_mode?: string;
        model_source?: string;
        generated_token_count?: number;
        used_fallback?: boolean;
      };
    };
  };
};

const DEFAULT_API_BASE = import.meta.env.VITE_API_BASE ?? "http://127.0.0.1:8000/api/v1";
const DEFAULT_API_KEY = import.meta.env.VITE_API_KEY ?? "dev-local-key";

export default function App() {
  const [apiBase, setApiBase] = useState(DEFAULT_API_BASE);
  const [apiKey, setApiKey] = useState(DEFAULT_API_KEY);
  const [patientId, setPatientId] = useState("p-ui-001");
  const [idempotency, setIdempotency] = useState("");
  const [notes, setNotes] = useState("");
  const [images, setImages] = useState<File[]>([]);
  const [status, setStatus] = useState<{ text: string; kind: "idle" | "polling" | "ok" | "err" }>({
    text: "Idle",
    kind: "idle",
  });
  const [logs, setLogs] = useState<string>("");
  const [result, setResult] = useState<JobResult | null>(null);
  const [isDragging, setIsDragging] = useState(false);
  const fileInputRef = useRef<HTMLInputElement | null>(null);

  const statusClass = useMemo(() => status.kind, [status.kind]);

  function appendLog(title: string, data: unknown) {
    const block = `\n=== ${title} ===\n${typeof data === "string" ? data : JSON.stringify(data, null, 2)}\n`;
    setLogs((prev) => prev + block);
  }

  function headers() {
    if (!apiKey.trim()) throw new Error("Please provide x-api-key first");
    return { "x-api-key": apiKey.trim() };
  }

  async function requestJson(url: string, options: RequestInit = {}) {
    const res = await fetch(url, options);
    const text = await res.text();
    const json = text ? safeJson(text) : null;
    if (!res.ok) throw new Error(`${res.status} ${res.statusText}: ${JSON.stringify(json)}`);
    return json as any;
  }

  function safeJson(text: string) {
    try {
      return JSON.parse(text);
    } catch {
      return { raw: text };
    }
  }

  async function uploadArtifact(caseId: string, kind: string, file: File) {
    const form = new FormData();
    form.append("file", file);
    return requestJson(`${apiBase.replace(/\/$/, "")}/cases/${caseId}/artifacts?kind=${encodeURIComponent(kind)}`, {
      method: "POST",
      headers: headers(),
      body: form,
    });
  }

  async function pingInference() {
    try {
      setStatus({ text: "Checking inference service...", kind: "polling" });
      const data = await requestJson(`${apiBase.replace(/\/$/, "")}/inference/ping`, { headers: headers() });
      appendLog("inference_ping", data);
      setStatus({ text: `Inference service: ${data.reachable ? "reachable" : "unreachable"}`, kind: data.reachable ? "ok" : "err" });
    } catch (err) {
      const msg = String((err as Error).message ?? err);
      setStatus({ text: msg, kind: "err" });
      appendLog("error", msg);
    }
  }

  async function runWorkflow() {
    try {
      setStatus({ text: "Processing and polling...", kind: "polling" });
      setResult(null);
      const base = apiBase.replace(/\/$/, "");
      const h = headers();
      if (!notes.trim() && images.length === 0) throw new Error("Notes and Images cannot both be empty");

      const caseResp = await requestJson(`${base}/cases`, {
        method: "POST",
        headers: { ...h, "Content-Type": "application/json" },
        body: JSON.stringify({ patient_pseudo_id: patientId.trim() || "p-ui-001" }),
      });
      appendLog("case_created", caseResp);

      const caseId = caseResp.case_id as string;
      if (notes.trim()) {
        const notesFile = new File([notes.trim()], "notes.txt", { type: "text/plain;charset=utf-8" });
        appendLog("notes_uploaded", await uploadArtifact(caseId, "input_notes", notesFile));
      }

      for (const image of images) {
        appendLog(`image_uploaded:${image.name}`, await uploadArtifact(caseId, "input_image", image));
      }

      const idem = idempotency.trim() || `ui-${crypto.randomUUID()}`;
      const jobResp = await requestJson(`${base}/jobs`, {
        method: "POST",
        headers: { ...h, "Content-Type": "application/json" },
        body: JSON.stringify({ case_id: caseId, stage: "inference", idempotency_key: idem }),
      });
      appendLog("job_created", jobResp);
      const jobId = jobResp.job_id as string;

      for (let i = 0; i < 60; i += 1) {
        await new Promise((r) => setTimeout(r, 1000));
        const poll = (await requestJson(`${base}/workflow/jobs/${jobId}/result`, {
          method: "GET",
          headers: h,
        })) as JobResult;
        appendLog(`poll_${i + 1}`, { job_state: poll.job?.state, has_output: !!poll.output });
        if (poll.job?.state === "succeeded" || poll.job?.state === "failed") {
          appendLog("final_result", poll);
          setResult(poll);
          setStatus({ text: `Completed: ${poll.job?.state}`, kind: poll.job?.state === "succeeded" ? "ok" : "err" });
          return;
        }
      }
      throw new Error("Polling timeout: make sure worker is running");
    } catch (err) {
      const msg = String((err as Error).message ?? err);
      setStatus({ text: msg, kind: "err" });
      appendLog("error", msg);
    }
  }

  function clearAll() {
    setLogs("");
    setResult(null);
    setStatus({ text: "Idle", kind: "idle" });
  }

  function addImages(files: File[]) {
    if (!files.length) return;
    setImages((prev) => [...prev, ...files]);
  }

  function removeImage(index: number) {
    setImages((prev) => prev.filter((_, i) => i !== index));
  }

  function badgeClass(value: string | undefined) {
    const v = (value || "").toLowerCase();
    if (v.includes("succeeded") || v.includes("pass")) return "badge success";
    if (v.includes("failed") || v.includes("blocked") || v.includes("err")) return "badge danger";
    if (v.includes("review") || v.includes("running") || v.includes("polling")) return "badge warn";
    return "badge neutral";
  }

  function normalizeQcIssues(
    raw: JobResult["output"] extends infer O
      ? O extends { qc_issues?: infer Q }
        ? Q
        : never
      : never,
  ) {
    if (!raw) return [] as string[];
    if (Array.isArray(raw)) return raw;
    const out: string[] = [];
    for (const key of ["format", "completeness", "safety"] as const) {
      for (const item of raw[key] || []) out.push(`${key}: ${item}`);
    }
    return out;
  }

  const output = result?.output;
  const inference = output?.inference;
  const rag = output?.rag;
  const ob = output?.observability;
  const qcIssues = normalizeQcIssues(output?.qc_issues);
  const inputMode = notes.trim() && images.length > 0 ? "Multimodal" : notes.trim() ? "Text only" : images.length > 0 ? "Image only" : "Not set";

  return (
    <div className="shell">
      <header className="topbar">
        <div className="brand-mark">M</div>
        <div className="brand-text">
          <div className="brand-eyebrow">Medgent Diagnostic Agent</div>
        <div className="brand-title">Clinical Copilot Console</div>
        </div>
      </header>

      <section className="hero card card-hero">
        <div className="hero-copy">
          <div className="eyebrow">Clinical Pro</div>
          <h1>Clinical Imaging Inference Workbench</h1>
          <div className="hero-tags">
            <span className="tiny-badge">{inputMode}</span>
            <span className="tiny-badge">Images: {images.length}</span>
            <span className={badgeClass(result?.job?.state)}>{result?.job?.state || "unknown"}</span>
          </div>
        </div>
      </section>

      <section className="grid output-first">
        <article className="card output-card">
          <div className="section-head">
            <h2>Structured Output</h2>
            <span>Explainability</span>
          </div>
          <div className="result-layout">
            <section className="panel">
              <h3>Inference</h3>
              <div className="kv-row"><span className="kv-key">confidence</span><span className="kv-value">{inference?.confidence ?? "n/a"}</span></div>
              <div className="kv-row"><span className="kv-key">job_state</span><span className={badgeClass(result?.job?.state)}>{result?.job?.state || "unknown"}</span></div>
              <div className="kv-block"><span className="kv-key">summary</span><p className="kv-text">{inference?.summary || "-"}</p></div>
              <ul>{(inference?.findings || []).map((x, i) => <li key={i}>{x}</li>)}</ul>
            </section>

            <section className="panel">
              <h3>QC</h3>
              <div className="kv-row"><span className="kv-key">qc_status</span><span className={badgeClass(output?.qc_status)}>{output?.qc_status || "n/a"}</span></div>
              <ul>{qcIssues.map((x, i) => <li key={i}>{x}</li>)}</ul>
            </section>

            <section className="panel">
              <h3>RAG</h3>
              <div className="kv-row"><span className="kv-key">query</span><span className="kv-value">{rag?.query || "-"}</span></div>
              <div className="kv-row"><span className="kv-key">hits</span><span className="kv-value">{rag?.hits?.length || 0}</span></div>
              <ul>{(rag?.hits || []).map((h, i) => <li key={i}>{h.title} | score={h.score} | {h.source}</li>)}</ul>
              <details>
                <summary>View injected prompt context</summary>
                <pre>{rag?.context_used || ""}</pre>
              </details>
            </section>

            <section className="panel">
              <h3>Observability</h3>
              <div className="kv-row"><span className="kv-key">rag(ms)</span><span className="kv-value">{ob?.durations_ms?.rag ?? 0}</span></div>
              <div className="kv-row"><span className="kv-key">inference(ms)</span><span className="kv-value">{ob?.durations_ms?.inference ?? 0}</span></div>
              <div className="kv-row"><span className="kv-key">qc(ms)</span><span className="kv-value">{ob?.durations_ms?.qc ?? 0}</span></div>
              <div className="kv-row"><span className="kv-key">total(ms)</span><span className="kv-value">{ob?.durations_ms?.total ?? 0}</span></div>
              <div className="kv-row"><span className="kv-key">run_mode</span><span className="kv-value">{ob?.inference_runtime?.run_mode || "-"}</span></div>
              <div className="kv-row"><span className="kv-key">generated_token_count</span><span className="kv-value">{ob?.inference_runtime?.generated_token_count ?? 0}</span></div>
              <div className="kv-row"><span className="kv-key">used_fallback</span><span className={badgeClass(String(ob?.inference_runtime?.used_fallback ?? false))}>{String(ob?.inference_runtime?.used_fallback ?? false)}</span></div>
            </section>
          </div>
          {status.kind === "polling" && !result && (
            <div className="skeleton-grid" aria-label="loading">
              <div className="skeleton-card" />
              <div className="skeleton-card" />
              <div className="skeleton-card" />
              <div className="skeleton-card" />
            </div>
          )}
        </article>

        <article className="card input-card">
          <div className="section-head">
            <h2>Task Configuration</h2>
            <span>Input & Dispatch</span>
          </div>
          <div className="form-grid">
            <div className="field">
              <label>patient_pseudo_id</label>
              <input value={patientId} onChange={(e) => setPatientId(e.target.value)} />
            </div>
            <div className="field">
              <label>idempotency_key</label>
              <input value={idempotency} onChange={(e) => setIdempotency(e.target.value)} placeholder="Optional, auto-generated if empty" />
            </div>
          </div>
          <details className="advanced">
            <summary>Advanced Settings</summary>
            <div className="form-grid advanced-grid">
              <div className="field">
                <label>API Base</label>
                <input value={apiBase} onChange={(e) => setApiBase(e.target.value)} />
              </div>
              <div className="field">
                <label>x-api-key</label>
                <input value={apiKey} onChange={(e) => setApiKey(e.target.value)} />
              </div>
            </div>
          </details>
          <div className="field">
            <label>Notes (optional)</label>
            <textarea value={notes} onChange={(e) => setNotes(e.target.value)} />
          </div>
          <div className="field">
            <label>Images (optional, multiple)</label>
            <input
              ref={fileInputRef}
              type="file"
              accept="image/*"
              multiple
              hidden
              onChange={(e) => addImages(Array.from(e.target.files || []))}
            />
            <div
              className={`dropzone ${isDragging ? "dragging" : ""}`}
              onClick={() => fileInputRef.current?.click()}
              onDragEnter={(e) => {
                e.preventDefault();
                setIsDragging(true);
              }}
              onDragOver={(e) => e.preventDefault()}
              onDragLeave={(e) => {
                e.preventDefault();
                setIsDragging(false);
              }}
              onDrop={(e) => {
                e.preventDefault();
                setIsDragging(false);
                addImages(Array.from(e.dataTransfer.files || []));
              }}
            >
              <div className="drop-icon">+</div>
              <div className="drop-title">Click or drag images here</div>
              <div className="drop-sub">DICOM preview images, PNG, JPG</div>
            </div>
            {images.length > 0 && (
              <div className="file-chips">
                {images.map((file, idx) => (
                  <button key={`${file.name}-${idx}`} type="button" className="chip" onClick={() => removeImage(idx)}>
                    {file.name} x
                  </button>
                ))}
              </div>
            )}
          </div>

          <div className="actions">
            <button className="btn-primary" onClick={runWorkflow}>Submit and Poll</button>
            <button className="btn-soft" onClick={pingInference}>Check Inference Service</button>
            <button className="btn-soft" onClick={clearAll}>Clear</button>
          </div>

          <div className={`status-pill ${statusClass}`}><span className="dot" /><span>{status.text}</span></div>
        </article>
      </section>

      <section className="card log-wrap">
        <details>
          <summary className="log-summary">Execution Log (Debug Trace)</summary>
          <pre className="raw-log">{logs}</pre>
        </details>
      </section>
    </div>
  );
}
