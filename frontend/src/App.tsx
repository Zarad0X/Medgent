import { useEffect, useMemo, useRef, useState } from "react";

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

type ChatMessage = {
  id: string;
  role: "user" | "assistant";
  text: string;
  loading?: boolean;
  result?: JobResult;
};

const DEFAULT_API_BASE = import.meta.env.VITE_API_BASE ?? "http://127.0.0.1:8000/api/v1";
const DEFAULT_API_KEY = import.meta.env.VITE_API_KEY ?? "dev-local-key";

function shortText(text: string, max = 180) {
  if (text.length <= max) return text;
  return `${text.slice(0, max - 3)}...`;
}

export default function App() {
  const [apiBase, setApiBase] = useState(DEFAULT_API_BASE);
  const [apiKey, setApiKey] = useState(DEFAULT_API_KEY);
  const [patientId, setPatientId] = useState("p-ui-001");
  const [idempotency, setIdempotency] = useState("");
  const [notes, setNotes] = useState("");
  const [images, setImages] = useState<File[]>([]);
  const [imagePreviews, setImagePreviews] = useState<string[]>([]);
  const [isDragging, setIsDragging] = useState(false);
  const [status, setStatus] = useState<{ text: string; kind: "idle" | "polling" | "ok" | "err" }>({
    text: "Idle",
    kind: "idle",
  });
  const [messages, setMessages] = useState<ChatMessage[]>([
    {
      id: crypto.randomUUID(),
      role: "assistant",
      text: "Ready. Send notes, images, or both to run a follow-up analysis.",
    },
  ]);
  const [logs, setLogs] = useState<string>("");

  const fileInputRef = useRef<HTMLInputElement | null>(null);
  const statusClass = useMemo(() => status.kind, [status.kind]);
  const inputMode = notes.trim() && images.length > 0 ? "Multimodal" : notes.trim() ? "Text only" : images.length > 0 ? "Image only" : "Not set";
  const quickPrompts = [
    "Follow-up CT: evaluate lesion progression and recommendations.",
    "Summarize key findings and whether disease is stable.",
    "Generate a concise radiology follow-up report in JSON style.",
    "Only image input: extract likely abnormal findings.",
  ];

  useEffect(() => {
    const previews = images.map((file) => URL.createObjectURL(file));
    setImagePreviews(previews);
    return () => {
      previews.forEach((url) => URL.revokeObjectURL(url));
    };
  }, [images]);

  function appendLog(title: string, data: unknown) {
    const block = `\n=== ${title} ===\n${typeof data === "string" ? data : JSON.stringify(data, null, 2)}\n`;
    setLogs((prev) => prev + block);
  }

  function headers() {
    if (!apiKey.trim()) throw new Error("Please provide x-api-key first");
    return { "x-api-key": apiKey.trim() };
  }

  function safeJson(text: string) {
    try {
      return JSON.parse(text);
    } catch {
      return { raw: text };
    }
  }

  async function requestJson(url: string, options: RequestInit = {}) {
    const res = await fetch(url, options);
    const text = await res.text();
    const json = text ? safeJson(text) : null;
    if (!res.ok) throw new Error(`${res.status} ${res.statusText}: ${JSON.stringify(json)}`);
    return json as any;
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

  function normalizeQcIssues(
    raw:
      | string[]
      | {
          format?: string[];
          completeness?: string[];
          safety?: string[];
        }
      | undefined,
  ) {
    if (!raw) return [] as string[];
    if (Array.isArray(raw)) return raw;
    const out: string[] = [];
    for (const key of ["format", "completeness", "safety"] as const) {
      for (const item of raw[key] || []) out.push(`${key}: ${item}`);
    }
    return out;
  }

  function badgeClass(value: string | undefined) {
    const v = (value || "").toLowerCase();
    if (v.includes("succeeded") || v.includes("pass") || v.includes("ok")) return "badge success";
    if (v.includes("failed") || v.includes("blocked") || v.includes("err")) return "badge danger";
    if (v.includes("review") || v.includes("running") || v.includes("polling")) return "badge warn";
    return "badge neutral";
  }

  function addImages(files: File[]) {
    if (!files.length) return;
    setImages((prev) => [...prev, ...files]);
  }

  function removeImage(index: number) {
    setImages((prev) => prev.filter((_, i) => i !== index));
  }

  async function runWorkflow() {
    const inputNotes = notes.trim();
    const inputImages = images;

    if (!inputNotes && inputImages.length === 0) {
      setStatus({ text: "Notes and Images cannot both be empty", kind: "err" });
      return;
    }

    const userText = inputNotes
      ? shortText(inputNotes)
      : `Image-only input (${inputImages.length} file${inputImages.length > 1 ? "s" : ""})`;

    const userMessage: ChatMessage = {
      id: crypto.randomUUID(),
      role: "user",
      text: userText,
    };
    const assistantLoadingId = crypto.randomUUID();

    setMessages((prev) => [
      ...prev,
      userMessage,
      { id: assistantLoadingId, role: "assistant", text: "Running analysis...", loading: true },
    ]);

    try {
      setStatus({ text: "Processing and polling...", kind: "polling" });
      const base = apiBase.replace(/\/$/, "");
      const h = headers();

      const caseResp = await requestJson(`${base}/cases`, {
        method: "POST",
        headers: { ...h, "Content-Type": "application/json" },
        body: JSON.stringify({ patient_pseudo_id: patientId.trim() || "p-ui-001" }),
      });
      appendLog("case_created", caseResp);

      const caseId = caseResp.case_id as string;
      if (inputNotes) {
        const notesFile = new File([inputNotes], "notes.txt", { type: "text/plain;charset=utf-8" });
        appendLog("notes_uploaded", await uploadArtifact(caseId, "input_notes", notesFile));
      }

      for (const image of inputImages) {
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

      let final: JobResult | null = null;
      for (let i = 0; i < 60; i += 1) {
        await new Promise((r) => setTimeout(r, 1000));
        const poll = (await requestJson(`${base}/workflow/jobs/${jobId}/result`, {
          method: "GET",
          headers: h,
        })) as JobResult;
        appendLog(`poll_${i + 1}`, { job_state: poll.job?.state, has_output: !!poll.output });
        if (poll.job?.state === "succeeded" || poll.job?.state === "failed") {
          final = poll;
          break;
        }
      }

      if (!final) throw new Error("Polling timeout: make sure worker is running");

      appendLog("final_result", final);
      const summary = final.output?.inference?.summary || "No summary from model.";
      setMessages((prev) =>
        prev.map((m) =>
          m.id === assistantLoadingId
            ? {
                id: assistantLoadingId,
                role: "assistant",
                text: summary,
                loading: false,
                result: final || undefined,
              }
            : m,
        ),
      );
      setStatus({ text: `Completed: ${final.job?.state}`, kind: final.job?.state === "succeeded" ? "ok" : "err" });
      setNotes("");
      setImages([]);
    } catch (err) {
      const msg = String((err as Error).message ?? err);
      appendLog("error", msg);
      setMessages((prev) =>
        prev.map((m) =>
          m.id === assistantLoadingId
            ? { id: assistantLoadingId, role: "assistant", text: `Error: ${msg}`, loading: false }
            : m,
        ),
      );
      setStatus({ text: msg, kind: "err" });
    }
  }

  async function pingInference() {
    try {
      setStatus({ text: "Checking inference service...", kind: "polling" });
      const data = await requestJson(`${apiBase.replace(/\/$/, "")}/inference/ping`, { headers: headers() });
      appendLog("inference_ping", data);
      const reach = !!data.reachable;
      setStatus({ text: `Inference service: ${reach ? "reachable" : "unreachable"}`, kind: reach ? "ok" : "err" });
    } catch (err) {
      const msg = String((err as Error).message ?? err);
      appendLog("error", msg);
      setStatus({ text: msg, kind: "err" });
    }
  }

  function clearAll() {
    setLogs("");
    setMessages([
      {
        id: crypto.randomUUID(),
        role: "assistant",
        text: "Ready. Send notes, images, or both to run a follow-up analysis.",
      },
    ]);
    setStatus({ text: "Idle", kind: "idle" });
  }

  return (
    <div className="agent-shell">
      <header className="agent-topbar">
        <div className="agent-brand">M</div>
        <div>
          <div className="agent-title">Medgent Diagnostic Agent</div>
          <div className="agent-subtitle">Clinical Copilot</div>
        </div>
        <div className={`status-pill ${statusClass}`}><span className="dot" /><span>{status.text}</span></div>
      </header>

      <section className="workspace">
        <aside className="side-panel">
          <section className="side-card">
            <h3>Session</h3>
            <div className="meta-row"><span>Input mode</span><strong>{inputMode}</strong></div>
            <div className="meta-row"><span>Selected images</span><strong>{images.length}</strong></div>
            <div className="field">
              <label>patient_pseudo_id</label>
              <input value={patientId} onChange={(e) => setPatientId(e.target.value)} />
            </div>
            <div className="field">
              <label>idempotency_key</label>
              <input value={idempotency} onChange={(e) => setIdempotency(e.target.value)} placeholder="Optional" />
            </div>
          </section>

          <details className="advanced compact">
            <summary>Advanced Settings</summary>
            <div className="advanced-grid">
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

          <div className="side-actions">
            <button className="btn-soft" onClick={pingInference}>Check Service</button>
            <button className="btn-soft" onClick={clearAll}>Clear</button>
          </div>

          <details className="log-panel">
            <summary>Execution Log</summary>
            <pre className="raw-log">{logs}</pre>
          </details>
        </aside>

        <div className="main-panel">
          <main className="chat-frame">
            <section className="messages">
              {messages.map((m) => {
                const output = m.result?.output;
                const inference = output?.inference;
                const rag = output?.rag;
                const ob = output?.observability;
                const qcIssues = normalizeQcIssues(output?.qc_issues);

                return (
                  <article key={m.id} className={`message ${m.role}`}>
                    <div className="bubble">
                      {m.loading ? <div className="typing"><span /><span /><span /></div> : <p>{m.text}</p>}

                      {m.role === "assistant" && m.result && (
                        <details className="structured" open>
                          <summary>Structured Output</summary>
                          <div className="grid-mini">
                            <div className="mini-card">
                              <h4>Inference</h4>
                              <div className="kv"><span>confidence</span><strong>{inference?.confidence ?? "n/a"}</strong></div>
                              <div className="kv"><span>job_state</span><span className={badgeClass(m.result?.job?.state)}>{m.result?.job?.state || "unknown"}</span></div>
                            </div>
                            <div className="mini-card">
                              <h4>QC</h4>
                              <div className="kv"><span>qc_status</span><span className={badgeClass(output?.qc_status)}>{output?.qc_status || "n/a"}</span></div>
                              <ul>{qcIssues.map((x, i) => <li key={i}>{x}</li>)}</ul>
                            </div>
                            <div className="mini-card">
                              <h4>Findings</h4>
                              <ul>{(inference?.findings || []).map((x, i) => <li key={i}>{x}</li>)}</ul>
                            </div>
                            <div className="mini-card">
                              <h4>RAG / Runtime</h4>
                              <div className="kv"><span>hits</span><strong>{rag?.hits?.length || 0}</strong></div>
                              <div className="kv"><span>mode</span><strong>{ob?.inference_runtime?.run_mode || "-"}</strong></div>
                              <div className="kv"><span>total(ms)</span><strong>{ob?.durations_ms?.total ?? 0}</strong></div>
                              <details>
                                <summary>Prompt context</summary>
                                <pre>{rag?.context_used || ""}</pre>
                              </details>
                            </div>
                          </div>
                        </details>
                      )}
                    </div>
                  </article>
                );
              })}
            </section>
          </main>

          <section className="composer-wrap">
            <div className="composer">
              <input
                ref={fileInputRef}
                type="file"
                accept="image/*"
                multiple
                hidden
                onChange={(e) => addImages(Array.from(e.target.files || []))}
              />
              <div
                className={`prompt-dock ${isDragging ? "dragging" : ""}`}
                onDragEnter={(e) => {
                  e.preventDefault();
                  setIsDragging(true);
                }}
                onDragOver={(e) => {
                  e.preventDefault();
                  setIsDragging(true);
                }}
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
                <textarea
                  className="prompt-input"
                  value={notes}
                  onChange={(e) => setNotes(e.target.value)}
                  placeholder="Ask Medgent. You can input notes, upload images, or both..."
                />

                <div className="prompt-toolbar">
                  <div className="tool-group">
                    <button type="button" className="tool-btn" onClick={() => fileInputRef.current?.click()}>
                      + Add images
                    </button>
                  </div>
                  <button className="btn-primary send-btn" onClick={runWorkflow}>
                    Send
                  </button>
                </div>
              </div>

              {images.length > 0 && (
                <div className="image-strip">
                  {images.map((file, idx) => (
                    <div key={`${file.name}-${idx}`} className="image-thumb-card">
                      <img className="image-thumb" src={imagePreviews[idx]} alt={file.name} />
                      <button
                        type="button"
                        className="image-remove-x"
                        aria-label={`Remove ${file.name}`}
                        onClick={() => removeImage(idx)}
                      >
                        x
                      </button>
                    </div>
                  ))}
                </div>
              )}

              <div className="quick-prompts">
                {quickPrompts.map((prompt) => (
                  <button key={prompt} type="button" className="quick-chip" onClick={() => setNotes(prompt)}>
                    {prompt}
                  </button>
                ))}
              </div>
            </div>
          </section>
        </div>
      </section>
    </div>
  );
}
