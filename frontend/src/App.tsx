import { useMemo, useState } from "react";

type JobResult = {
  job?: { state?: string; case_id?: string };
  output?: {
    inference?: {
      summary?: string;
      findings?: string[];
      confidence?: number;
    };
    qc_status?: string;
    qc_issues?: string[];
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
  const [status, setStatus] = useState<{ text: string; ok: boolean }>({ text: "待命", ok: true });
  const [logs, setLogs] = useState<string>("");
  const [result, setResult] = useState<JobResult | null>(null);

  const statusClass = useMemo(() => (status.ok ? "ok" : "err"), [status.ok]);

  function appendLog(title: string, data: unknown) {
    const block = `\n=== ${title} ===\n${typeof data === "string" ? data : JSON.stringify(data, null, 2)}\n`;
    setLogs((prev) => prev + block);
  }

  function headers() {
    if (!apiKey.trim()) throw new Error("请先填写 x-api-key");
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
      setStatus({ text: "检查中...", ok: true });
      const data = await requestJson(`${apiBase.replace(/\/$/, "")}/inference/ping`, { headers: headers() });
      appendLog("inference_ping", data);
      setStatus({ text: `推理服务: ${data.reachable ? "reachable" : "unreachable"}`, ok: !!data.reachable });
    } catch (err) {
      const msg = String((err as Error).message ?? err);
      setStatus({ text: msg, ok: false });
      appendLog("error", msg);
    }
  }

  async function runWorkflow() {
    try {
      setStatus({ text: "执行中...", ok: true });
      setResult(null);
      const base = apiBase.replace(/\/$/, "");
      const h = headers();
      if (!notes.trim() && images.length === 0) throw new Error("Notes 和 Images 不能同时为空");

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
          setStatus({ text: `完成: ${poll.job?.state}`, ok: poll.job?.state === "succeeded" });
          return;
        }
      }
      throw new Error("轮询超时: 请确认 worker 正在运行");
    } catch (err) {
      const msg = String((err as Error).message ?? err);
      setStatus({ text: msg, ok: false });
      appendLog("error", msg);
    }
  }

  function clearAll() {
    setLogs("");
    setResult(null);
    setStatus({ text: "待命", ok: true });
  }

  const output = result?.output;
  const inference = output?.inference;
  const rag = output?.rag;
  const ob = output?.observability;

  return (
    <div className="shell">
      <section className="hero">
        <div className="eyebrow">Medgent Diagnostic Agent</div>
        <h1>临床影像智能推理工作台</h1>
        <p>支持文本、影像、图文混合输入。执行 Case 创建、Artifact 上传、Job 调度、RAG 检索、模型推理与 QC。</p>
      </section>

      <section className="grid">
        <article className="card">
          <h2>任务配置</h2>
          <div className="form-grid">
            <div className="field">
              <label>API Base</label>
              <input value={apiBase} onChange={(e) => setApiBase(e.target.value)} />
            </div>
            <div className="field">
              <label>x-api-key</label>
              <input value={apiKey} onChange={(e) => setApiKey(e.target.value)} />
            </div>
            <div className="field">
              <label>patient_pseudo_id</label>
              <input value={patientId} onChange={(e) => setPatientId(e.target.value)} />
            </div>
            <div className="field">
              <label>idempotency_key</label>
              <input value={idempotency} onChange={(e) => setIdempotency(e.target.value)} placeholder="可留空自动生成" />
            </div>
          </div>
          <div className="field">
            <label>Notes (可选)</label>
            <textarea value={notes} onChange={(e) => setNotes(e.target.value)} />
          </div>
          <div className="field">
            <label>Images (可选，多张)</label>
            <input type="file" accept="image/*" multiple onChange={(e) => setImages(Array.from(e.target.files || []))} />
            <div className="hint">支持 text-only / image-only / multimodal</div>
          </div>

          <div className="actions">
            <button className="btn-primary" onClick={runWorkflow}>提交并轮询</button>
            <button className="btn-soft" onClick={pingInference}>检查推理服务</button>
            <button className="btn-soft" onClick={clearAll}>清空</button>
          </div>

          <div className={`status-pill ${statusClass}`}><span className="dot" /><span>{status.text}</span></div>
        </article>

        <article className="card">
          <h2>结构化结果</h2>
          <div className="result-layout">
            <section className="panel">
              <h3>Inference</h3>
              <div className="kv"><strong>summary:</strong> {inference?.summary || ""}</div>
              <div className="kv"><strong>confidence:</strong> {inference?.confidence ?? "n/a"}</div>
              <div className="kv"><strong>job_state:</strong> {result?.job?.state || "unknown"}</div>
              <ul>{(inference?.findings || []).map((x, i) => <li key={i}>{x}</li>)}</ul>
            </section>

            <section className="panel">
              <h3>QC</h3>
              <div className="kv"><strong>qc_status:</strong> {output?.qc_status || "n/a"}</div>
              <ul>{(output?.qc_issues || []).map((x, i) => <li key={i}>{x}</li>)}</ul>
            </section>

            <section className="panel">
              <h3>RAG</h3>
              <div className="kv"><strong>query:</strong> {rag?.query || ""}</div>
              <div className="kv"><strong>hits:</strong> {rag?.hits?.length || 0}</div>
              <ul>{(rag?.hits || []).map((h, i) => <li key={i}>{h.title} | score={h.score} | {h.source}</li>)}</ul>
              <details>
                <summary>查看注入 Prompt 的 context</summary>
                <pre>{rag?.context_used || ""}</pre>
              </details>
            </section>

            <section className="panel">
              <h3>Observability</h3>
              <div className="kv"><strong>durations(ms):</strong> rag={ob?.durations_ms?.rag ?? 0}, inference={ob?.durations_ms?.inference ?? 0}, qc={ob?.durations_ms?.qc ?? 0}, total={ob?.durations_ms?.total ?? 0}</div>
              <div className="kv"><strong>run_mode:</strong> {ob?.inference_runtime?.run_mode || ""}</div>
              <div className="kv"><strong>model_source:</strong> {ob?.inference_runtime?.model_source || ""}</div>
              <div className="kv"><strong>generated_token_count:</strong> {ob?.inference_runtime?.generated_token_count ?? 0}</div>
              <div className="kv"><strong>used_fallback:</strong> {String(ob?.inference_runtime?.used_fallback ?? false)}</div>
            </section>
          </div>
        </article>
      </section>

      <section className="card log-wrap">
        <h2>执行日志</h2>
        <pre className="raw-log">{logs}</pre>
      </section>
    </div>
  );
}
