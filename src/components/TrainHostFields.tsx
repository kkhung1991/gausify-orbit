import { useEffect, useState } from "react";
import { useI18n } from "../i18n";
import type { Messages } from "../i18n/pt-BR";
import type { TrainBackend, TrainHost } from "../lib/api";
import { TrainHostClient, TrainHostError, type HostInfo } from "../lib/trainhost";
import { Field, Select } from "./fields";

function errorText(t: Messages, e: unknown): string {
  const te = t.sidebar.hostErrors;
  if (e instanceof TrainHostError) {
    switch (e.reason) {
      case "network":
        return te.network;
      case "auth":
        return te.auth;
      case "notHost":
        return te.notHost;
      case "http":
        return te.http(e.detail);
    }
  }
  return (e as Error)?.name === "TimeoutError" ? te.network : String((e as Error)?.message ?? e);
}

/** "2× RTX 4090" for identical GPUs, the names otherwise. */
function gpuSummary(info: HostInfo): string {
  const names = info.gpus.map((g) => g.name.replace(/^NVIDIA\s+/i, "").replace(/^GeForce\s+/i, ""));
  return new Set(names).size === 1 && names.length > 1 ? `${names.length}× ${names[0]}` : names.join(", ");
}

type Probe = { state: "idle" | "testing" } | { state: "ok"; info: HostInfo } | { state: "error"; text: string };

/** Where training runs: this browser, or a training host (docs/host.md) with
 *  native Brush or gsplat, and which of the host's GPUs. */
export function TrainHostFields({ value, onChange }: { value: TrainHost; onChange: (h: TrainHost) => void }) {
  const { t } = useI18n();
  const ts = t.sidebar;
  const [probe, setProbe] = useState<Probe>({ state: "idle" });
  const [attempt, setAttempt] = useState(0);
  const remote = value.backend !== "browser";

  // Asks the host what it has whenever its address changes (and on "Test").
  useEffect(() => {
    if (!remote || !value.url.trim()) return setProbe({ state: "idle" });
    const ctl = new AbortController();
    const timer = setTimeout(async () => {
      setProbe({ state: "testing" });
      try {
        const info = await new TrainHostClient(value).info(AbortSignal.any([ctl.signal, AbortSignal.timeout(8000)]));
        if (!ctl.signal.aborted) setProbe({ state: "ok", info });
      } catch (e) {
        if (!ctl.signal.aborted) setProbe({ state: "error", text: errorText(t, e) });
      }
    }, 500);
    return () => {
      clearTimeout(timer);
      ctl.abort();
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [remote, value.url, value.token, attempt]);

  const info = probe.state === "ok" ? probe.info : null;
  const trainer = info && remote ? info.backends[value.backend] : null;
  const hints: Record<TrainBackend, string> = { browser: ts.trainOnBrowserHint, brush: ts.trainOnBrushHint, gsplat: ts.trainOnGsplatHint };
  // A GPU chosen earlier stays listed while the host isn't reachable.
  const gpus = info?.gpus ?? (typeof value.gpu === "number" ? [{ index: value.gpu, name: "", memoryMb: null, usedMb: null, jobs: 0 }] : []);

  return (
    <>
      <Field label={ts.trainOn} hint={hints[value.backend]}>
        <Select<TrainBackend>
          value={value.backend}
          options={[
            { value: "browser", label: ts.trainOnBrowser },
            { value: "brush", label: ts.trainOnBrush },
            { value: "gsplat", label: ts.trainOnGsplat },
          ]}
          onChange={(backend) => onChange({ ...value, backend })}
        />
      </Field>
      {remote && (
        <>
          <Field label={ts.hostUrl} hint={ts.hostUrlHint}>
            <span className="comfy-url-row">
              <input
                className="input mono"
                value={value.url}
                spellCheck={false}
                placeholder="http://localhost:1441"
                onChange={(e) => onChange({ ...value, url: e.currentTarget.value })}
              />
              <button type="button" className="btn-sm" onClick={() => setAttempt((n) => n + 1)} disabled={probe.state === "testing" || !value.url.trim()}>
                {probe.state === "testing" ? ts.hostTesting : ts.hostTest}
              </button>
            </span>
          </Field>
          {probe.state === "error" && <p className="comfy-test bad">{probe.text}</p>}
          {info && <p className={`comfy-test ${trainer?.ok ? "good" : "bad"}`}>{trainer?.ok ? ts.hostConnected(gpuSummary(info)) : ts.hostTrainerMissing(trainer?.detail ?? value.backend)}</p>}
          <Field label={ts.hostGpu} hint={value.gpu === "auto" ? ts.hostGpuAutoHint : undefined}>
            <Select<number | "auto">
              value={value.gpu}
              options={[
                { value: "auto", label: ts.hostGpuAuto },
                ...gpus.map((g) => ({
                  value: g.index,
                  label: [`GPU ${g.index}`, g.name.replace(/^NVIDIA\s+/i, ""), g.memoryMb && `${Math.round(g.memoryMb / 1024)} GB`, g.jobs > 0 && ts.hostGpuBusy].filter(Boolean).join(" · "),
                })),
              ]}
              onChange={(gpu) => onChange({ ...value, gpu })}
            />
          </Field>
          <Field label={ts.hostToken} hint={ts.hostTokenHint}>
            <input className="input mono" type="password" autoComplete="off" value={value.token} spellCheck={false} onChange={(e) => onChange({ ...value, token: e.currentTarget.value })} />
          </Field>
          <p className="help">{ts.hostHelp}</p>
        </>
      )}
    </>
  );
}
