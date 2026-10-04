import { execFile } from "node:child_process";
import { promisify } from "node:util";
import { fileURLToPath } from "node:url";
import path from "node:path";
import { Type } from "@sinclair/typebox";

const executeFile = promisify(execFile);
export const root = fileURLToPath(new URL("../../", import.meta.url));
export const python = path.join(root, ".venv", process.platform === "win32" ? "Scripts/python.exe" : "bin/python");

export async function pythonBridge(round, operation, candidate, signal) {
  const args = ["-m", "physio_autoresearch", operation, path.resolve(round)];
  if (operation === "attempt") args.push("--candidate-json", JSON.stringify(candidate));
  let stdout;
  try {
    ({ stdout } = await executeFile(python, args, { cwd: root, shell: false, signal,
      timeout: 650_000, maxBuffer: 2 * 1024 * 1024, windowsHide: true }));
  } catch (error) {
    if (error.stdout) {
      try { return JSON.parse(error.stdout); } catch { /* no raw process output goes to the agent */ }
    }
    throw new Error("Research worker failed; inspect its local logs.");
  }
  return JSON.parse(stdout);
}

export function publicResult(result) {
  // The model sees configuration and aggregate scores, never subject IDs, raw
  // feature values, labels, predictions, local filesystem paths or credentials.
  const candidate = (entry) => entry && ({ attempt: entry.attempt, candidate: entry.candidate,
    candidate_sha256: entry.candidate_sha256, decision: entry.decision, reason: entry.reason,
    elapsed_seconds: entry.elapsed_seconds,
    evaluation: entry.evaluation && { metric: entry.evaluation.metric, status: entry.evaluation.status } });
  if (result.error) return { error: result.error };
  if (result.kind === "result") return candidate(result);
  return { data_version: result.data_version, source: result.source, subjects: result.qc.subjects,
    attempts: result.attempts, budget: result.budget, elapsed_seconds: result.elapsed_seconds,
    model_usage: result.model_usage,
    stop_reason: result.stop_reason,
    pending_attempts: result.pending_attempts, champion: candidate(result.champion),
    recent_results: result.recent_results.map(candidate), model_status: result.model_status };
}

export function researchTools(round, bridge = pythonBridge) {
  const output = (value) => ({ content: [{ type: "text", text: JSON.stringify(publicResult(value)) }], details: {} });
  return [
    { name: "research_status", label: "Research status", description: "Read aggregate frozen-round status and recent experiment memory.",
      parameters: Type.Object({}, { additionalProperties: false }),
      execute: async (_id, _params, signal) => output(await bridge(round, "status", undefined, signal)) },
    { name: "research_candidate", label: "Evaluate candidate", description: "Submit a bounded candidate configuration. The fixed Python trainer/evaluator decides KEEP/REVERT and enforces the persisted data-version budget.",
      parameters: Type.Object({
        model: Type.Union([Type.Literal("naive"), Type.Literal("elastic_net"), Type.Literal("random_forest"), Type.Literal("xgboost")]),
        params: Type.Optional(Type.Record(Type.String(), Type.Number())),
        features: Type.Optional(Type.Array(Type.String(), { minItems: 1, uniqueItems: true })),
        hypothesis: Type.String({ maxLength: 2000 }),
      }, { additionalProperties: false }),
      execute: async (_id, params, signal) => output(await bridge(round, "attempt", params, signal)) },
  ];
}
