// Reuse the pinned upstream's pure JSONL state reconstruction, without loading
// its full-permission experiment/commit/revert extension.
import fs from "node:fs";
import path from "node:path";
import { stripTypeScriptTypes } from "node:module";
import { root } from "./tools.mjs";

const packageRoot = path.join(root, "node_modules/pi-autoresearch");
const manifest = JSON.parse(fs.readFileSync(path.join(packageRoot, "package.json"), "utf8"));
if (manifest.version !== "1.8.1") throw new Error("Unexpected pi-autoresearch version; review before upgrading.");
// Node refuses direct type stripping inside node_modules; transform the single
// dependency-free pure module in memory through Node's standard TS API.
const source = fs.readFileSync(path.join(packageRoot, "extensions/pi-autoresearch/jsonl.ts"), "utf8");
const module = await import(`data:text/javascript;base64,${Buffer.from(stripTypeScriptTypes(source)).toString("base64")}`);

export function reconstructResearch(entries) {
  const evaluated = entries.filter(entry => entry.kind === "result" && Number.isFinite(entry.evaluation?.metric));
  const lines = [{ type: "config", name: "Physio smoke/development research", metricName: "subject_macro_normalized_mae",
    metricUnit: "normalized", bestDirection: "lower" },
  ...evaluated.map(entry => ({ run: entry.attempt, commit: "local", metric: entry.evaluation.metric,
    metrics: { wall_s: entry.elapsed_seconds }, status: entry.decision === "KEEP" ? "keep" : "discard",
    description: `${entry.candidate.model}: ${entry.reason}`, timestamp: Date.parse(entry.finished_at), confidence: null }))];
  return module.reconstructJsonlState(lines.map(line => JSON.stringify(line)).join("\n"));
}
