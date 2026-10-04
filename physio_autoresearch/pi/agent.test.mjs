import assert from "node:assert/strict";
import fs from "node:fs";
import path from "node:path";
import test from "node:test";
import { execFile } from "node:child_process";
import { promisify } from "node:util";
import { ModelRuntime } from "@earendil-works/pi-coding-agent";
import { createAssistantMessageEventStream } from "@earendil-works/pi-ai/utils/event-stream";
import { createResearchSession } from "./agent.mjs";
import { publicResult, python, pythonBridge, researchTools, root } from "./tools.mjs";
import { reconstructResearch } from "./upstream.mjs";
import { withPiLock } from "./lock.mjs";

const executeFile = promisify(execFile);
const fakeStatus = { data_version: "demo", source: "synthetic", qc: { subjects: 5 }, attempts: 0,
  budget: { max_experiments: 20 }, elapsed_seconds: 0, pending_attempts: [], champion: null,
  recent_results: [], model_status: "SMOKE_MODEL_NOT_VALIDATED" };

test("provider-visible result excludes raw data, subject scores and local paths", () => {
  const result = publicResult({ ...fakeStatus, champion: { attempt: 1, artifact_directory: "secret-path", candidate: { model: "naive" },
    evaluation: { metric: 1, per_subject_normalized_mae: { "private-subject": 99 }, status: "SMOKE" } } });
  const text = JSON.stringify(result);
  assert.ok(!text.includes("private-subject"));
  assert.ok(!text.includes("secret-path"));
  assert.equal(result.champion.evaluation.metric, 1);
});

test("pure upstream history module reconstructs only actually scored trials", () => {
  const state = reconstructResearch([{ kind: "result", attempt: 1, decision: "KEEP", candidate: { model: "naive" },
    evaluation: { metric: 0.5 }, elapsed_seconds: 1, finished_at: new Date().toISOString(), reason: "baseline" },
  { kind: "result", attempt: 2, decision: "REVERT", evaluation: null, reason: "TRAIN_CRASH" }]);
  assert.equal(state.results.length, 1);
  assert.equal(state.results[0].metric, 0.5);
  assert.equal(state.bestDirection, "lower");
});

test("real Pi SDK session exposes only research tools and executes a scripted offline tool turn", async () => {
  fs.mkdirSync(path.join(root, "runs"), { recursive: true });
  const runtimeDirectory = fs.mkdtempSync(path.join(root, "runs/pi-sdk-test-"));
  const modelRuntime = await ModelRuntime.create({ authPath: path.join(runtimeDirectory, "auth.json"), modelsPath: null,
    modelsStorePath: path.join(runtimeDirectory, "models.json"), allowModelNetwork: false, refreshOnCreate: false });
  const model = { id: "offline-test", name: "offline-test", provider: "offline-test", api: "openai-completions",
    reasoning: false, input: ["text"], cost: { input: 0, output: 0, cacheRead: 0, cacheWrite: 0 }, contextWindow: 8192, maxTokens: 1024 };
  let calls = 0;
  const { session } = await createResearchSession({ round: runtimeDirectory, runtimeDirectory, modelRuntime, model,
    bridge: async () => { calls++; return fakeStatus; } });
  try {
    assert.deepEqual(session.getActiveToolNames().sort(), ["research_candidate", "research_status"]);
    let turns = 0;
    session.agent.streamFunction = () => {
      const stream = createAssistantMessageEventStream();
      const message = { role: "assistant", content: turns++ === 0 ? [{ type: "toolCall", id: "offline-1", name: "research_status", arguments: {} }]
        : [{ type: "text", text: "Smoke pipeline checked." }], api: model.api, provider: model.provider, model: model.id,
      usage: { input: 0, output: 0, cacheRead: 0, cacheWrite: 0, totalTokens: 0,
        cost: { input: 0, output: 0, cacheRead: 0, cacheWrite: 0, total: 0 } }, stopReason: turns === 1 ? "toolUse" : "stop", timestamp: Date.now() };
      stream.push({ type: "start", partial: message });
      stream.push({ type: "done", reason: message.stopReason, message });
      stream.end(message);
      return stream;
    };
    await session.agent.prompt("Check the research status.");
    assert.equal(calls, 1);
    assert.equal(turns, 2);
  } finally { session.dispose(); }
});

test("Pi tool invokes real Python ingestion/training/evaluation through shell-free bridge", async () => {
  const directory = fs.mkdtempSync(path.join(root, "runs/pi-bridge-test-"));
  const code = `from pathlib import Path\nimport sys\nfrom physio_autoresearch.__main__ import make_synthetic\nfrom physio_autoresearch.harness import ingest\np=Path(sys.argv[1]); make_synthetic(p/'input.csv',2); print(ingest(p/'input.csv',Path('physio_autoresearch/constitution.synthetic.yaml'),p/'store'))`;
  const { stdout } = await executeFile(python, ["-c", code, directory], { cwd: root, shell: false, windowsHide: true });
  const round = stdout.trim();
  await withPiLock(round, async () => {
    await assert.rejects(withPiLock(round, async () => assert.fail("concurrent Pi entered")), /PI_SESSION_BUSY_OR_INVALID_ROUND/);
  });
  await withPiLock(round, async () => {}); // OS lock released, no stale file assumption
  const before = await pythonBridge(round, "status");
  assert.equal(before.qc.subjects, 2);
  const tools = researchTools(round);
  const result = await tools[1].execute("bridge-test", { model: "naive", hypothesis: "real offline bridge test" });
  const value = JSON.parse(result.content[0].text);
  assert.equal(value.decision, "KEEP");
  assert.ok(Number.isFinite(value.evaluation.metric));
  const rejected = await tools[1].execute("leakage-test", { model: "elastic_net", features: ["post_hr"], hypothesis: "should be rejected" });
  assert.equal(JSON.parse(rejected.content[0].text).reason, "FUTURE_FEATURE_LEAKAGE");
});
