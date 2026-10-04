import fs from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";
import { parseArgs } from "node:util";
import { createAgentSession, createExtensionRuntime, ModelRuntime, SessionManager, SettingsManager } from "@earendil-works/pi-coding-agent";
import { publicResult, pythonBridge, researchTools, root } from "./tools.mjs";
import { reconstructResearch } from "./upstream.mjs";
import { withPiLock } from "./lock.mjs";

export const RESEARCH_PROMPT = `You run bounded offline physiological research.
Use only research_status and research_candidate. First obtain status. Start with
naive if no baseline exists. Then propose small ElasticNet/RF/XGBoost candidates,
using recent results and explicit hypotheses. Feature names come only from the
frozen contract supplied by the host. Never use post outcomes as inputs, access
test sets, change target/scales/split/metric/budgets, request code execution or
modify trusted code. Stop on budget exhaustion, interruption or patience limit.
SMOKE_MODEL is not validated physiology. Return the current champion and limits.
KEEP/REVERT is decided by the evaluator, not by you. No Git operations are allowed.`;

export async function createResearchSession({ round, model, modelRuntime, runtimeDirectory, bridge = pythonBridge }) {
  const customTools = researchTools(round, bridge);
  const resourceLoader = {
    getExtensions: () => ({ extensions: [], errors: [], runtime: createExtensionRuntime() }),
    getSkills: () => ({ skills: [], diagnostics: [] }), getPrompts: () => ({ prompts: [], diagnostics: [] }),
    getThemes: () => ({ themes: [], diagnostics: [] }), getAgentsFiles: () => ({ agentsFiles: [] }),
    getSystemPrompt: () => RESEARCH_PROMPT, getSystemPromptSource: () => undefined,
    getAppendSystemPrompt: () => [], getAppendSystemPromptSources: () => [],
    extendResources: () => {}, reload: async () => {},
  };
  return createAgentSession({ cwd: root, agentDir: runtimeDirectory, modelRuntime, model,
    thinkingLevel: "off", resourceLoader, tools: customTools.map(tool => tool.name), customTools,
    sessionManager: SessionManager.inMemory(root), settingsManager: SettingsManager.inMemory({
      compaction: { enabled: false }, retry: { enabled: false }, cacheWarming: { enabled: false } }) });
}

async function runLive(round, values) {
  const status = await pythonBridge(round, "status");
  if (status.error) throw new Error(status.error);
  if (status.stop_reason) throw new Error(`Round stopped: ${status.stop_reason}; no model call needed.`);
  const runtimeDirectory = path.join(root, "runs/physio-autoresearch/pi-agent");
  fs.mkdirSync(runtimeDirectory, { recursive: true });
  const modelRuntime = await ModelRuntime.create({ authPath: path.join(runtimeDirectory, "auth.json"),
    modelsPath: path.join(runtimeDirectory, "models.json"), modelsStorePath: path.join(runtimeDirectory, "model-store.json"),
    allowModelNetwork: false, refreshOnCreate: false });
  const model = modelRuntime.getModel(values.provider, values.model);
  if (!model || !modelRuntime.hasConfiguredAuth(values.provider)) throw new Error("Chosen Pi provider/model has no configured authorization. No model fallback or login was attempted.");
  const { session } = await createResearchSession({ round, model, modelRuntime, runtimeDirectory });
  const usagePath = path.join(round, "pi-usage.jsonl");
  const oldUsage = fs.existsSync(usagePath) ? fs.readFileSync(usagePath, "utf8").split("\n").filter(Boolean).map(JSON.parse) : [];
  let requestCount = oldUsage.filter(entry => entry.kind === "request").length;
  const settled = new Set(oldUsage.filter(entry => entry.kind === "usage").map(entry => entry.request));
  if (oldUsage.some(entry => entry.kind === "request" && !settled.has(entry.request))) {
    session.dispose();
    throw new Error("UNSETTLED_MODEL_REQUEST: inspect the interrupted call before resuming.");
  }
  const originalStream = session.agent.streamFunction;
  session.agent.streamFunction = async (...args) => {
    if (++requestCount > 20) throw new Error("MODEL_REQUEST_BUDGET_EXHAUSTED");
    const request = requestCount, began = Date.now();
    fs.appendFileSync(usagePath, JSON.stringify({ kind: "request", request, at: new Date().toISOString(), provider: values.provider, model: values.model }) + "\n");
    try {
      const stream = await originalStream(...args);
      void stream.result().then(message => {
        fs.appendFileSync(usagePath, JSON.stringify({ kind: "usage", request, at: new Date().toISOString(),
          elapsed_seconds: (Date.now() - began) / 1000, usage: message.usage }) + "\n");
      });
      return stream;
    } catch (error) {
      fs.appendFileSync(usagePath, JSON.stringify({ kind: "usage", request, at: new Date().toISOString(),
        elapsed_seconds: (Date.now() - began) / 1000, usage: {}, failed: true }) + "\n");
      throw error;
    }
  };
  const watchdog = setTimeout(() => { void session.abort(); }, Math.max(1, status.budget.max_seconds - status.elapsed_seconds) * 1000);
  try {
    const contract = JSON.parse(fs.readFileSync(path.join(round, "contract.json"), "utf8"));
    await session.prompt(`Run the bounded research round. Frozen features: ${JSON.stringify(contract.features.map(f => f.name))}. ${JSON.stringify(publicResult(status))}`);
    console.log(session.getLastAssistantText());
  } finally {
    clearTimeout(watchdog);
    session.dispose();
  }
}

async function main() {
  const { values } = parseArgs({ options: {
    round: { type: "string" }, provider: { type: "string" }, model: { type: "string" },
    check: { type: "boolean", default: false }, "allow-model-calls": { type: "boolean", default: false },
  } });
  if (!values.round) throw new Error("--round is required");
  const round = path.resolve(values.round);
  const status = await pythonBridge(round, "status");
  if (status.error) throw new Error(status.error);
  const entries = fs.existsSync(path.join(round, "experiments.jsonl"))
    ? fs.readFileSync(path.join(round, "experiments.jsonl"), "utf8").trim().split("\n").filter(Boolean).map(JSON.parse) : [];
  if (values.check) {
    console.log(JSON.stringify({ status: publicResult(status), upstream: reconstructResearch(entries),
      tools: researchTools(round).map(tool => tool.name), model_called: false }, null, 2));
    return;
  }
  if (!values["allow-model-calls"] || !values.provider || !values.model) {
    throw new Error("Live research requires explicit --allow-model-calls --provider NAME --model ID. Use --check offline.");
  }
  await withPiLock(round, () => runLive(round, values));
}

if (process.argv[1] && path.resolve(process.argv[1]) === fileURLToPath(import.meta.url)) {
  main().catch(error => { console.error(error.message); process.exitCode = 1; });
}
