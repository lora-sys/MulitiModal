import { spawn } from "node:child_process";
import { once } from "node:events";
import { python, root } from "./tools.mjs";

export async function withPiLock(round, work) {
  const helper = spawn(python, ["-m", "physio_autoresearch.pi_lock", round], {
    cwd: root, shell: false, windowsHide: true, stdio: ["pipe", "pipe", "pipe"] });
  let output = "";
  const exited = once(helper, "exit");
  try {
    await new Promise((resolve, reject) => {
      helper.once("error", reject);
      helper.once("exit", () => reject(new Error("PI_SESSION_BUSY_OR_INVALID_ROUND")));
      helper.stdout.on("data", chunk => {
        output += chunk.toString();
        if (output.startsWith("LOCKED")) resolve();
        else if (output.includes("\n")) reject(new Error("PI_SESSION_BUSY_OR_INVALID_ROUND"));
      });
    });
    return await work();
  } finally {
    helper.stdin.end();
    await exited;
  }
}
