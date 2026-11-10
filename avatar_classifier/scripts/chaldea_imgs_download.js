import fs from "node:fs";
import path from "node:path";

const HERE = import.meta.dirname;
const CHALDEA_DIR = path.join(HERE, "..", "res", "chaldea");
const OUT_ROOT = path.join(HERE, "..", "res", "assets");
const STATUS_FILE = path.join(HERE, "download_status.json");
const ASSET_TYPES = ["charaGraph", "faces", "narrowFigure", "commands", "status"];

const args = process.argv.slice(2);
const flags = new Set(args.filter((a) => a.startsWith("--")));
const positional = args.filter((a) => !a.startsWith("--")).map(Number);
const [startId, endId = startId] = positional;

/**
 * node scripts/chaldea_imgs_download.js 1 50          # 下载 collectionNo 1~50
 * node scripts/chaldea_imgs_download.js 3             # 仅下载 collectionNo 3
 * node scripts/chaldea_imgs_download.js 1 10 --dry-run # 只打印不下载
 * node scripts/chaldea_imgs_download.js 1 10 --force   # 覆盖已存在文件
 */

if (!startId || isNaN(startId)) {
  console.error("用法: node scripts/chaldea_imgs_download.js <起始collectionNo> [结束collectionNo] [--dry-run] [--force]");
  console.error("示例: node scripts/chaldea_imgs_download.js 1 50");
  console.error("  --force  强制重新下载并重置状态");
  process.exit(1);
}

function loadStatus() {
  if (!fs.existsSync(STATUS_FILE)) return {};
  try {
    return JSON.parse(fs.readFileSync(STATUS_FILE, "utf8"));
  } catch {
    return {};
  }
}

function saveStatus(status) {
  fs.writeFileSync(STATUS_FILE, JSON.stringify(status, null, 2));
}

function loadServants() {
  const files = fs.readdirSync(CHALDEA_DIR).filter((f) => f.startsWith("servants.") && f.endsWith(".json"));
  const all = [];
  for (const f of files) {
    const data = JSON.parse(fs.readFileSync(path.join(CHALDEA_DIR, f), "utf8"));
    all.push(...data);
  }
  return all;
}

async function download(url, dest) {
  if (fs.existsSync(dest) && !flags.has("--force")) return "skipped";
  const res = await fetch(url);
  if (!res.ok) throw new Error(`HTTP ${res.status} ${url}`);
  const buf = Buffer.from(await res.arrayBuffer());
  fs.mkdirSync(path.dirname(dest), { recursive: true });
  fs.writeFileSync(dest, buf);
  return `${(buf.length / 1024).toFixed(0)}KB`;
}

function extractUrls(servant) {
  const tasks = [];
  const collectionNo = servant.collectionNo;

  for (const type of ASSET_TYPES) {
    const root = servant.extraAssets?.[type];
    if (!root) continue;

    for (const [category, entries] of Object.entries(root)) {
      if (!entries || typeof entries !== "object") continue;

      for (const [key, url] of Object.entries(entries)) {
        if (!url) continue;
        const filename = url.split("/").pop();
        const destDir = path.join(OUT_ROOT, String(collectionNo), type, category);
        tasks.push({ url, dest: path.join(destDir, filename) });
      }
    }
  }

  return tasks;
}

async function main() {
  const servants = loadServants();
  const status = loadStatus();
  const force = flags.has("--force");
  const targets = servants.filter((s) => s.collectionNo >= startId && s.collectionNo <= endId);

  if (!targets.length) {
    console.error(`collectionNo ${startId}~${endId} 无匹配从者`);
    process.exit(1);
  }

  const skipped = targets.filter((s) => !force && status[String(s.collectionNo)] === 1);
  const toDownload = force ? targets : targets.filter((s) => status[String(s.collectionNo)] !== 1);

  console.log(`匹配到 ${targets.length} 个从者 (collectionNo ${startId}~${endId})`);
  if (skipped.length) console.log(`跳过已下载: ${skipped.length} 个`);
  console.log(`待处理: ${toDownload.length} 个`);
  console.log(`输出目录: ${path.resolve(OUT_ROOT)}`);
  console.log();

  let totalFiles = 0;
  let failed = 0;
  let successCount = 0;

  for (const s of toDownload) {
    const key = String(s.collectionNo);
    const tasks = extractUrls(s);
    console.log(`[${s.collectionNo}] ${s.name} — ${tasks.length} 个文件`);
    totalFiles += tasks.length;

    if (flags.has("--dry-run")) {
      for (const t of tasks) console.log(`  ${t.url}`);
      continue;
    }

    let servantFailed = false;
    for (const t of tasks) {
      try {
        const result = await download(t.url, t.dest);
        if (result !== "skipped") {
          console.log(`  ${path.basename(t.dest)} -> ${result}`);
        }
      } catch (e) {
        servantFailed = true;
        failed++;
        console.error(`  ${t.url} 失败: ${e.message}`);
      }
    }

    status[key] = servantFailed ? 0 : 1;
    saveStatus(status);
    if (!servantFailed) successCount++;

    await new Promise((r) => setTimeout(r, 100));
  }

  console.log(`\n完成: ${toDownload.length} 个从者 (成功 ${successCount}), ${totalFiles} 个文件, ${failed} 个失败`);
}

main();
