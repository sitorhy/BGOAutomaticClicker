/* 抓取 fgo.wiki 从者详情页素材：卡面立绘 + 再临阶段图标 + 从者头像
 *
 * 用法（在仓库任意位置执行均可）:
 *   node scripts/fetch_servant_assets.mjs 1          仅编号 1
 *   node scripts/fetch_servant_assets.mjs 1 20       编号 1~20
 *   node scripts/fetch_servant_assets.mjs 1 20 --dry-run   只解析并打印，不下载
 *   node scripts/fetch_servant_assets.mjs 1 20 --force     覆盖已存在的文件
 *
 * 依赖 scripts/servants.json（由 scrape_fgo_servants.js 在浏览器 console 导出生成）。
 */
import fs from "node:fs";
import path from "node:path";

const API = "https://fgo.wiki/api.php";
const HEADERS = { "User-Agent": "avatar-classifier-py-server/1.0 (personal asset fetcher)" };
const HERE = import.meta.dirname;
const OUT_ROOT = path.join(HERE, "..", "res", "assets");

const args = process.argv.slice(2);
const flags = new Set(args.filter((a) => a.startsWith("--")));
const [startId, endId = startId] = args.filter((a) => !a.startsWith("--")).map(Number);
if (!startId) {
  console.error("用法: node scripts/fetch_servant_assets.mjs <起始编号> [结束编号] [--dry-run] [--force]");
  process.exit(1);
}

const servants = JSON.parse(fs.readFileSync(path.join(HERE, "servants.json"), "utf8"));
const targets = servants.filter((s) => s.id >= startId && s.id <= endId);
if (!targets.length) {
  console.error(`编号区间 ${startId}~${endId} 在 servants.json 中没有匹配记录`);
  process.exit(1);
}
console.log(`共 ${targets.length} 个从者，输出目录 ${path.resolve(OUT_ROOT)}`);

const get = async (url) => {
  const res = await fetch(url, { headers: HEADERS });
  if (!res.ok) throw new Error(`HTTP ${res.status} ${url}`);
  return res;
};

/* {{基础数值}} 里的 |立绘N= / |文件N= 成对出现，只保留 label 为「第N阶段」的再临立绘；
 * Beast 等无阶段划分的从者只有一张「默认」立绘，此时全部取用。 */
const parseCardFiles = (wiki) => {
  // console.log(wiki)
  const begin = wiki.indexOf("{{基础数值");
  if (begin < 0) return [];
  const body = wiki.slice(begin, wiki.indexOf("\n}}", begin));
  const labels = new Map([...body.matchAll(/^\|立绘(\d+)=(.*)$/gm)].map((m) => [m[1], m[2].trim()]));
  // console.log('labels', labels)
  const files = [...body.matchAll(/^\|文件(\d+)=(.*)$/gm)].map((m) => ({ n: m[1], file: m[2].trim() }));
  console.log('files', files)
  // const staged = files.filter((f) => /^第[1-9]阶段$/.test((labels.get(f.n) || "").normalize("NFKC")));
  // console.log('staged', staged)
  // return (staged.length ? staged : files).map((f) => f.file);
  return files.map((f) => f.file);
};

/* {{再临阶段图标}} 可能调用多次（基础 + 灵衣），图标写在 |图标= 或 |图标N= 参数里，
 * 多条用 ; 分隔（常用 ;;），也可换行；文件名允许含空格，所以按参数值切分而不是搜 .png。 */
const parseIconFiles = (wiki) => {
  const blocks = [...wiki.matchAll(/\{\{再临阶段图标[\s\S]*?^\}\}/gm)].map((m) => m[0]);
  const values = blocks.flatMap((b) => [...b.matchAll(/^\|(?:图标|图标\d+|文件\d*)=([\s\S]*?)(?=^\||^\}\})/gm)].map((m) => m[1]));
  const names = values
    .flatMap((v) => v.split(/[;\n]+/))
    .map((t) => /^\s*\[\[\s*(?:File|文件)\s*:\s*([\s\S]+?)(?:\||\]\])/i.exec(t)?.[1] ?? t)
    .map((t) => t.trim())
    .filter((t) => t && !/[{}=<]/.test(t));
  return [...new Set(names)];
};

const hasExt = (name) => /\.(?:png|jpe?g|gif)$/i.test(name);
/* 维基标题会把下划线归一化成空格，比对时统一掉；落盘用图片 URL 里的真实文件名 */
const baseOf = (name) => name.replace(/\.(?:png|jpe?g|gif)$/i, "").replace(/_/g, " ").toLowerCase();
const fileFromUrl = (url) => decodeURIComponent(new URL(url).pathname.split("/").pop());

/* 维基里的 |文件N= 大多不带扩展名，用候选扩展名一次性查询 imageinfo 拿原图地址 */
const resolveUrls = async (names) => {
  const titles = [...new Set(names.flatMap((n) => (hasExt(n) ? [n] : [`${n}.png`, `${n}.jpg`, `${n}.gif`])))];
  const urls = new Map();
  for (let i = 0; i < titles.length; i += 20) {
    const query = titles.slice(i, i + 20).map((t) => `文件:${t}`).join("|");
    const url = `${API}?action=query&format=json&prop=imageinfo&iiprop=url&redirects=1&titles=${encodeURIComponent(query)}`;
    const data = await (await get(url)).json();
    for (const page of Object.values(data.query?.pages || {})) {
      if (!page.imageinfo?.[0]?.url) continue;
      urls.set(baseOf(page.title.replace(/^文件:/, "")), page.imageinfo[0].url);
    }
  }
  return names.map((n) => {
    const url = urls.get(baseOf(n)) || null;
    return { name: n, url, file: url ? fileFromUrl(url) : n };
  });
};

const safeName = (name) => name.replace(/[<>:"/\\|?*]/g, "_");

const download = async (url, dest) => {
  if (fs.existsSync(dest) && !flags.has("--force")) return "已存在";
  const buf = Buffer.from(await (await get(url)).arrayBuffer());
  fs.writeFileSync(dest, buf);
  return `${(buf.length / 1024).toFixed(0)}KB`;
};

let failed = 0;
for (const s of targets) {
  const dir = path.join(OUT_ROOT, String(s.id));
  try {
    const wiki = await (await get(`${s.page_url}?action=raw`)).text();
    const cards = parseCardFiles(wiki);
    const icons = parseIconFiles(wiki);
    const avatarName = `Servant${String(s.id).padStart(3, "0")}.jpg`;
    const files = [...(await resolveUrls([...cards, ...icons])), { name: avatarName, url: s.avatar, file: avatarName }];
    const missing = files.filter((f) => !f.url).map((f) => f.name);
    console.log(
      `[${s.id}] ${s.name_cn}（${s.servant_class}） 卡面 ${cards.length} ${JSON.stringify(cards)} | 图标 ${icons.length} ${JSON.stringify(icons)}${missing.length ? ` | 未解析: ${missing.join(", ")}` : ""}`
    );
    if (flags.has("--dry-run")) continue;

    fs.mkdirSync(dir, { recursive: true });
    for (const f of files) {
      const result = f.url ? await download(f.url, path.join(dir, safeName(f.file))) : "跳过（无地址）";
      console.log(`      ${f.file} -> ${result}`);
    }
  } catch (e) {
    console.error(e);
    failed++;
    console.error(`[${s.id}] ${s.name_cn} 失败: ${e.message}`);
  }
  await new Promise((r) => setTimeout(r, 150));
}
console.log(failed ? `完成，${failed} 个从者失败` : "完成");
