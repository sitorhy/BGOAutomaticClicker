/* 抓取 fgo.wiki 从者详情页数值数据：基础数值 + 宝具 + 持有技能 + 职阶技能
 *
 * 用法（在仓库任意位置执行均可）:
 *   node scripts/fetch_servant_info.mjs 1          仅编号 1
 *   node scripts/fetch_servant_info.mjs 1 20       编号 1~20
 *   node scripts/fetch_servant_info.mjs 2 --print  打印 JSON 而不落盘
 *
 * 依赖 scripts/servants.json（由 scrape_fgo_servants.js 在浏览器 console 导出生成）。
 * 输出 res/info/<编号>.json，素材抓取见 fetch_servant_assets.mjs。
 *
 * 输出结构（一个宝具/技能是一个「单元」，单元下挂多个「变体」）：
 *   base[]           基础数值（满级ATK/HP、卡色hit数、np率、特性……），多形态从者会有多条
 *   np[] / skills[]  { 标签: 小节标题或tabber分支名, 变体: [{ 形态, 分支, 强化活动, 强化说明, 数据 }] }
 *     形态  当前 / 初始 / 强化 / 再强化（{{强化信息}} 的前后版本）
 *     分支  {{复合标签}} 的分支名（卡色不同或真名判明前后名称不同），无则 null
 *     数据  宝具为 名称/卡色/类型/阶级/种类/效果[{对象,效果,数值[]}]；技能为 图标/名称/日文名/冷却/效果
 *   class_skills[]   职阶技能（{ 标签, 形态, 名称, 日文名, 等级, 效果 }）
 *   extra_skill_ids[] 追加技能 id（效果文本不在 wikitext 里，由页面 JS 另外取）
 */
import fs from "node:fs";
import path from "node:path";

const HEADERS = { "User-Agent": "avatar-classifier-py-server/1.0 (personal data fetcher)" };
const HERE = import.meta.dirname;
const OUT_ROOT = path.join(HERE, "..", "res", "info");

const args = process.argv.slice(2);
const flags = new Set(args.filter((a) => a.startsWith("--")));
const [startId, endId = startId] = args.filter((a) => !a.startsWith("--")).map(Number);
if (!startId) {
  console.error("用法: node scripts/fetch_servant_info.mjs <起始编号> [结束编号] [--print]");
  process.exit(1);
}

const servants = JSON.parse(fs.readFileSync(path.join(HERE, "servants.json"), "utf8"));
const targets = servants.filter((s) => s.id >= startId && s.id <= endId);
if (!targets.length) {
  console.error(`编号区间 ${startId}~${endId} 在 servants.json 中没有匹配记录`);
  process.exit(1);
}

/* ---------- 维基模板解析（按大括号配对，支持嵌套与 [[链接]] 内的竖线） ---------- */

const closeOf = (src, i) => {
  let d = 1;
  let j = src[i + 2] === "{" ? i + 3 : i + 2;
  while (j < src.length) {
    if (src.startsWith("[[", j)) {
      const k = src.indexOf("]]", j);
      j = k < 0 ? src.length : k + 2;
    } else if (src.startsWith("{{", j)) {
      d++;
      j += src[j + 2] === "{" ? 3 : 2;
    } else if (src.startsWith("}}", j)) {
      if (!--d) return j + 2;
      j += 2;
    } else j++;
  }
  return src.length;
};

/* 在深度 0 处按 | 切分模板体，返回 [{text, start}]，start 是相对 src 的绝对偏移 */
const splitTop = (src, from, to) => {
  const parts = [];
  let d = 0;
  let start = from;
  for (let j = from; j < to; ) {
    if (src.startsWith("[[", j)) {
      const k = src.indexOf("]]", j);
      j = k < 0 ? to : k + 2;
    } else if (src.startsWith("{{", j)) {
      d++;
      j += src[j + 2] === "{" ? 3 : 2;
    } else if (src.startsWith("}}", j)) {
      d--;
      j += 2;
    } else if (src[j] === "|" && !d) {
      parts.push({ text: src.slice(start, j), start });
      start = j + 1;
      j++;
    } else j++;
  }
  parts.push({ text: src.slice(start, to), start });
  return parts;
};

const parseAt = (src, i) => {
  const end = closeOf(src, i);
  const bodyFrom = src[i + 2] === "{" ? i + 3 : i + 2;
  const parts = splitTop(src, bodyFrom, end - 2);
  const head = parts.shift();
  const tpl = {
    name: clean(head.text),
    args: parts.map((p) => {
      const value = clean(p.text);
      const m = /^([^[\]={}#|]*)=/.exec(value);
      return m && m[1].trim()
        ? {
            key: clean(m[1]),
            value: clean(value.slice(m[0].length)),
            raw: p.text.slice(p.text.indexOf("=") + 1),
            start: p.start,
            valueStart: src.indexOf("=", p.start) + 1,
          }
        : { value, raw: p.text, start: p.start };
    }),
    start: i,
    end,
    nested: parseTemplates(src, bodyFrom, end - 2),
  };
  return tpl;
};

const parseTemplates = (src, from = 0, to = src.length) => {
  const out = [];
  for (let i = from; i < to; ) {
    if (src.startsWith("{{", i)) {
      const t = parseAt(src, i);
      out.push(t);
      i = t.end;
    } else i++;
  }
  return out;
};

const clean = (s) =>
  s
    .replace(/<!--[\s\S]*?-->/g, "")
    .replace(/<ref[^>]*\/>|<ref[\s\S]*?<\/ref>|<\/?references[^>]*>/g, "")
    .replace(/\s+/g, " ")
    .trim();

/* ---------- 归一化 ---------- */

/* 维基用 ∅ 表示「这一等级/这一OC阶段没有数值」，统一成 null */
const num = (v) => (v === "∅" ? null : /^-?\d+(\.\d+)?$/.test(v) ? Number(v) : v);
/* 数值模板偶尔带后缀写法，如无法召唤从者的 {{基础数值(无法召唤)}} */
const kindOf = (name) => name.replace(/\(.*$/, "").trim();
/* {{特攻|X}} / {{特性|X}} 渲染成 〔X〕（保留数值语义）；
 * {{黑幕|X}} {{heimu|X}} {{示亡号|X}} {{修正|X|注释}} 只是排版或修订说明，取第一个参数 */
const render = (v) =>
  v
    .replace(/\{\{\s*(?:特攻|特性)\s*\|\s*([^|{}]+?)(?:\s*\|[^{}]*)?\}\}/g, "〔$1〕")
    .replace(/\{\{\s*(?:黑幕|heimu|示亡号|修正)\s*\|\s*([^|{}]+?)(?:\s*\|[^{}]*)?\}\}/g, "$1")
    .trim();

const paramMap = (t) =>
  Object.fromEntries(
    t.args
      .filter((a) => a.key)
      .map((a) => {
        const v = /分布/.test(a.key) ? unwiki(a.value) : num(unwiki(a.value));
        return [a.key, v === "" ? null : v];
      })
  );

/* 效果 + 数值：宝具写成 |效果X= / |数值X1..5=（可能带 |对象X=），技能写成 描述|10个等级值 */
const npEffects = (params) => {
  const out = [];
  for (const letter of "ABCDEFGH") {
    const text = params[`效果${letter}`];
    if (text === undefined) continue;
    const values = [];
    for (let n = 1; n <= 5; n++) {
      const v = params[`数值${letter}${n}`];
      if (v === undefined) continue;
      delete params[`数值${letter}${n}`];
      values.push(v);
    }
    while (values.length && values.at(-1) === null) values.pop();
    out.push({ 对象: params[`对象${letter}`] ?? null, 效果: text, 数值: values.length ? values : null });
    for (const k of [`效果${letter}`, `对象${letter}`]) delete params[k];
  }
  return out;
};

/* 冷却多写成数字，无法召唤从者的未知技能写成「充能时间：8」 */
const cooldown = (v) => {
  const m = /(\d+(?:\.\d+)?)\s*$/.exec(v.replace(/[。\.\s]+$/, ""));
  return m ? Number(m[1]) : v || null;
};
/* 效果无等级数值时用空串或 ∅ 占位（|∅|||||||||），去掉尾部空位 */
const levelValues = (args) => {
  const out = args.map((a) => (a.value === "" ? null : num(unwiki(a.value))));
  while (out.length && out[out.length - 1] === null) out.pop();
  return out.length ? out : null;
};

/* |技能信息= / |宝具信息= 的块写法：;; 效果行、:: 数值行、**对象##自身 目标行 */
const parseEffectBlock = (raw) => {
  const out = [];
  let cur = null;
  for (const line of raw.split(/\r?\n/)) {
    const l = line.trim();
    if (!l) continue;
    if (l.startsWith(";;")) {
      cur = { 对象: null, 效果: unwiki(l.slice(2)), 数值: null };
      out.push(cur);
    } else if (l.startsWith("::")) {
      const values = l.split("::").map((v) => v.trim()).filter(Boolean).map((v) => num(unwiki(v)));
      if (cur) cur.数值 = (cur.数值 || []).concat(values);
    } else if (l.startsWith("**")) {
      if (cur) {
        const [k, v] = l.slice(2).split("##");
        cur[unwiki(k)] = num(unwiki(v || "")) || null;
      }
    } else if (cur) {
      cur.效果 = unwiki(`${cur.效果} ${l}`);
    }
  }
  return out;
};

const nz = (v) => (v === "" ? null : v);

const skillEffects = (t) => {
  /* |用途=剧情限定 之类的命名参数可能出现在位置参数之前，先按原序分离 */
  const positional = t.args.filter((a) => !a.key);
  const 其他 = Object.fromEntries(
    t.args.filter((a) => a.key && a.key !== "技能信息").map((a) => [a.key, a.value === "" ? null : num(unwiki(a.value))])
  );
  const head = positional.slice(0, 4);
  const rest = positional.slice(4);
  const effects = [];
  for (let i = 0; i + 11 <= rest.length; i += 11) {
    effects.push({ 对象: null, 效果: unwiki(rest[i].value), 数值: levelValues(rest.slice(i + 1, i + 11)) });
  }
  const 信息 = t.args.find((a) => a.key === "技能信息");
  const 列表 = 信息 ? parseEffectBlock(信息.raw) : [];
  return {
    图标: head[0] ? nz(unwiki(head[0].value)) : null,
    名称: head[1] ? nz(unwiki(head[1].value)) : null,
    日文名: head[2] ? nz(unwiki(head[2].value)) : null,
    冷却: head[3] ? cooldown(head[3].value) : null,
    ...(Object.keys(其他).length ? { 其他 } : {}),
    效果: 列表.length ? 列表 : rest.length % 11 === 0 ? effects : rest.map((a) => num(unwiki(a.value))),
  };
};

/* 文档序展开模板树，chain 是祖先节点（用于判定某个块是否写在 {{强化信息}} 里） */
const flatten = (list, chain = []) => list.flatMap((t) => [{ t, chain }, ...flatten(t.nested, [...chain, t])]);

const isKind = (t, ...kinds) => kinds.includes(kindOf(t.name));

const shape = (t) => {
  if (!isKind(t, "宝具")) return skillEffects(t);
  const params = paramMap(t);
  /* 少数从者的宝具效果写成 |宝具信息= 块而不是 |效果X=/|数值X= */
  const 信息 = t.args.find((a) => a.key === "宝具信息");
  const 效果 = 信息 ? parseEffectBlock(信息.raw) : npEffects(params);
  delete params.宝具信息;
  const { 中文名, ...rest } = params;
  return { 名称: 中文名 ?? null, ...rest, 效果 };
};

/* 模板前最近的 '''粗体小标题''' 或 <tabber> 分支名，作为形态/槽位标签。
 * 分支名行可能带同行内容，如「灵基解放后=通关「…」后灵基解放。」；
 * 窗口起点会把行中间（如 |宝具卡hit数=5 被切成 it数=5）当成行首，得排除掉。 */
const labelBefore = (src, pos) => {
  const from = Math.max(0, pos - 400);
  const win = src.slice(from, pos);
  const bold = [...win.matchAll(/'''([^'\n]{1,60})'''/g)].pop();
  const tab = [...win.matchAll(/^[ \t]*([^|\n{}<>#=]{2,30})=/gm)]
    .filter((m) => m.index > 0 || from === 0 || src[from - 1] === "\n")
    .pop();
  const pick = (bold?.index ?? -1) > (tab?.index ?? -1) ? bold : tab;
  return pick ? pick[1].trim() : null;
};

/* [[链接|显示]] / [[页面]] 取显示文本；<br> 与 ''斜体'' 只是排版，数值说明里不需要。
 * 链接要先展开，{{示亡号|[[声优一览#X|X]]}} 这类包装参数的值里带竖线。
 * 效果文本里的 <Over Charge时效果提升> 是维基的语义标记，不能按 HTML 标签删掉，
 * 所以只剥 div/span 这类纯排版标签。 */
const unwiki = (v) =>
  render(
    clean(v)
      .replace(/\{\{=\}\}/g, "=")
      .replace(/\[\[[^[\]|]*\|([^\]]*)\]\]/g, "$1")
      .replace(/\[\[([^\]]*)\]\]/g, "$1")
  )
    .replace(/<\/?(?:div|span|sup|sub|center|u|b|i|small)\b[^>]*>/g, "")
    .replace(/<br\s*\/?>/g, " ")
    .replace(/''/g, "")
    .replace(/\s+/g, " ")
    .trim();

/* {{强化信息}}|初始内容=块A |强化活动=… |强化内容=说明+块B（部分从者还有再强化） */
const ENHANCE_SLOTS = { 初始内容: "初始", 强化内容: "强化", 再强化内容: "再强化" };
const slotRanges = (enh) => {
  const args = Object.keys(ENHANCE_SLOTS)
    .map((k) => enh.args.find((a) => a.key === k))
    .filter(Boolean);
  return args.map((arg, i) => ({
    形态: ENHANCE_SLOTS[arg.key],
    arg,
    from: arg.valueStart,
    to: i + 1 < args.length ? args[i + 1].start : enh.end - 2,
  }));
};

const enhanceOf = (chain) => [...chain].reverse().find((a) => isKind(a, "强化信息"));
const formOf = (t, chain) => {
  const enh = enhanceOf(chain);
  if (!enh) return "当前";
  return (slotRanges(enh).find((r) => t.start >= r.from && t.start < r.to) || {}).形态 || "强化";
};

/* {{复合标签|分支1|内容1|分支2|内容2}} 是同一单元的多个版本
 * （卡色不同，如卫宫 Buster/Arts；或真名判明前后名称不同），分支名取内容前一个参数 */
const branchOf = (t, chain) => {
  const sw = [...chain].reverse().find((a) => kindOf(a.name) === "复合标签");
  if (!sw) return null;
  for (let i = 1; i < sw.args.length; i += 2) {
    const to = i + 1 < sw.args.length ? sw.args[i + 1].start : sw.end - 2;
    if (t.start >= sw.args[i].start && t.start < to) return unwiki(sw.args[i - 1].value) || null;
  }
  return null;
};

/* 强化说明到第一个 {{复合标签}} 为止，分支标签另存 */
const noteEnd = (src, from, kidStart) => {
  const m = /\{\{\s*复合标签/.exec(src.slice(from, kidStart));
  return m ? from + m.index : kidStart;
};

/* 一个 {{强化信息}} 里的前后两块合成同一单元的多个形态 */
const variantsOf = (enh, src, entries) => {
  const out = [];
  for (const r of slotRanges(enh)) {
    const kids = entries.filter((e) => isKind(e.t, "宝具", "持有技能") && e.t.start >= r.from && e.t.start < r.to);
    if (!kids.length) continue;
    const 活动 = enh.args.find((a) => a.key === (r.形态 === "再强化" ? "再强化活动" : "强化活动"));
    const 说明 = r.形态 === "初始" ? null : unwiki(src.slice(r.from, noteEnd(src, r.from, kids[0].t.start))) || null;
    kids.forEach((k) =>
      out.push({
        形态: r.形态,
        分支: branchOf(k.t, k.chain),
        强化活动: 活动 ? unwiki(活动.value) || null : null,
        强化说明: 说明,
        数据: shape(k.t),
      })
    );
  }
  return out;
};

/* ---------- 单个从者 ---------- */

const build = (s, wiki) => {
  const nodes = flatten(parseTemplates(wiki));
  const units = [];
  for (const { t, chain } of nodes) {
    if (chain.some((a) => isKind(a, "强化信息"))) continue;
    if (isKind(t, "强化信息")) {
      const 变体 = variantsOf(t, wiki, flatten(t.nested));
      if (!变体.length) continue;
      /* 宝具块有卡色，技能块有图标，据此分单元类型 */
      units.push({ 类型: 变体[0].数据.卡色 !== undefined ? "宝具" : "持有技能", 标签: labelBefore(wiki, t.start), 变体 });
    } else if (isKind(t, "宝具", "持有技能")) {
      units.push({
        类型: kindOf(t.name),
        标签: labelBefore(wiki, t.start),
        变体: [{ 形态: "当前", 分支: branchOf(t, chain), 强化活动: null, 强化说明: null, 数据: shape(t) }],
      });
    }
  }
  const np = units.filter((u) => u.类型 === "宝具");
  const skills = units.filter((u) => u.类型 === "持有技能");

  const classSkills = nodes
    .filter(({ t }) => isKind(t, "职阶技能"))
    .flatMap(({ t, chain }) => {
      const out = [];
      for (let i = 0; i + 4 <= t.args.length; i += 4) {
        out.push({
          标签: labelBefore(wiki, t.start),
          形态: formOf(t, chain),
          名称: unwiki(t.args[i].value),
          日文名: unwiki(t.args[i + 1].value),
          等级: num(t.args[i + 2].value),
          效果: unwiki(t.args[i + 3].value),
        });
      }
      return out;
    });

  const extraIds = nodes.filter(({ t }) => isKind(t, "追加技能")).flatMap(({ t }) => t.args.map((a) => num(unwiki(a.value))));

  const base = nodes
    .filter(({ t }) => isKind(t, "基础数值"))
    .map(({ t }) => {
      const stats = paramMap(t);
      for (const k of Object.keys(stats)) if (/^(立绘|文件)\d*$/.test(k)) delete stats[k];
      return { 形态: labelBefore(wiki, t.start), ...stats };
    });

  return {
    id: s.id,
    name_cn: s.name_cn,
    name_jp: s.name_jp,
    name_en: s.name_en,
    servant_class: s.servant_class,
    page_url: s.page_url,
    base,
    np,
    skills,
    class_skills: classSkills,
    extra_skill_ids: extraIds,
  };
};

const get = async (url) => {
  const res = await fetch(url, { headers: HEADERS });
  if (!res.ok) throw new Error(`HTTP ${res.status} ${url}`);
  return res;
};

fs.mkdirSync(OUT_ROOT, { recursive: true });
let failed = 0;
for (const s of targets) {
  try {
    const wiki = await (await get(`${s.page_url}?action=raw`)).text();
    const info = build(s, wiki);
    const summary = `基础 ${(info.base[0]?.满级ATK ?? "-")}/${info.base[0]?.满级HP ?? "-"} ATK/HP | 宝具 ${info.np.length} | 技能 ${info.skills.length} | 职阶技能 ${info.class_skills.length} | 追加 ${info.extra_skill_ids.length}`;
    console.log(`[${s.id}] ${s.name_cn}（${s.servant_class}） ${summary}`);
    if (flags.has("--print")) console.log(JSON.stringify(info, null, 2));
    else fs.writeFileSync(path.join(OUT_ROOT, `${s.id}.json`), JSON.stringify(info, null, 2) + "\n");
  } catch (e) {
    failed++;
    console.error(`[${s.id}] ${s.name_cn} 失败: ${e.message}`);
  }
  await new Promise((r) => setTimeout(r, 150));
}
console.log(failed ? `完成，${failed} 个从者失败` : `完成${flags.has("--print") ? "" : `，输出目录 ${path.resolve(OUT_ROOT)}`}`);
