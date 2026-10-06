/* fgo.wiki 英灵图鉴从者信息抓取脚本 —— 粘贴到浏览器 console 执行
 * 目标页面: https://fgo.wiki/w/英灵图鉴
 * 先把 #per-page 调到最大值再遍历 #cur-page-bottom 的全部 option，
 * 用 change 事件触发页面自带的 set_page() 重渲染 #list，按从者 id 去重（表格是纯客户端渲染，不发请求）。
 */
(async () => {
  const TABLE = document.getElementById("list");
  const PAGE_SEL = document.getElementById("cur-page-bottom");
  if (!TABLE || !PAGE_SEL) {
    console.error("未找到 #list 或 #cur-page-bottom，请在英灵图鉴页面执行本脚本");
    return;
  }

  // 表格模式（取消极简模式）+ 尽量放大每页条数，减少翻页次数
  const iconMode = document.getElementById("iconmode");
  if (iconMode && iconMode.checked) {
    iconMode.checked = false;
    apply_filters();
  }
  const perPage = document.getElementById("per-page");
  if (perPage) {
    perPage.value = perPage.options[perPage.options.length - 1].value;
    perPage.dispatchEvent(new Event("change"));
  }
  if (!TABLE.querySelector("tr.column-header")) {
    console.error("表格不是 10 列的桌面模式表头，请关闭极简模式后重试");
    return;
  }

  const lastPage = Number(PAGE_SEL.options[PAGE_SEL.options.length - 1].value);
  const byId = new Map();
  const tick = () => new Promise((r) => setTimeout(r, 30));

  const imgSrc = (img) => (img ? img.getAttribute("src") || img.src : "");
  const absUrl = (src) => (src ? new URL(src, location.href).href : null);
  // 经 URL 归一化后再解码，避免标题里的裸 % 触发 URIError
  const linkTarget = (a) => decodeURIComponent(new URL(a.getAttribute("href"), location.origin).pathname.split("/w/").pop());
  // 图标文件名即语义: Quick.png / 金卡Berserker.png
  const iconBase = (src) => decodeURIComponent(imgSrc(src).split("?")[0].split("/").pop().replace(/\.\w+$/, ""));
  // 属性以 & 连接，获取途径以 <br> 分行
  const splitValues = (cell, sep) => breakLines(cell).flatMap((s) => s.split(sep)).map((s) => s.trim()).filter(Boolean);
  const breakLines = (cell) => {
    const lines = [];
    let cur = "";
    const walk = (node) => {
      for (const child of node.childNodes) {
        if (child.nodeType === 3) cur += child.nodeValue;
        else if (child.nodeName === "BR") { lines.push(cur); cur = ""; }
        else walk(child);
      }
    };
    walk(cell);
    lines.push(cur);
    return lines.map((s) => s.trim()).filter(Boolean);
  };

  const parseRow = (tr) => {
    const td = [...tr.children];
    if (td.length < 10) return null;

    const nameLink = td[2].querySelector("a") || td[1].querySelector("a");
    const nameCell = td[2];
    const jpSpan = nameCell.querySelector('span[lang="ja"]');
    const enSpan = [...nameCell.querySelectorAll("span")].find((s) => s !== jpSpan);
    const classLink = td[4].querySelector("a");
    const classIcon = iconBase(td[4].querySelector("img"));
    const rarity = /^([金银铜]卡)/.exec(classIcon);

    return {
      id: Number(td[0].textContent.trim()),
      name_cn: (nameLink ? nameLink.textContent : nameCell.textContent).trim(),
      name_jp: (jpSpan ? jpSpan.textContent : "").trim(),
      name_en: (enSpan ? enSpan.textContent : "").trim(),
      page_url: nameLink ? new URL(nameLink.getAttribute("href"), location.origin).href : null,
      avatar: absUrl(imgSrc(td[1].querySelector("img"))),
      card_rarity: rarity ? rarity[1] : "",
      servant_class: classLink ? linkTarget(classLink) : classIcon.replace(/^[金银铜]卡/, ""),
      np_color: iconBase(td[3].querySelector("img")) || null,
      np_type: (td[3].querySelector("b") || {}).textContent?.trim() || null,
      cards: [...td[5].querySelectorAll("img")].map((i) => iconBase(i)),
      faction: splitValues(td[6], "&"),
      get: splitValues(td[7], "&"),
      atk_max: Number(td[8].textContent.trim()) || null,
      hp_max: Number(td[9].textContent.trim()) || null,
    };
  };

  for (let page = 1; page <= lastPage; page++) {
    PAGE_SEL.value = String(page);
    PAGE_SEL.dispatchEvent(new Event("change"));
    await tick();

    let added = 0;
    for (const tr of TABLE.querySelectorAll("tr")) {
      if (tr.classList.contains("column-header")) continue;
      const row = parseRow(tr);
      if (row && !byId.has(row.id)) {
        byId.set(row.id, row);
        added++;
      }
    }
    console.log(`第 ${page}/${lastPage} 页：新增 ${added} 条，累计 ${byId.size} 条`);
  }

  const servants = [...byId.values()].sort((a, b) => a.id - b.id);
  window.FGO_SERVANTS = servants;

  const download = (name, text, type) => {
    const a = document.createElement("a");
    a.href = URL.createObjectURL(new Blob([text], { type }));
    a.download = name;
    a.click();
    URL.revokeObjectURL(a.href);
  };
  const toCsvValue = (v) => `"${(Array.isArray(v) ? v.join("&") : String(v ?? "")).replace(/"/g, '""')}"`;
  const headers = Object.keys(servants[0] || {});
  const csv = [headers.join(",")]
    .concat(servants.map((s) => headers.map((h) => toCsvValue(s[h])).join(",")))
    .join("\n");

  download("servants.json", JSON.stringify(servants, null, 2), "application/json");
  download("servants.csv", "\ufeff" + csv, "text/csv;charset=utf-8");

  console.log(`完成：共 ${servants.length} 条，已下载 servants.json / servants.csv，可用 window.FGO_SERVANTS 访问`);
  console.table(servants.slice(0, 5));
})();
