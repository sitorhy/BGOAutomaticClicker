import {GoogleGenerativeAI} from "@google/generative-ai"
import {ProxyAgent, setGlobalDispatcher} from "undici";
import Config from "./config.ts";

if (!Config.apiKey || !Config.proxyUrl) {
    throw new Error("请配置apiKey和网络代理");
}

const proxyUrl = Config.proxyUrl;
const dispatcher = new ProxyAgent({uri: new URL(proxyUrl).toString()});
//全局fetch调用启用代理
setGlobalDispatcher(dispatcher);

// 1. 初始化 API
const genAI = new GoogleGenerativeAI(Config.apiKey);

async function runGeminiDecision(mockData: Record<string, any>) {
    // 2. 配置模型
    // 可选模型：
    // https://ai.google.dev/gemini-api/docs/models?hl=zh-cn
    const model = genAI.getGenerativeModel({
        model: "gemini-2.5-flash",
        // 通过设置 systemInstruction 避免每次重复发送
        systemInstruction: "",
        generationConfig: { responseMimeType: "application/json" }
    });

    // 3. 构造 Prompt
    const prompt = `
    数据：${JSON.stringify(mockData)}。请给出 a,b 的和。
    `;

    try {
        const result = await model.generateContent(prompt);
        const response = result.response;
        const text = response.text();

        // 解析 AI 返回的 JSON 指令
        console.log("AI 计算结果:", text);
        return JSON.parse(text);
    } catch (error) {
        console.error("Gemini 请求失败:", error);
    }
}

const mockData = {
    a: 100,
    b: 100
};
runGeminiDecision(mockData).then(() => {
});