import type { RetrieveResponse, SourceType, TopKItem } from "./types";
import { clamp01, uid } from "./utils";

const API_BASE = import.meta.env.VITE_API_BASE || "";
const USE_MOCK = String(import.meta.env.VITE_USE_MOCK || "false") === "true";

function mockTopK(): TopKItem[] {
  const arr: TopKItem[] = [];
  const base = 0.55 + Math.random() * 0.25;
  for (let i = 1; i <= 50; i++) {
    const cosine = clamp01(base + (Math.random() - 0.5) * 0.15);
    const score = Math.round(60 + cosine * 40 * 10) / 10;
    arr.push({
      rank: i,
      // 用随机占位图（演示用），你接后端后会换成真实 url
      image_url: `https://picsum.photos/seed/${uid("img")}/480/320`,
      caption: `Mock caption #${i}`,
      cosine,
      score
    });
  }
  arr.sort((a, b) => b.cosine - a.cosine);
  arr.forEach((x, idx) => (x.rank = idx + 1));
  return arr;
}

export async function retrieve(source: SourceType, file: File): Promise<RetrieveResponse> {
  if (USE_MOCK) {
    const isCat = Math.random() > 0.5;
    const probs =
      source === "catdog"
        ? [
            { label: "cat", prob: isCat ? 0.9 : 0.1 },
            { label: "dog", prob: isCat ? 0.1 : 0.9 }
          ]
        : [
            { label: "car", prob: 0.18 },
            { label: "street", prob: 0.12 },
            { label: "mountain", prob: 0.10 },
            { label: "cat", prob: 0.09 },
            { label: "dog", prob: 0.07 }
          ].map((x) => ({ ...x, prob: clamp01(x.prob + (Math.random() - 0.5) * 0.05) }));

    return {
      source,
      predicted_label: source === "catdog" ? (isCat ? "cat" : "dog") : "car",
      probs,
      gallery_size: 15000,
      elapsed_ms: 180 + Math.round(Math.random() * 120),
      request_id: uid("req"),
      topk: mockTopK()
    };
  }

  const fd = new FormData();
  fd.append("file", file);
  fd.append("source", source);

  const res = await fetch(`${API_BASE}/api/retrieve`, {
    method: "POST",
    body: fd
  });

  if (!res.ok) {
    const text = await res.text().catch(() => "");
    throw new Error(`HTTP ${res.status}: ${text || "retrieve failed"}`);
  }
  return (await res.json()) as RetrieveResponse;
}
