import React, { useEffect, useMemo, useRef, useState } from "react";
import {
  BarChart,
  Bar,
  XAxis,
  YAxis,
  Tooltip,
  ResponsiveContainer,
  AreaChart,
  Area,
  Cell
} from "recharts";
import {
  Upload,
  Sparkles,
  History as HistoryIcon,
  Search,
  LayoutGrid,
  Download,
  Star,
  StarOff,
  Sun,
  Moon,
  GitCompare,
  X,
  Timer,
  Database,
  Tag,
  Zap,
  ChevronRight
} from "lucide-react";

import type { HistoryRecord, RetrieveResponse, SourceType, TopKItem } from "./types";
import { retrieve } from "./api";
import { cn, downloadText, formatNum, readFileAsDataURL, toCSV, uid } from "./utils";

const LS_KEY = "vitweb_history_v1";
const LS_THEME = "vitweb_theme_v1";

// --- WHIC: keyword extraction helpers (front-end only) ---
const STOPWORDS = new Set([
  "a","an","the","and","or","to","of","in","on","at","for","with","from","by","as","is","are","was","were",
  "this","that","these","those","it","its","their","his","her","your","my","our","we","you","they",
  "image","photo","picture","view","shot","scene","one","two","three",
  "person","people","man","woman","boy","girl","someone","anyone"
]);

// 英文->中文（可按需要继续补充）
const CN_MAP: Record<string, string> = {
  landscape: "风景",
  scenery: "风景",
  mountain: "山",
  mountains: "群山",
  lake: "湖",
  river: "河流",
  ocean: "海洋",
  sea: "海",
  forest: "森林",
  trees: "树林",
  sky: "天空",
  clouds: "云",
  snow: "雪",
  beach: "海滩",
  sunrise: "日出",
  sunset: "日落",
  aerial: "航拍",
  valley: "山谷",
  waterfall: "瀑布",
  coast: "海岸",
  shore: "岸边",
  city: "城市",
  street: "街道",
  building: "建筑",
  road: "道路",
  bridge: "桥",
  car: "汽车",
  dog: "狗",
  cat: "猫"
};

const LANDSCAPE_WORDS = new Set([
  "landscape","scenery","mountain","mountains","lake","river","ocean","sea","forest","trees",
  "sky","clouds","snow","beach","sunrise","sunset","valley","waterfall","coast","shore","meadow","field","glacier"
]);
const PEOPLE_WORDS = new Set(["person","people","man","woman","boy","girl","portrait","face","selfie"]);
const CITY_WORDS = new Set(["city","street","building","buildings","bridge","downtown","traffic","road","skyline"]);
const ANIMAL_WORDS = new Set(["dog","cat","bird","horse","cow","sheep","wildlife","animal"]);

const WHIC_BAR_COLORS = [
  "rgba(99,102,241,0.95)",  // indigo
  "rgba(34,197,94,0.95)",   // green
  "rgba(236,72,153,0.95)",  // pink
  "rgba(56,189,248,0.95)",  // sky
  "rgba(168,85,247,0.95)",  // purple
  "rgba(251,191,36,0.95)"   // amber
];

function loadHistory(): HistoryRecord[] {
  try {
    const raw = localStorage.getItem(LS_KEY);
    if (!raw) return [];
    const arr = JSON.parse(raw);
    if (!Array.isArray(arr)) return [];
    return arr as HistoryRecord[];
  } catch {
    return [];
  }
}
function saveHistory(arr: HistoryRecord[]) {
  localStorage.setItem(LS_KEY, JSON.stringify(arr.slice(0, 50)));
}

function Badge({ children, className }: { children: React.ReactNode; className?: string }) {
  return (
    <span className={cn("inline-flex items-center rounded-full px-3 py-1 text-xs font-semibold", className)}>
      {children}
    </span>
  );
}

function Card({ className, children }: { className?: string; children: React.ReactNode }) {
  return (
    <div
      className={cn(
        "rounded-2xl bg-white/5 backdrop-blur border border-white/10",
        "shadow-[0_0_0_1px_rgba(255,255,255,0.06)]",
        className
      )}
    >
      {children}
    </div>
  );
}

function Button({
  children,
  className,
  onClick,
  disabled,
  type = "button"
}: {
  children: React.ReactNode;
  className?: string;
  onClick?: () => void;
  disabled?: boolean;
  type?: "button" | "submit";
}) {
  return (
    <button
      type={type}
      disabled={disabled}
      onClick={onClick}
      className={cn(
        "inline-flex items-center gap-2 rounded-xl px-4 py-2 text-sm font-semibold transition",
        "bg-indigo-500 hover:bg-indigo-400 disabled:bg-white/10 disabled:text-white/40",
        "focus:outline-none focus:ring-2 focus:ring-indigo-400/60",
        className
      )}
    >
      {children}
    </button>
  );
}

function GhostButton({
  children,
  className,
  onClick,
  disabled
}: {
  children: React.ReactNode;
  className?: string;
  onClick?: () => void;
  disabled?: boolean;
}) {
  return (
    <button
      disabled={disabled}
      onClick={onClick}
      className={cn(
        "inline-flex items-center gap-2 rounded-xl px-4 py-2 text-sm font-semibold transition",
        "bg-white/5 hover:bg-white/10 border border-white/10 disabled:opacity-40",
        "focus:outline-none focus:ring-2 focus:ring-white/20",
        className
      )}
    >
      {children}
    </button>
  );
}

function Modal({
  open,
  title,
  onClose,
  children
}: {
  open: boolean;
  title: string;
  onClose: () => void;
  children: React.ReactNode;
}) {
  if (!open) return null;
  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center p-4">
      <div className="absolute inset-0 bg-black/60" onClick={onClose} />
      <div className="relative w-full max-w-5xl">
        <Card className="p-4">
          <div className="flex items-center justify-between gap-2 px-2">
            <div className="text-lg font-extrabold">{title}</div>
            <button
              onClick={onClose}
              className="rounded-lg p-2 hover:bg-white/10 border border-white/10"
              aria-label="close"
            >
              <X className="h-5 w-5" />
            </button>
          </div>
          <div className="p-2">{children}</div>
        </Card>
      </div>
    </div>
  );
}

/** KPI ring */
function Ring({ value, label }: { value: number; label: string }) {
  const v = Math.max(0, Math.min(1, value));
  const r = 18;
  const c = 2 * Math.PI * r;
  const dash = c * v;
  return (
    <div className="flex items-center gap-3">
      <div className="relative h-12 w-12">
        <svg className="h-12 w-12 -rotate-90" viewBox="0 0 48 48">
          <circle cx="24" cy="24" r={r} stroke="rgba(255,255,255,0.12)" strokeWidth="4" fill="none" />
          <circle
            cx="24"
            cy="24"
            r={r}
            stroke="rgba(99,102,241,0.95)"
            strokeWidth="4"
            fill="none"
            strokeDasharray={`${dash} ${c - dash}`}
            strokeLinecap="round"
          />
        </svg>
        <div className="absolute inset-0 grid place-items-center text-[10px] font-black">
          {Math.round(v * 100)}
        </div>
      </div>
      <div>
        <div className="text-xs opacity-70">{label}</div>
        <div className="text-sm font-extrabold">{Math.round(v * 100)}%</div>
      </div>
    </div>
  );
}

function KpiTile({
  icon,
  title,
  value,
  sub,
  tone = "indigo",
  right
}: {
  icon: React.ReactNode;
  title: string;
  value: string;
  sub?: string;
  tone?: "indigo" | "emerald" | "fuchsia";
  right?: React.ReactNode;
}) {
  const toneCls =
    tone === "indigo"
      ? "from-indigo-500/30 to-indigo-500/5 border-indigo-400/20"
      : tone === "emerald"
      ? "from-emerald-500/30 to-emerald-500/5 border-emerald-400/20"
      : "from-fuchsia-500/30 to-fuchsia-500/5 border-fuchsia-400/20";

  return (
    <div className={cn("relative overflow-hidden rounded-2xl border p-4 bg-gradient-to-br", toneCls)}>
      <div className="absolute -top-10 -right-10 h-28 w-28 rounded-full bg-white/10 blur-2xl" />
      <div className="flex items-start justify-between gap-3">
        <div className="flex items-start gap-3">
          <div className="rounded-xl bg-white/10 border border-white/10 p-2">{icon}</div>
          <div>
            <div className="text-xs font-bold opacity-80">{title}</div>
            <div className="mt-1 text-2xl font-black tracking-tight">{value}</div>
            {sub && <div className="mt-1 text-xs opacity-70">{sub}</div>}
          </div>
        </div>
        {right}
      </div>
    </div>
  );
}

function SkeletonCard({ dense }: { dense: boolean }) {
  return (
    <div className="rounded-2xl border border-white/10 bg-white/5 overflow-hidden animate-pulse">
      <div className={cn("w-full bg-white/10", dense ? "h-28" : "h-44")} />
      <div className="p-3 space-y-2">
        <div className="h-3 w-4/5 bg-white/10 rounded" />
        <div className="h-3 w-2/3 bg-white/10 rounded" />
        <div className="h-3 w-1/2 bg-white/10 rounded" />
      </div>
    </div>
  );
}

/** History: compact timeline */
function Timeline({ items, onClear }: { items: HistoryRecord[]; onClear: () => void }) {
  return (
    <Card className="p-5">
      <div className="flex items-center justify-between gap-2">
        <div className="flex items-center gap-2">
          <HistoryIcon className="h-5 w-5 text-fuchsia-300" />
          <div className="text-lg font-extrabold">History</div>
          <Badge className="bg-white/10 border border-white/10 text-white/80">{items.length}</Badge>
        </div>
        <GhostButton onClick={onClear}>清空</GhostButton>
      </div>

      <div className="mt-5">
        {items.length === 0 && <div className="text-sm opacity-70">暂无记录</div>}

        <div className="relative">
          <div className="absolute left-3 top-0 bottom-0 w-px bg-white/10" />
          <div className="space-y-4">
            {items.map((h) => (
              <div key={h.id} className="relative pl-10">
                <div
                  className={cn(
                    "absolute left-[10px] top-5 h-3 w-3 rounded-full border",
                    h.source === "catdog"
                      ? "bg-indigo-400 border-indigo-300/40"
                      : "bg-emerald-400 border-emerald-300/40"
                  )}
                />
                <div className="rounded-2xl border border-white/10 bg-white/5 hover:bg-white/10 transition p-4">
                  <div className="flex items-center justify-between gap-2">
                    <Badge
                      className={cn(
                        "border",
                        h.source === "catdog"
                          ? "bg-indigo-500/20 border-indigo-400/20 text-indigo-200"
                          : "bg-emerald-500/20 border-emerald-400/20 text-emerald-200"
                      )}
                    >
                      {h.source.toUpperCase()}
                    </Badge>
                    <div className="text-xs opacity-70">{new Date(h.created_at).toLocaleString()}</div>
                  </div>

                  <div className="mt-3 flex items-center gap-3">
                    <img
                      src={h.query_preview || ""}
                      alt="query"
                      className="h-14 w-14 rounded-xl border border-white/10 object-cover bg-black/30"
                    />
                    <div className="min-w-0 flex-1">
                      <div className="font-extrabold truncate">{h.query_name}</div>
                      <div className="text-sm opacity-80">
                        <span className="font-bold">{h.predicted_label}</span>
                        {h.top1 && (
                          <span className="ml-3 text-xs opacity-70">top1 {formatNum(h.top1.cosine, 4)}</span>
                        )}
                      </div>
                      {h.request_id && <div className="text-xs opacity-60 truncate">id: {h.request_id}</div>}
                    </div>

                    {h.top1 && (
                      <div className="hidden md:flex items-center gap-2 rounded-xl bg-black/20 border border-white/10 p-2">
                        <img
                          src={h.top1.image_url}
                          className="h-12 w-16 rounded-lg border border-white/10 object-cover bg-black/30"
                          alt="top1"
                          onError={(e) => ((e.currentTarget as HTMLImageElement).style.display = "none")}
                        />
                        <ChevronRight className="h-4 w-4 opacity-60" />
                      </div>
                    )}
                  </div>
                </div>
              </div>
            ))}
          </div>
        </div>
      </div>
    </Card>
  );
}

/** Similarity band grouping (no backend change) */
type Band = {
  key: string;
  title: string;
  hint: string;
  min: number;
  max: number;
  headBg: string;
  headBorder: string;
  badgeBg: string;
  badgeBorder: string;
  bar: string;
};

const BANDS: Band[] = [
  {
    key: "A",
    title: "Strong",
    hint: "0.60+",
    min: 0.6,
    max: 1.01,
    headBg: "bg-emerald-500/10",
    headBorder: "border-emerald-400/20",
    badgeBg: "bg-emerald-500/20",
    badgeBorder: "border-emerald-400/20",
    bar: "linear-gradient(90deg, rgba(34,197,94,0.95), rgba(16,185,129,0.7))"
  },
  {
    key: "B",
    title: "Good",
    hint: "0.45–0.60",
    min: 0.45,
    max: 0.6,
    headBg: "bg-indigo-500/10",
    headBorder: "border-indigo-400/20",
    badgeBg: "bg-indigo-500/20",
    badgeBorder: "border-indigo-400/20",
    bar: "linear-gradient(90deg, rgba(99,102,241,0.95), rgba(168,85,247,0.65))"
  },
  {
    key: "C",
    title: "Ok",
    hint: "0.30–0.45",
    min: 0.3,
    max: 0.45,
    headBg: "bg-fuchsia-500/10",
    headBorder: "border-fuchsia-400/20",
    badgeBg: "bg-fuchsia-500/20",
    badgeBorder: "border-fuchsia-400/20",
    bar: "linear-gradient(90deg, rgba(236,72,153,0.95), rgba(168,85,247,0.55))"
  },
  {
    key: "D",
    title: "Weak",
    hint: "<0.30",
    min: -1,
    max: 0.3,
    headBg: "bg-white/5",
    headBorder: "border-white/10",
    badgeBg: "bg-white/10",
    badgeBorder: "border-white/10",
    bar: "linear-gradient(90deg, rgba(148,163,184,0.75), rgba(100,116,139,0.45))"
  }
];

function bandOf(cos: number): Band {
  return BANDS.find((b) => cos >= b.min && cos < b.max) ?? BANDS[BANDS.length - 1];
}

/** Mini pipeline (visual) */
type PipeStepKey = "Upload" | "Preprocess" | "Embed" | "Search" | "Rank";
const PIPE_STEPS: PipeStepKey[] = ["Upload", "Preprocess", "Embed", "Search", "Rank"];

function Pipeline({
  running,
  done,
  elapsedMs
}: {
  running: boolean;
  done: boolean;
  elapsedMs?: number | null;
}) {
  const [active, setActive] = useState<number>(-1);
  const [split, setSplit] = useState<Record<PipeStepKey, number> | null>(null);

  useEffect(() => {
    let timer: number | null = null;

    if (running) {
      setSplit(null);
      setActive(0);
      timer = window.setInterval(() => {
        setActive((x) => Math.min(PIPE_STEPS.length - 1, x + 1));
      }, 260);
    }

    return () => {
      if (timer) window.clearInterval(timer);
    };
  }, [running]);

  useEffect(() => {
    if (done && elapsedMs != null && elapsedMs > 0) {
      // demo-friendly split (no backend change)
      const total = elapsedMs;
      const p = [0.18, 0.22, 0.33, 0.17, 0.10];
      const s: Record<PipeStepKey, number> = {
        Upload: Math.max(1, Math.round(total * p[0])),
        Preprocess: Math.max(1, Math.round(total * p[1])),
        Embed: Math.max(1, Math.round(total * p[2])),
        Search: Math.max(1, Math.round(total * p[3])),
        Rank: Math.max(
          1,
          total -
            (Math.round(total * p[0]) +
              Math.round(total * p[1]) +
              Math.round(total * p[2]) +
              Math.round(total * p[3]))
        )
      };
      setSplit(s);
      setActive(PIPE_STEPS.length - 1);
    }
  }, [done, elapsedMs]);

  return (
    <div className="rounded-2xl border border-white/10 bg-white/5 p-4">
      <div className="flex items-center justify-between">
        <div className="text-sm font-extrabold">Pipeline</div>
        <div className="text-xs opacity-70">{running ? "running…" : done ? "done" : "—"}</div>
      </div>

      <div className="mt-3 flex items-center gap-2 flex-wrap">
        {PIPE_STEPS.map((k, i) => {
          const isOn = running ? i <= active : done ? true : false;
          const ms = split ? split[k] : null;
          return (
            <div key={k} className="flex items-center gap-2">
              <div
                className={cn(
                  "rounded-xl px-3 py-2 border text-xs font-extrabold",
                  isOn ? "bg-indigo-500/20 border-indigo-400/30" : "bg-white/5 border-white/10 opacity-60"
                )}
              >
                {k}
                {ms != null && <span className="ml-2 font-black opacity-80">{ms}ms</span>}
              </div>
              {i !== PIPE_STEPS.length - 1 && <div className="h-px w-6 bg-white/15" />}
            </div>
          );
        })}
      </div>
    </div>
  );
}

export default function App() {
  const [tab, setTab] = useState<"catdog" | "whic" | "history">("catdog");
  const [source, setSource] = useState<SourceType>("catdog");

  const [theme, setTheme] = useState<"dark" | "light">("dark");

  const [queryFile, setQueryFile] = useState<File | null>(null);
  const [queryPreview, setQueryPreview] = useState<string | null>(null);

  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const [resp, setResp] = useState<RetrieveResponse | null>(null);
  const [history, setHistory] = useState<HistoryRecord[]>(() => loadHistory());

  const [searchText, setSearchText] = useState("");
  const [grid, setGrid] = useState<"grid" | "dense">("grid"); // grid=瀑布流，dense=密集网格
  const [sortBy, setSortBy] = useState<"cosine" | "score">("cosine");

  const [favorites, setFavorites] = useState<Record<string, TopKItem>>({});
  const [favOpen, setFavOpen] = useState(false);
  const [compareA, setCompareA] = useState<TopKItem | null>(null);
  const [compareB, setCompareB] = useState<TopKItem | null>(null);

  const [modalItem, setModalItem] = useState<TopKItem | null>(null);
  const fileRef = useRef<HTMLInputElement | null>(null);

  // Hero particles seeds
  const particles = useMemo(() => {
    const n = 26;
    return Array.from({ length: n }).map((_, i) => {
      const r = Math.random();
      const r2 = Math.random();
      const r3 = Math.random();
      return {
        id: i,
        left: `${Math.round(r * 100)}%`,
        top: `${Math.round(r2 * 100)}%`,
        size: `${8 + Math.round(r3 * 22)}px`,
        delay: `${Math.round(Math.random() * 1400)}ms`,
        dur: `${3800 + Math.round(Math.random() * 3800)}ms`,
        alpha: 0.12 + Math.random() * 0.18
      };
    });
  }, []);

  useEffect(() => {
    const saved = localStorage.getItem(LS_THEME);
    if (saved === "light" || saved === "dark") setTheme(saved);
  }, []);

  useEffect(() => {
    localStorage.setItem(LS_THEME, theme);
    document.documentElement.style.colorScheme = theme;
    if (theme === "light") {
      document.body.classList.remove("bg-slate-950", "text-slate-100");
      document.body.classList.add("bg-slate-50", "text-slate-950");
    } else {
      document.body.classList.remove("bg-slate-50", "text-slate-950");
      document.body.classList.add("bg-slate-950", "text-slate-100");
    }
  }, [theme]);

  useEffect(() => {
    saveHistory(history);
  }, [history]);

  function openPicker() {
    fileRef.current?.click();
  }

  async function onPickFile(f: File) {
    setError(null);
    setQueryFile(f);
    const preview = await readFileAsDataURL(f);
    setQueryPreview(preview);
    await runRetrieve(source, f, preview);
  }

  async function runRetrieve(s: SourceType, f: File, preview?: string) {
    setLoading(true);
    setError(null);
    try {
      const r = await retrieve(s, f);
      setResp(r);

      const rec: HistoryRecord = {
        id: uid("his"),
        created_at: Date.now(),
        source: s,
        query_name: f.name,
        query_preview: preview ?? queryPreview ?? undefined,
        predicted_label: r.predicted_label,
        request_id: r.request_id,
        top1: r.topk?.[0]
      };
      setHistory([rec, ...history].slice(0, 50));
      setTab(s);
      setSource(s);
    } catch (e: any) {
      setError(String(e?.message || e));
    } finally {
      setLoading(false);
    }
  }

  const isWhicMode = (resp?.source ?? source) === "whic";

  const filteredTopK = useMemo(() => {
    if (!resp?.topk) return [];
    let arr = resp.topk.slice();

    const q = searchText.trim().toLowerCase();
    if (q) {
      arr = arr.filter((x) => (x.caption || "").toLowerCase().includes(q) || x.image_url.toLowerCase().includes(q));
    }

    arr.sort((a, b) => (sortBy === "cosine" ? b.cosine - a.cosine : b.score - a.score));
    arr.forEach((x, i) => (x.rank = i + 1));
    return arr;
  }, [resp, searchText, sortBy]);

  // 分组（按相似度区间）
  const groupedTopK = useMemo(() => {
    const groups = BANDS.map((b) => ({ band: b, items: [] as TopKItem[] }));
    for (const it of filteredTopK) {
      const b = bandOf(it.cosine);
      const g = groups.find((x) => x.band.key === b.key);
      (g ? g.items : groups[groups.length - 1].items).push(it);
    }
    return groups.filter((g) => g.items.length > 0);
  }, [filteredTopK]);

  // ✅ Cat/Dog：分类概率；✅ WHIC：关键词（从 TopK captions 提取）
  const probData = useMemo(() => {
    if (isWhicMode) {
      const items = (resp?.topk ?? []).slice(0, 50);
      const counts: Record<string, number> = {};

      for (const it of items) {
        const text = (it.caption || "").toLowerCase();
        const tokens = text.match(/[a-z]+/g) || [];
        // 权重：越相似越重要（展示效果更好）
        const w = Math.max(0.2, Math.min(1.2, 0.6 + it.cosine));

        for (const t of tokens) {
          if (t.length < 3) continue;
          if (STOPWORDS.has(t)) continue;
          counts[t] = (counts[t] || 0) + w;
        }
      }

      return Object.entries(counts)
        .sort((a, b) => b[1] - a[1])
        .slice(0, 8)
        .map(([name, score]) => ({
          name: CN_MAP[name] ?? name,
          prob: Math.round(score * 10) / 10
        }));
    }

    const p = resp?.probs ?? [];
    return p
      .slice()
      .sort((a, b) => b.prob - a.prob)
      .map((x) => ({ name: x.label, prob: Math.round(x.prob * 1000) / 10 }));
  }, [resp, source, isWhicMode]);

  // ✅ WHIC 的“先判断这是什么”（从 Top captions 里做轻量场景判断）
  const sceneGuess = useMemo(() => {
    if (!isWhicMode) return null;
    const text = (resp?.topk ?? []).slice(0, 12).map(x => (x.caption || "").toLowerCase()).join(" ");
    const has = (set: Set<string>) => [...set].some(w => text.includes(w));

    if (has(LANDSCAPE_WORDS)) return "风景 / 自然";
    if (has(CITY_WORDS)) return "城市 / 建筑";
    if (has(ANIMAL_WORDS)) return "动物";
    if (has(PEOPLE_WORDS)) return "人物";
    return "综合场景";
  }, [isWhicMode, resp]);

  const simDist = useMemo(() => {
    const bins = Array.from({ length: 10 }, (_, i) => ({
      bin: `${(i / 10).toFixed(1)}-${((i + 1) / 10).toFixed(1)}`,
      count: 0
    }));
    const arr = filteredTopK;
    for (const x of arr) {
      const idx = Math.min(9, Math.max(0, Math.floor(x.cosine * 10)));
      bins[idx].count += 1;
    }
    return bins;
  }, [filteredTopK]);

  const topProb = useMemo(() => {
    const p = resp?.probs ?? [];
    if (p.length === 0) return { label: "-", prob: 0 };
    const sorted = p.slice().sort((a, b) => b.prob - a.prob);
    return { label: sorted[0].label, prob: sorted[0].prob };
  }, [resp]);

  function toggleFav(item: TopKItem) {
    setFavorites((prev) => {
      const key = item.image_url;
      const next = { ...prev };
      if (next[key]) delete next[key];
      else next[key] = item;
      return next;
    });
  }

  function exportCSV() {
    if (!resp) return;
    const rows = filteredTopK.map((x) => ({
      rank: x.rank,
      image_url: x.image_url,
      caption: x.caption ?? "",
      cosine: x.cosine,
      score: x.score
    }));
    downloadText(`top50_${resp.source}_${resp.request_id || "session"}.csv`, toCSV(rows));
  }

  function pickCompare(item: TopKItem) {
    if (!compareA) setCompareA(item);
    else if (!compareB) setCompareB(item);
    else {
      setCompareA(item);
      setCompareB(null);
    }
  }

  const heroBg =
    theme === "dark"
      ? "bg-gradient-to-br from-indigo-500/25 via-fuchsia-500/15 to-emerald-400/10"
      : "bg-gradient-to-br from-indigo-500/10 via-fuchsia-500/10 to-emerald-400/10";

  return (
    <div className="min-h-screen">
      {/* Top bar */}
      <div className={cn("relative overflow-hidden border-b border-white/10", heroBg)}>
        {/* animated gradient layer */}
        <div className="pointer-events-none absolute inset-0 opacity-70">
          <div className="absolute inset-0 bg-[radial-gradient(closest-side,rgba(255,255,255,0.10),transparent)]" />
          <div className="absolute inset-0 hero-anim-gradient" />
          {/* particles */}
          {particles.map((p) => (
            <span
              key={p.id}
              className="absolute rounded-full blur-[2px] particle-float"
              style={{
                left: p.left,
                top: p.top,
                width: p.size,
                height: p.size,
                background: `rgba(255,255,255,${p.alpha})`,
                animationDelay: p.delay,
                animationDuration: p.dur
              }}
            />
          ))}
        </div>

        <div className="mx-auto max-w-7xl px-5 py-6 relative">
          <div className="flex flex-col gap-4 md:flex-row md:items-center md:justify-between">
            <div>
              <div className="flex items-center gap-3">
                <div className="rounded-2xl bg-white/10 p-3 border border-white/10">
                  <Sparkles className="h-6 w-6" />
                </div>
                <div>
                  <div className="text-2xl md:text-3xl font-black tracking-tight">
                    ViT Image Intelligence Hub <span className="opacity-90">🐱🐶🌍</span>
                  </div>
                </div>
              </div>

              <div className="mt-4 flex flex-wrap gap-2">
                <Badge className="bg-indigo-500/20 text-indigo-200 border border-indigo-400/20">Cat/Dog</Badge>
                <Badge className="bg-emerald-500/20 text-emerald-200 border border-emerald-400/20">WHIC</Badge>
                <Badge className="bg-fuchsia-500/20 text-fuchsia-200 border border-fuchsia-400/20">Top-50</Badge>
                <Badge className="bg-white/10 text-white/80 border border-white/10">History</Badge>
              </div>
            </div>

            <div className="flex items-center gap-2">
              <GhostButton onClick={() => setTheme(theme === "dark" ? "light" : "dark")}>
                {theme === "dark" ? <Sun className="h-4 w-4" /> : <Moon className="h-4 w-4" />}
                {theme === "dark" ? "Light" : "Dark"}
              </GhostButton>
              <Button onClick={openPicker}>
                <Upload className="h-4 w-4" />
                上传图片
              </Button>
              <input
                ref={fileRef}
                type="file"
                accept="image/*"
                className="hidden"
                onChange={(e) => {
                  const f = e.target.files?.[0];
                  if (f) void onPickFile(f);
                }}
              />
            </div>
          </div>
        </div>
      </div>

      {/* Tabs */}
      <div className="mx-auto max-w-7xl px-5 py-6">
        <div className="flex flex-wrap gap-2">
          <button
            onClick={() => {
              setTab("catdog");
              setSource("catdog");
            }}
            className={cn(
              "rounded-2xl px-4 py-2 text-sm font-extrabold border transition",
              tab === "catdog" ? "bg-indigo-500/20 border-indigo-400/30" : "bg-white/5 border-white/10 hover:bg-white/10"
            )}
          >
            🐾 Cat/Dog
          </button>
          <button
            onClick={() => {
              setTab("whic");
              setSource("whic");
            }}
            className={cn(
              "rounded-2xl px-4 py-2 text-sm font-extrabold border transition",
              tab === "whic" ? "bg-emerald-500/20 border-emerald-400/30" : "bg-white/5 border-white/10 hover:bg-white/10"
            )}
          >
            🌐 WHIC
          </button>
          <button
            onClick={() => setTab("history")}
            className={cn(
              "rounded-2xl px-4 py-2 text-sm font-extrabold border transition",
              tab === "history" ? "bg-fuchsia-500/20 border-fuchsia-400/30" : "bg-white/5 border-white/10 hover:bg-white/10"
            )}
          >
            🗂 History
          </button>

          <div className="ml-auto flex gap-2">
            <GhostButton onClick={() => setFavOpen(true)}>
              <Star className="h-4 w-4" />
              收藏({Object.keys(favorites).length})
            </GhostButton>
            <GhostButton onClick={exportCSV} disabled={!resp}>
              <Download className="h-4 w-4" />
              CSV
            </GhostButton>
          </div>
        </div>

        {/* Content */}
        <div className="mt-6 grid grid-cols-1 gap-5 lg:grid-cols-12">
          {/* Left */}
          <div className="lg:col-span-4 space-y-5">
            <Card className="p-5">
              <div className="flex items-center justify-between">
                <div className="text-lg font-extrabold">上传检索</div>
                <Badge className="bg-white/10 border border-white/10 text-white/80">{source.toUpperCase()}</Badge>
              </div>

              <div
                className={cn(
                  "mt-4 rounded-2xl border border-dashed border-white/20 bg-white/5 p-4",
                  "hover:bg-white/10 transition cursor-pointer"
                )}
                onClick={openPicker}
                onDragOver={(e) => e.preventDefault()}
                onDrop={(e) => {
                  e.preventDefault();
                  const f = e.dataTransfer.files?.[0];
                  if (f) void onPickFile(f);
                }}
              >
                <div className="flex items-center gap-3">
                  <div className="rounded-xl bg-indigo-500/20 p-3 border border-indigo-400/20">
                    <Upload className="h-5 w-5 text-indigo-200" />
                  </div>
                  <div>
                    <div className="font-bold">点击或拖拽上传</div>
                    <div className="text-xs opacity-70">{loading ? "处理中…" : "自动检索 Top-50"}</div>
                  </div>
                </div>

                {queryPreview && (
                  <div className="mt-4">
                    <img
                      src={queryPreview}
                      className="w-full rounded-xl border border-white/10 object-cover max-h-64 image-fadein"
                      alt="query preview"
                    />
                    <div className="mt-2 text-xs opacity-70 truncate">{queryFile?.name}</div>
                  </div>
                )}
              </div>

              {error && (
                <div className="mt-4 rounded-xl bg-red-500/10 border border-red-400/20 p-3 text-sm text-red-200">
                  {error}
                </div>
              )}

              <div className="mt-4 flex gap-2">
                <GhostButton
                  onClick={() => {
                    setSource("catdog");
                    setTab("catdog");
                  }}
                >
                  🐾 Cat/Dog
                </GhostButton>
                <GhostButton
                  onClick={() => {
                    setSource("whic");
                    setTab("whic");
                  }}
                >
                  🌐 WHIC
                </GhostButton>
              </div>
            </Card>

            {/* Pipeline */}
            <Pipeline running={loading} done={!loading && !!resp && !error} elapsedMs={resp?.elapsed_ms} />

            {/* KPI */}
            <Card className="p-5">
              <div className="flex items-center justify-between">
                <div className="text-lg font-extrabold">KPI</div>
                {resp?.request_id ? (
                  <Badge className="bg-black/20 border border-white/10 text-white/70">{resp.request_id}</Badge>
                ) : (
                  <Badge className="bg-white/10 border border-white/10 text-white/70">—</Badge>
                )}
              </div>

              <div className="mt-4 grid grid-cols-1 gap-3">
                <KpiTile
                  tone="indigo"
                  icon={<Tag className="h-5 w-5" />}
                  title="Predicted"
                  value={resp?.predicted_label ?? "—"}
                  sub={resp?.source ? resp.source.toUpperCase() : undefined}
                  right={<Ring value={topProb.prob || 0} label={topProb.label} />}
                />

                <KpiTile
                  tone="emerald"
                  icon={<Timer className="h-5 w-5" />}
                  title="Latency"
                  value={resp?.elapsed_ms != null ? `${resp.elapsed_ms}` : "—"}
                  sub="ms"
                  right={
                    <div className="w-28">
                      <div className="text-xs opacity-70 mb-2">speed</div>
                      <div className="h-2 rounded-full bg-white/10 overflow-hidden">
                        <div
                          className="h-2 rounded-full bg-emerald-400/90"
                          style={{
                            width:
                              resp?.elapsed_ms != null
                                ? `${Math.max(10, Math.min(100, 1000 / (resp.elapsed_ms + 1)))}%`
                                : "0%"
                          }}
                        />
                      </div>
                    </div>
                  }
                />

                <KpiTile
                  tone="fuchsia"
                  icon={<Database className="h-5 w-5" />}
                  title="Gallery"
                  value={resp?.gallery_size != null ? resp.gallery_size.toLocaleString() : "—"}
                  sub="images"
                  right={
                    <div className="flex items-center gap-2">
                      <div className="rounded-xl bg-white/10 border border-white/10 p-2">
                        <Zap className="h-5 w-5 text-fuchsia-200" />
                      </div>
                    </div>
                  }
                />
              </div>
            </Card>
          </div>

          {/* Right */}
          <div className="lg:col-span-8 space-y-5">
            {tab === "history" ? (
              <Timeline
                items={history}
                onClear={() => {
                  setHistory([]);
                  localStorage.removeItem(LS_KEY);
                }}
              />
            ) : (
              <>
                {/* Charts */}
                <Card className="p-5">
                  <div className="flex flex-col gap-4 md:flex-row md:items-center md:justify-between">
                    <div className="flex items-center gap-2">
                      <Sparkles className="h-5 w-5 text-indigo-300" />
                      <div className="text-lg font-extrabold">结果</div>
                    </div>

                    <div className="flex flex-wrap gap-2 items-center">
                      <div className="relative">
                        <Search className="absolute left-3 top-2.5 h-4 w-4 opacity-60" />
                        <input
                          value={searchText}
                          onChange={(e) => setSearchText(e.target.value)}
                          placeholder="搜索…"
                          className="w-64 rounded-xl bg-white/5 border border-white/10 px-9 py-2 text-sm outline-none focus:ring-2 focus:ring-white/20"
                        />
                      </div>

                      <GhostButton onClick={() => setSortBy(sortBy === "cosine" ? "score" : "cosine")}>
                        {sortBy === "cosine" ? "cosine" : "score"}
                      </GhostButton>

                      <GhostButton onClick={() => setGrid(grid === "grid" ? "dense" : "grid")}>
                        <LayoutGrid className="h-4 w-4" />
                        {grid === "grid" ? "瀑布流" : "密集"}
                      </GhostButton>

                      <GhostButton
                        onClick={() => {
                          setCompareA(null);
                          setCompareB(null);
                        }}
                        disabled={!compareA && !compareB}
                      >
                        <GitCompare className="h-4 w-4" />
                        清空对比
                      </GhostButton>
                    </div>
                  </div>

                  <div className="mt-5 grid grid-cols-1 lg:grid-cols-2 gap-4">
                    {/* 左图：Cat/Dog=分类概率；WHIC=关键词 + 判断 */}
                    <div className="rounded-2xl bg-white/5 border border-white/10 p-4">
                      <div className="flex items-center justify-between">
                        <div className="text-sm font-extrabold">{isWhicMode ? "关键词" : "分类概率"}</div>
                        {isWhicMode && sceneGuess && (
                          <Badge className="bg-emerald-500/15 border border-emerald-400/20 text-emerald-200">
                            判断：{sceneGuess}
                          </Badge>
                        )}
                      </div>

                      <div className="mt-2 h-44">
                        {probData.length === 0 ? (
                          <div className="h-full grid place-items-center text-sm opacity-70">—</div>
                        ) : (
                          <ResponsiveContainer width="100%" height="100%">
                            <BarChart data={probData}>
                              <XAxis dataKey="name" stroke="rgba(255,255,255,0.4)" />
                              <YAxis stroke="rgba(255,255,255,0.4)" />
                              <Tooltip
                                contentStyle={{
                                  background: "rgba(15,23,42,0.9)",
                                  border: "1px solid rgba(255,255,255,0.1)",
                                  borderRadius: 12
                                }}
                              />
                              <Bar dataKey="prob" radius={[10, 10, 0, 0]}>
                                {probData.map((entry, index) => {
                                  // Cat/Dog 用固定配色；WHIC 用调色盘（更好看）
                                  if (!isWhicMode) {
                                    const name = String(entry.name || "").toLowerCase();
                                    const fill = name.includes("cat")
                                      ? "rgba(99,102,241,0.95)"
                                      : name.includes("dog")
                                      ? "rgba(34,197,94,0.95)"
                                      : "rgba(236,72,153,0.95)";
                                    return <Cell key={`cell-${index}`} fill={fill} />;
                                  }
                                  return <Cell key={`cell-${index}`} fill={WHIC_BAR_COLORS[index % WHIC_BAR_COLORS.length]} />;
                                })}
                              </Bar>
                            </BarChart>
                          </ResponsiveContainer>
                        )}
                      </div>
                    </div>

                    <div className="rounded-2xl bg-white/5 border border-white/10 p-4">
                      <div className="text-sm font-extrabold">相似度分布</div>
                      <div className="mt-2 h-44">
                        <ResponsiveContainer width="100%" height="100%">
                          <AreaChart data={simDist}>
                            <XAxis dataKey="bin" stroke="rgba(255,255,255,0.4)" />
                            <YAxis stroke="rgba(255,255,255,0.4)" />
                            <Tooltip
                              contentStyle={{
                                background: "rgba(15,23,42,0.9)",
                                border: "1px solid rgba(255,255,255,0.1)",
                                borderRadius: 12
                              }}
                            />
                            <Area dataKey="count" />
                          </AreaChart>
                        </ResponsiveContainer>
                      </div>
                    </div>
                  </div>
                </Card>

                {/* Results */}
                <Card className="p-5">
                  <div className="flex items-center justify-between">
                    <div className="text-lg font-extrabold">Top-50</div>
                    <div className="text-xs opacity-70">{loading ? "loading…" : "cosine / score"}</div>
                  </div>

                  {!resp && !loading && (
                    <div className="mt-4 rounded-2xl bg-white/5 border border-white/10 p-6 text-sm opacity-70">—</div>
                  )}

                  {/* skeleton */}
                  {loading && (
                    <div
                      className={cn(
                        "mt-4 grid gap-4",
                        grid === "grid"
                          ? "grid-cols-1 sm:grid-cols-2 lg:grid-cols-3"
                          : "grid-cols-2 sm:grid-cols-3 lg:grid-cols-5"
                      )}
                    >
                      {Array.from({ length: grid === "grid" ? 12 : 20 }).map((_, i) => (
                        <SkeletonCard key={i} dense={grid === "dense"} />
                      ))}
                    </div>
                  )}

                  {/* grouped + waterfall */}
                  {resp && !loading && (
                    <div className="mt-4 space-y-6">
                      {groupedTopK.map((g) => {
                        const mean = g.items.reduce((s, x) => s + x.cosine, 0) / Math.max(1, g.items.length);
                        return (
                          <div key={g.band.key}>
                            <div
                              className={cn(
                                "flex items-center justify-between rounded-2xl border px-4 py-3",
                                g.band.headBg,
                                g.band.headBorder
                              )}
                            >
                              <div className="flex items-center gap-2">
                                <div className="text-sm font-extrabold">{g.band.title}</div>
                                <Badge className={cn("border text-white/90", g.band.badgeBg, g.band.badgeBorder)}>
                                  {g.band.hint}
                                </Badge>
                              </div>
                              <div className="text-xs opacity-70">
                                {g.items.length} · mean {formatNum(mean, 4)}
                              </div>
                            </div>

                            {grid === "grid" ? (
                              <div className="mt-4 columns-1 sm:columns-2 lg:columns-3 gap-4 [column-fill:_balance]">
                                {g.items.map((item) => (
                                  <ResultCard
                                    key={`${item.rank}_${item.image_url}`}
                                    item={item}
                                    band={g.band}
                                    favored={!!favorites[item.image_url]}
                                    onFav={() => toggleFav(item)}
                                    onCompare={() => pickCompare(item)}
                                    onOpen={() => setModalItem(item)}
                                    masonry
                                  />
                                ))}
                              </div>
                            ) : (
                              <div className="mt-4 grid grid-cols-2 sm:grid-cols-3 lg:grid-cols-5 gap-4">
                                {g.items.map((item) => (
                                  <ResultCard
                                    key={`${item.rank}_${item.image_url}`}
                                    item={item}
                                    band={g.band}
                                    favored={!!favorites[item.image_url]}
                                    onFav={() => toggleFav(item)}
                                    onCompare={() => pickCompare(item)}
                                    onOpen={() => setModalItem(item)}
                                  />
                                ))}
                              </div>
                            )}
                          </div>
                        );
                      })}
                    </div>
                  )}
                </Card>

                {/* Compare */}
                <Card className="p-5">
                  <div className="flex items-center justify-between">
                    <div className="text-lg font-extrabold">对比面板</div>
                  </div>

                  <div className="mt-4 grid grid-cols-1 md:grid-cols-2 gap-4">
                    {[compareA, compareB].map((it, idx) => (
                      <div
                        key={idx}
                        className={cn(
                          "rounded-2xl border border-white/10 p-4",
                          "bg-gradient-to-br from-white/5 to-white/0 hover:from-white/10 hover:to-white/5 transition"
                        )}
                      >
                        <div className="flex items-center justify-between">
                          <div className="font-extrabold">{idx === 0 ? "对比 A" : "对比 B"}</div>
                          <GhostButton onClick={() => (idx === 0 ? setCompareA(null) : setCompareB(null))}>清除</GhostButton>
                        </div>

                        {!it ? (
                          <div className="mt-4 text-sm opacity-70">—</div>
                        ) : (
                          <div className="mt-4 flex gap-3">
                            <img
                              src={it.image_url}
                              className="h-24 w-36 rounded-xl border border-white/10 object-cover bg-black/30"
                              alt="compare"
                            />
                            <div className="text-sm">
                              <div className="font-extrabold">#{it.rank}</div>
                              <div className="mt-1 opacity-80">cosine: {formatNum(it.cosine, 4)}</div>
                              <div className="opacity-80">score: {formatNum(it.score, 2)}</div>
                              <div className="mt-2 text-xs opacity-70">{it.caption || ""}</div>
                            </div>
                          </div>
                        )}
                      </div>
                    ))}
                  </div>
                </Card>
              </>
            )}
          </div>
        </div>
      </div>

      {/* Favorites */}
      <Modal open={favOpen} title="收藏夹" onClose={() => setFavOpen(false)}>
        <div className="flex items-center justify-between">
          <div className="text-sm opacity-80">—</div>
          <GhostButton onClick={() => setFavorites({})}>清空</GhostButton>
        </div>

        <div className="mt-4 grid grid-cols-2 md:grid-cols-4 gap-3">
          {Object.values(favorites).map((it) => (
            <div
              key={it.image_url}
              className="rounded-2xl bg-white/5 border border-white/10 overflow-hidden hover:bg-white/10 transition"
            >
              <img src={it.image_url} className="h-28 w-full object-cover bg-black/30 image-fadein" alt="fav" />
              <div className="p-3 text-xs">
                <div className="flex items-center justify-between">
                  <span className="opacity-70">cosine</span>
                  <span className="font-extrabold">{formatNum(it.cosine, 4)}</span>
                </div>
                <div className="flex items-center justify-between mt-1">
                  <span className="opacity-70">score</span>
                  <span className="font-extrabold">{formatNum(it.score, 2)}</span>
                </div>
                <button
                  className="mt-2 w-full rounded-xl bg-white/5 hover:bg-white/10 border border-white/10 py-2 font-bold"
                  onClick={() => toggleFav(it)}
                >
                  取消收藏
                </button>
              </div>
            </div>
          ))}
          {Object.keys(favorites).length === 0 && <div className="text-sm opacity-70">—</div>}
        </div>
      </Modal>

      {/* Item Detail */}
      <Modal open={!!modalItem} title="图片详情" onClose={() => setModalItem(null)}>
        {modalItem && (
          <div className="grid grid-cols-1 lg:grid-cols-2 gap-4">
            <div className="rounded-2xl bg-white/5 border border-white/10 p-3">
              <img
                src={modalItem.image_url}
                className="w-full rounded-xl object-cover bg-black/30 max-h-[520px] image-fadein"
                alt="detail"
              />
            </div>
            <div className="rounded-2xl bg-white/5 border border-white/10 p-4">
              <div className="text-sm font-extrabold">Rank #{modalItem.rank}</div>
              <div className="mt-2 text-sm opacity-80">{modalItem.caption || "—"}</div>

              <div className="mt-4 grid grid-cols-2 gap-3">
                <div className="rounded-xl bg-indigo-500/10 border border-indigo-400/20 p-3">
                  <div className="text-xs opacity-70">cosine</div>
                  <div className="text-lg font-black">{formatNum(modalItem.cosine, 4)}</div>
                </div>
                <div className="rounded-xl bg-emerald-500/10 border border-emerald-400/20 p-3">
                  <div className="text-xs opacity-70">score</div>
                  <div className="text-lg font-black">{formatNum(modalItem.score, 2)}</div>
                </div>
              </div>

              <div className="mt-4">
                <div className="text-xs font-bold opacity-70 mb-2">URL</div>
                <div className="rounded-xl bg-black/30 border border-white/10 p-3 text-xs break-all">
                  {modalItem.image_url}
                </div>
                <div className="mt-3 flex gap-2">
                  <GhostButton onClick={() => navigator.clipboard.writeText(modalItem.image_url)}>复制</GhostButton>
                  <GhostButton onClick={() => toggleFav(modalItem)}>
                    <Star className="h-4 w-4" />
                    {favorites[modalItem.image_url] ? "取消收藏" : "收藏"}
                  </GhostButton>
                </div>
              </div>
            </div>
          </div>
        )}
      </Modal>

      {/* keyframes */}
      <style>
        {`
          .hero-anim-gradient{
            background: radial-gradient(1200px 600px at 10% 10%, rgba(99,102,241,0.35), transparent 60%),
                        radial-gradient(900px 500px at 80% 25%, rgba(236,72,153,0.28), transparent 60%),
                        radial-gradient(1000px 600px at 60% 90%, rgba(34,197,94,0.18), transparent 60%);
            background-size: 160% 160%;
            animation: gradientShift 8s ease-in-out infinite;
          }
          @keyframes gradientShift{
            0%{ transform: translate3d(0,0,0); filter: saturate(1.05); }
            50%{ transform: translate3d(-2%,1%,0); filter: saturate(1.25); }
            100%{ transform: translate3d(0,0,0); filter: saturate(1.05); }
          }

          .particle-float{
            animation-name: particleFloat;
            animation-timing-function: ease-in-out;
            animation-iteration-count: infinite;
          }
          @keyframes particleFloat{
            0%{ transform: translate3d(0,0,0) scale(1); opacity: .65; }
            50%{ transform: translate3d(18px,-14px,0) scale(1.08); opacity: 1; }
            100%{ transform: translate3d(0,0,0) scale(1); opacity: .65; }
          }

          .image-fadein{
            animation: imgFade 520ms ease-out both;
          }
          @keyframes imgFade{
            from{ opacity: 0; filter: blur(10px); transform: scale(1.015); }
            to{ opacity: 1; filter: blur(0); transform: scale(1); }
          }

          @keyframes shine {
            0% { transform: translateX(-40%) rotate(12deg); opacity: 0; }
            20% { opacity: .55; }
            60% { opacity: .15; }
            100% { transform: translateX(220%) rotate(12deg); opacity: 0; }
          }
        `}
      </style>
    </div>
  );
}

/** Result Card with hover + heatbar */
function ResultCard({
  item,
  band,
  favored,
  onFav,
  onCompare,
  onOpen,
  masonry
}: {
  item: TopKItem;
  band: Band;
  favored: boolean;
  onFav: () => void;
  onCompare: () => void;
  onOpen: () => void;
  masonry?: boolean;
}) {
  const w = Math.max(0, Math.min(100, item.cosine * 100));
  return (
    <div
      className={cn(
        "group rounded-2xl border border-white/10 bg-white/5 overflow-hidden",
        "transition-all duration-300 hover:bg-white/10 hover:-translate-y-1 hover:shadow-[0_18px_45px_rgba(0,0,0,0.35)]",
        masonry ? "mb-4 break-inside-avoid" : ""
      )}
    >
      <div className="relative">
        <img
          src={item.image_url}
          alt={item.caption || `rank ${item.rank}`}
          className={cn(
            "w-full object-cover bg-black/30 image-fadein",
            masonry ? "h-auto" : "h-28",
            "transition-transform duration-500 group-hover:scale-[1.03]"
          )}
          onClick={onOpen}
          onError={(e) => {
            (e.currentTarget as HTMLImageElement).style.opacity = "0.3";
          }}
        />

        <div className="absolute inset-0 opacity-0 group-hover:opacity-100 transition">
          <div className="absolute inset-0 bg-gradient-to-t from-black/55 via-black/10 to-transparent" />
          <div className="absolute -left-24 top-0 h-full w-24 rotate-12 bg-white/10 blur-xl animate-[shine_1.8s_ease-in-out_infinite]" />
        </div>

        <div className="absolute top-2 left-2">
          <Badge className="bg-black/50 border border-white/10 text-white">#{item.rank}</Badge>
        </div>

        <div className="absolute top-2 right-2 flex gap-2 opacity-0 group-hover:opacity-100 transition">
          <button
            onClick={onFav}
            className="rounded-lg p-2 bg-black/40 border border-white/10 hover:bg-black/60"
            title="收藏"
          >
            {favored ? <Star className="h-4 w-4 text-yellow-300" /> : <StarOff className="h-4 w-4" />}
          </button>
          <button
            onClick={onCompare}
            className="rounded-lg p-2 bg-black/40 border border-white/10 hover:bg-black/60"
            title="加入对比"
          >
            <GitCompare className="h-4 w-4" />
          </button>
        </div>
      </div>

      <div className="p-3">
        <div className="text-xs opacity-70 truncate">{item.caption || "—"}</div>

        <div className="mt-2 space-y-2">
          <div className="flex items-center justify-between text-xs">
            <span className="opacity-70">cosine</span>
            <span className="font-extrabold">{formatNum(item.cosine, 4)}</span>
          </div>

          <div className="h-2 rounded-full bg-white/10 overflow-hidden">
            <div className="h-2 rounded-full" style={{ width: `${w}%`, background: band.bar }} />
          </div>

          <div className="flex items-center justify-between text-xs">
            <span className="opacity-70">score</span>
            <span className="font-extrabold">{formatNum(item.score, 2)}</span>
          </div>
        </div>
      </div>
    </div>
  );
}