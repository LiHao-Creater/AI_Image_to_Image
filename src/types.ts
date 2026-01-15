export type SourceType = "catdog" | "whic";

export type ProbItem = {
  label: string;
  prob: number; // 0..1
};

export type TopKItem = {
  rank: number; // 1..50
  image_url: string;
  caption?: string;
  cosine: number; // cosine similarity
  score: number; // your score (any scale)
};

export type RetrieveResponse = {
  source: SourceType;
  predicted_label: string;
  probs?: ProbItem[];
  gallery_size?: number;
  elapsed_ms?: number;
  request_id?: string;
  topk: TopKItem[];
};

export type HistoryRecord = {
  id: string; // local id
  created_at: number;
  source: SourceType;
  query_name: string;
  query_preview?: string; // dataURL
  predicted_label: string;
  request_id?: string;
  top1?: TopKItem;
};
