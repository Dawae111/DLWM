interface PaperResult {
  method: string;
  accuracy: [number, number, number, number];
  memory: number;
  throughput: number;
}

// Column order and values from arXiv:2606.15160v2, Tables 1(a) and 1(b).
export const benchmarks = ['GQA', 'VQAv2', 'ScienceQA', 'MMVP'] as const;
export const results: PaperResult[] = [
  { method: 'LVR', accuracy: [66.4, 74.8, 72.1, 60.9], memory: 19.0, throughput: 6.1 },
  { method: 'LaRe', accuracy: [66.9, 75.0, 72.6, 61.7], memory: 19.5, throughput: 5.9 },
  { method: 'Heima', accuracy: [65.2, 74.3, 71.5, 59.8], memory: 13.8, throughput: 9.2 },
  { method: 'DLWM', accuracy: [69.3, 76.0, 75.2, 66.1], memory: 14.5, throughput: 8.9 },
];
