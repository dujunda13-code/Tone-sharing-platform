import { useSyncExternalStore } from "react";

import type {
  CloudEmotionLabel,
  DatasetSummary,
  EmotionLabel,
  JobStatus,
  LocalSynthesisOptions,
  SegmentListResponse,
  SynthesisLanguage,
  SynthesisMode,
} from "../api/types";
import type { SafeError } from "../components/ErrorPanel";
import type { CreateVoiceStep } from "../components/CreateVoiceWizard";

/**
 * 会话内草稿存储：页面切换会卸载页面组件，这里把用户未提交的操作保存在内存中，
 * 返回同一页面时可继续上次的进度。只在当前会话内生效，不写入浏览器存储，
 * 刷新页面或退出登录后清空。
 */
export type DraftStore<T extends object> = {
  get: () => T;
  set: (patch: Partial<T>) => void;
  reset: () => void;
  subscribe: (listener: () => void) => () => void;
};

export function createDraftStore<T extends object>(initial: T): DraftStore<T> {
  let state = initial;
  const listeners = new Set<() => void>();
  const emit = () => listeners.forEach((listener) => listener());
  return {
    get: () => state,
    set: (patch) => {
      state = { ...state, ...patch };
      emit();
    },
    reset: () => {
      state = initial;
      emit();
    },
    subscribe: (listener) => {
      listeners.add(listener);
      return () => {
        listeners.delete(listener);
      };
    },
  };
}

/** 组件读取草稿的钩子；返回 [草稿状态, 局部更新函数]。 */
export function useDraft<T extends object>(store: DraftStore<T>): [T, (patch: Partial<T>) => void] {
  const state = useSyncExternalStore(store.subscribe, store.get, store.get);
  return [state, store.set];
}

/** 「从音色库进入合成」的请求；token 每次点击递增，用于区分同音色的重复请求。 */
export type SynthesisRequest = { voiceId: string; voiceLabel?: string; token: number };

export type CreateVoiceReviewDraft = {
  segmentId: string;
  transcript: string;
  emotion: string;
  language: string;
  reason: string;
};

/** 向导当前正在执行的长操作；与 error 一起持久化，切走再回来仍能看到进度或失败原因。 */
export type CreateVoicePending = "upload" | "prepare" | "confirm" | "save" | null;

export type WizardAuxiliaryReference = {
  id: string;
  datasetId: string;
  effectiveSeconds: number;
  emotionLabel: string;
  transcript: string;
  fileName: string;
};

export type CreateVoiceDraft = {
  step: CreateVoiceStep;
  displayName: string;
  consent: boolean;
  file: File | null;
  dataset: DatasetSummary | null;
  preparedSeconds: number | null;
  segments: SegmentListResponse | null;
  review: CreateVoiceReviewDraft;
  auxiliaryReferences: WizardAuxiliaryReference[];
  pending: CreateVoicePending;
  error: SafeError | null;
};

export type SynthesisDraft = {
  selected: string;
  appliedRequestToken: number;
  text: string;
  language: SynthesisLanguage;
  synthesisMode: SynthesisMode;
  localOptions: LocalSynthesisOptions;
  emotionMode: "auto" | "manual";
  emotionLabel: EmotionLabel;
  strength: number;
  consent: boolean;
  cloudControlMode: "auto" | "label" | "description" | "vector";
  cloudEmotionLabel: CloudEmotionLabel;
  cloudDescription: string;
  cloudVector: number[];
  cloudVectorMode: "single" | "mixed";
  cloudSampleRate: 22050 | 44100 | 48000;
  cloudSpeed: number;
  cloudGain: number;
  cloudUseRandom: boolean;
  cloudIntervalSilence: number;
  cloudConsent: boolean;
  jobId: string | null;
};

export const createVoiceDraftStore = createDraftStore<CreateVoiceDraft>({
  step: "upload",
  displayName: "",
  consent: false,
  file: null,
  dataset: null,
  preparedSeconds: null,
  segments: null,
  review: { segmentId: "", transcript: "", emotion: "", language: "", reason: "" },
  auxiliaryReferences: [],
  pending: null,
  error: null,
});

export const synthesisDraftStore = createDraftStore<SynthesisDraft>({
  selected: "",
  appliedRequestToken: 0,
  text: "",
  language: "zh",
  synthesisMode: "local",
  localOptions: {
    cut_method: "none",
    speed: 1,
    pause_seconds: 0.3,
    top_k: 15,
    top_p: 1,
    temperature: 1,
  },
  emotionMode: "auto",
  emotionLabel: "neutral",
  strength: 0.65,
  consent: false,
  cloudControlMode: "label",
  cloudEmotionLabel: "happy",
  cloudDescription: "",
  cloudVector: [1, 0, 0, 0, 0, 0, 0, 0],
  cloudVectorMode: "single",
  cloudSampleRate: 44100,
  cloudSpeed: 1,
  cloudGain: 1,
  cloudUseRandom: false,
  cloudIntervalSilence: 200,
  cloudConsent: false,
  jobId: null,
});

export const taskFilterDraftStore = createDraftStore<{ status: JobStatus | undefined }>({
  status: undefined,
});

/** 退出登录时清空全部会话草稿，避免操作残留到下一个账号。 */
export function resetWorkspaceDrafts(): void {
  createVoiceDraftStore.reset();
  synthesisDraftStore.reset();
  taskFilterDraftStore.reset();
}
