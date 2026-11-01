import type {
  AdherenceDay,
  ChatMessage,
  ExerciseSession,
  ExerciseSpec,
  Program,
  ScheduleSlot,
  ClinicalEvent,
  HandoverResult,
  TelemetrySummary,
} from "./types";

export class ApiError extends Error {
  status: number;

  constructor(message: string, status: number) {
    super(message);
    this.status = status;
  }
}

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  let response: Response;
  try {
    response = await fetch(path, {
      ...init,
      headers: { "Content-Type": "application/json", ...(init?.headers ?? {}) },
    });
  } catch {
    throw new ApiError("Cannot reach the Clinura engine. Check that it is running.", 0);
  }
  if (!response.ok) {
    let message = `Request failed (${response.status}).`;
    try {
      const body = (await response.json()) as { detail?: unknown };
      if (typeof body.detail === "string") message = body.detail;
      else if (Array.isArray(body.detail)) message = "Check the request and try again.";
    } catch {
      message = response.statusText || message;
    }
    throw new ApiError(message, response.status);
  }
  return (await response.json()) as T;
}

export interface AssistantStatus {
  configured: boolean;
  model: string;
}

let clinicianPin = "";
export function setClinicianPin(pin: string) {
  clinicianPin = pin;
}

function pinHeaders(): Record<string, string> {
  return clinicianPin ? { "X-Clinician-Pin": clinicianPin } : {};
}

function send<T>(path: string, method: string, body?: unknown): Promise<T> {
  return request<T>(path, {
    method,
    headers: pinHeaders(),
    body: body === undefined ? undefined : JSON.stringify(body),
  });
}

export const athletics = {
  library: () => request<ExerciseSpec[]>("/api/athletics/library"),
  programs: () => request<Program[]>("/api/athletics/programs"),
  createProgram: (data: Partial<Program>) => send<Program>("/api/athletics/programs", "POST", data),
  updateProgram: (id: number, data: Partial<Program>) =>
    send<Program>(`/api/athletics/programs/${id}`, "PUT", data),
  deleteProgram: (id: number) => send<{ ok: boolean }>(`/api/athletics/programs/${id}`, "DELETE"),
  addItem: (programId: number, data: Record<string, unknown>) =>
    send<Program>(`/api/athletics/programs/${programId}/items`, "POST", data),
  updateItem: (id: number, data: Record<string, unknown>) =>
    send<{ ok: boolean }>(`/api/athletics/items/${id}`, "PUT", data),
  deleteItem: (id: number) => send<{ ok: boolean }>(`/api/athletics/items/${id}`, "DELETE"),
  today: () => request<ScheduleSlot[]>("/api/athletics/today"),
  adherence: (days = 14) => request<AdherenceDay[]>(`/api/athletics/adherence?days=${days}`),
  sessions: (days = 30) => request<ExerciseSession[]>(`/api/athletics/sessions?days=${days}`),
  saveSession: (data: ExerciseSession) =>
    send<{ ok: boolean; id: number }>("/api/athletics/sessions", "POST", data),
  pinStatus: () => request<{ set: boolean }>("/api/athletics/pin"),
  verifyPin: (pin: string) => send<{ ok: boolean }>("/api/athletics/pin/verify", "POST", { pin }),
  changePin: (pin: string | null, current: string) =>
    send<{ set: boolean }>("/api/athletics/pin", "POST", { pin, current }),
};

export interface CameraInfoItem {
  index: number;
  name: string;
}

export const api = {
  cameras: () => request<{ active: number; cameras: CameraInfoItem[] }>("/api/cameras"),
  assistant: () => request<AssistantStatus>("/api/assistant"),
  telemetry: (hours: number) => request<TelemetrySummary>(`/api/telemetry?hours=${hours}`),
  events: (hours: number) => request<ClinicalEvent[]>(`/api/events?hours=${hours}`),
  addNote: (title: string, detail?: string) =>
    request<{ ok: boolean; id: number }>("/api/events", {
      method: "POST",
      body: JSON.stringify({ title, detail: detail || null }),
    }),
  handover: (hours: number, bed: string, patient: string) =>
    request<HandoverResult>("/api/handover", {
      method: "POST",
      body: JSON.stringify({ hours, bed: bed || null, patient: patient || null }),
    }),
  ask: (question: string, hours: number, history: ChatMessage[]) =>
    request<{ answer: string; model: string }>("/api/ask", {
      method: "POST",
      body: JSON.stringify({
        question,
        hours,
        history: history
          .filter((message) => !message.error)
          .map(({ role, content }) => ({ role, content })),
      }),
    }),
};
