/**
 * lib/api.ts — Typed API client for SentioBot backend.
 *
 * All requests automatically attach the JWT token stored in localStorage.
 * Streaming chat uses the native fetch + ReadableStream API for true
 * token-by-token rendering.
 */

const BASE = process.env.NEXT_PUBLIC_API_URL ?? "http://localhost:8000";

// ---------------------------------------------------------------------------
// Types
// ---------------------------------------------------------------------------

export interface User {
  id: string;
  username: string;
  name: string;
  owned_products: { product_name: string; serial_number: string }[];
}

export interface Message {
  id: string;
  role: "user" | "assistant";
  content: string;
  metadata?: { sources?: Source[]; grounded?: Grounded; citations?: Citation[] };
  created_at: string;
}

export interface Source {
  source: string;
  section: string;
}

// F2 groundedness badge. Honest wording: this measures whether each claim matches
// a retrieved source passage (shown as citations), NOT factual correctness.
export interface Grounded {
  label: "grounded" | "partial" | "unverified";
  score: number;
  supported: number;
  total: number;
}

// F2 inline citation: the literal supporting source sentence behind a claim.
export interface Citation {
  n: number;
  source: string;
  section: string;
  span: string;
  claim?: string;
  similarity?: number;
}

export interface Conversation {
  id: string;
  title: string;
  created_at: string;
}

export type SSEEvent =
  | { type: "token"; data: string }
  | { type: "tool_start"; data: { name: string; input: string } }
  | { type: "tool_end"; data: { name: string; output: string } }
  | { type: "done"; data: { answer: string; sources: Source[]; grounded?: Grounded; citations?: Citation[]; cached?: boolean; interaction_id?: string } }
  | { type: "metrics"; data: Record<string, number | string | boolean> }
  | { type: "error"; data: { message: string } };

// ---------------------------------------------------------------------------
// Token storage
// ---------------------------------------------------------------------------

export const tokenStore = {
  get: () => (typeof window !== "undefined" ? localStorage.getItem("sentiobot_token") : null),
  set: (t: string) => localStorage.setItem("sentiobot_token", t),
  clear: () => localStorage.removeItem("sentiobot_token"),
};

// ---------------------------------------------------------------------------
// Base fetch helper
// ---------------------------------------------------------------------------

async function apiFetch<T>(path: string, init: RequestInit = {}): Promise<T> {
  const token = tokenStore.get();
  const headers: Record<string, string> = {
    "Content-Type": "application/json",
    ...(token ? { Authorization: `Bearer ${token}` } : {}),
    ...(init.headers as Record<string, string>),
  };

  const res = await fetch(`${BASE}${path}`, { ...init, headers });

  if (!res.ok) {
    const err = await res.json().catch(() => ({ detail: res.statusText }));
    throw new Error(err.detail ?? "Request failed");
  }

  return res.json() as Promise<T>;
}

// ---------------------------------------------------------------------------
// Auth
// ---------------------------------------------------------------------------

export async function login(username: string, password: string): Promise<User> {
  const form = new URLSearchParams({ username, password });
  const res = await fetch(`${BASE}/auth/login`, {
    method: "POST",
    headers: { "Content-Type": "application/x-www-form-urlencoded" },
    body: form.toString(),
  });

  if (!res.ok) {
    const err = await res.json().catch(() => ({}));
    throw new Error(err.detail ?? "Login failed");
  }

  const data = await res.json();
  tokenStore.set(data.access_token);
  return data.user as User;
}

export async function getMe(): Promise<User> {
  return apiFetch<User>("/auth/me");
}

// ---------------------------------------------------------------------------
// Conversations
// ---------------------------------------------------------------------------

export async function listConversations(): Promise<Conversation[]> {
  return apiFetch<Conversation[]>("/chat/conversations");
}

export async function createConversation(title: string): Promise<Conversation> {
  return apiFetch<Conversation>("/chat/conversations", {
    method: "POST",
    body: JSON.stringify({ title }),
  });
}

export async function getMessages(conversationId: string): Promise<Message[]> {
  return apiFetch<Message[]>(`/chat/conversations/${conversationId}/messages`);
}

// ---------------------------------------------------------------------------
// Streaming chat
// ---------------------------------------------------------------------------

export async function* streamChat(
  message: string,
  conversationId: string | null,
  onConversationCreated?: (id: string) => void
): AsyncGenerator<SSEEvent> {
  const token = tokenStore.get();
  const res = await fetch(`${BASE}/chat/stream`, {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
      ...(token ? { Authorization: `Bearer ${token}` } : {}),
    },
    body: JSON.stringify({ message, conversation_id: conversationId }),
  });

  if (!res.ok) throw new Error("Chat request failed");

  // Capture conversation ID from response headers
  const convId = res.headers.get("X-Conversation-Id");
  if (convId && !conversationId) onConversationCreated?.(convId);

  const reader = res.body!.getReader();
  const decoder = new TextDecoder();
  let buffer = "";

  while (true) {
    const { done, value } = await reader.read();
    if (done) break;

    buffer += decoder.decode(value, { stream: true });
    const lines = buffer.split("\n");
    buffer = lines.pop() ?? "";

    for (const line of lines) {
      if (line.startsWith("data: ")) {
        try {
          const event = JSON.parse(line.slice(6)) as SSEEvent;
          yield event;
        } catch {
          // skip malformed line
        }
      }
    }
  }
}

// ---------------------------------------------------------------------------
// Feedback
// ---------------------------------------------------------------------------

export async function submitFeedback(interactionId: string, feedback: 1 | -1): Promise<void> {
  await apiFetch("/feedback", {
    method: "POST",
    body: JSON.stringify({ interaction_id: interactionId, feedback }),
  });
}
