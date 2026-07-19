"use client";

/**
 * app/chat/page.tsx — Main chat interface.
 *
 * Features:
 *  - Token-level streaming (tokens appear as they are generated)
 *  - Live "tool call" badges (e.g. "🔧 Checking warranty…")
 *  - Conversation sidebar with history
 *  - Feedback buttons per message (👍 / 👎)
 *  - Markdown rendering with syntax highlighting
 *  - Auto-scroll to latest message
 *  - Keyboard shortcut: Cmd/Ctrl+Enter to send
 */

import { useState, useEffect, useRef, useCallback } from "react";
import ReactMarkdown from "react-markdown";
import remarkGfm from "remark-gfm";
import {
  Send,
  Plus,
  MessageSquare,
  ThumbsUp,
  ThumbsDown,
  LogOut,
  Zap,
  Wrench,
  ChevronRight,
  ShieldCheck,
  ShieldAlert,
  ShieldQuestion,
} from "lucide-react";
import {
  streamChat,
  getMessages,
  listConversations,
  createConversation,
  submitFeedback,
  tokenStore,
  type Message,
  type Conversation,
  type Source,
  type Grounded,
  type Citation,
  type SSEEvent,
} from "../../lib/api";
import { useRouter } from "next/navigation";

// ---------------------------------------------------------------------------
// Types
// ---------------------------------------------------------------------------

interface UIMessage {
  id: string;
  role: "user" | "assistant";
  content: string;
  sources?: Source[];
  grounded?: Grounded;
  citations?: Citation[];
  isStreaming?: boolean;
  toolCalls?: { name: string; status: "running" | "done"; output?: string }[];
  interactionId?: string;
  feedback?: 1 | -1 | 0;
}

// F2 groundedness badge styling + HONEST wording (ratified: never say
// "verified"/"correct"; the badge claims source MATCH, and the citations shown
// below are the real check the user makes).
const GROUNDED_STYLE: Record<
  Grounded["label"],
  { cls: string; label: string; title: string; Icon: typeof ShieldCheck }
> = {
  grounded: {
    cls: "bg-green-950 border-green-800 text-green-400",
    label: "Grounded",
    title:
      "Every claim matches a passage in the cited sources (shown below). This checks source support, not factual correctness.",
    Icon: ShieldCheck,
  },
  partial: {
    cls: "bg-amber-950 border-amber-800 text-amber-400",
    label: "Partially grounded",
    title:
      "Some claims matched a source passage; others could not be matched. Please check the sources below.",
    Icon: ShieldAlert,
  },
  unverified: {
    cls: "bg-gray-800 border-gray-600 text-gray-400",
    label: "Unverified",
    title: "No supporting source passages were matched for this answer.",
    Icon: ShieldQuestion,
  },
};

function GroundednessBadge({ grounded }: { grounded: Grounded }) {
  const g = GROUNDED_STYLE[grounded.label] ?? GROUNDED_STYLE.unverified;
  const Icon = g.Icon;
  return (
    <span
      className={`inline-flex items-center gap-1 text-xs px-2 py-0.5 rounded-full border w-fit ${g.cls}`}
      title={g.title}
    >
      <Icon size={11} />
      {g.label}
      {grounded.total > 0 && (
        <span className="opacity-70">
          {grounded.supported}/{grounded.total}
        </span>
      )}
    </span>
  );
}

// ---------------------------------------------------------------------------
// Component
// ---------------------------------------------------------------------------

export default function ChatPage() {
  const router = useRouter();
  const [messages, setMessages] = useState<UIMessage[]>([]);
  const [input, setInput] = useState("");
  const [isLoading, setIsLoading] = useState(false);
  const [conversations, setConversations] = useState<Conversation[]>([]);
  const [activeConvId, setActiveConvId] = useState<string | null>(null);
  const [userName, setUserName] = useState("Guest");
  const bottomRef = useRef<HTMLDivElement>(null);
  const textareaRef = useRef<HTMLTextAreaElement>(null);

  // Auto-scroll
  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages]);

  // Load user + conversations on mount
  useEffect(() => {
    const token = tokenStore.get();
    if (!token) { router.push("/login"); return; }

    // Load name from localStorage (set during login)
    const name = localStorage.getItem("sentiobot_name");
    if (name) setUserName(name);

    listConversations()
      .then(setConversations)
      .catch(() => router.push("/login"));
  }, [router]);

  // Load messages when switching conversation
  const loadConversation = useCallback(async (convId: string) => {
    setActiveConvId(convId);
    const msgs = await getMessages(convId);
    setMessages(
      msgs.map((m) => ({
        id: m.id,
        role: m.role,
        content: m.content,
        sources: m.metadata?.sources,
        grounded: m.metadata?.grounded,
        citations: m.metadata?.citations,
      }))
    );
  }, []);

  // Send message
  const handleSend = useCallback(async () => {
    const text = input.trim();
    if (!text || isLoading) return;

    setInput("");
    setIsLoading(true);

    // Optimistically add user message
    const userMsgId = `user-${Date.now()}`;
    setMessages((prev) => [...prev, { id: userMsgId, role: "user", content: text }]);

    // Placeholder assistant message
    const assistantMsgId = `assistant-${Date.now()}`;
    setMessages((prev) => [
      ...prev,
      { id: assistantMsgId, role: "assistant", content: "", isStreaming: true, toolCalls: [] },
    ]);

    let convId = activeConvId;

    try {
      for await (const event of streamChat(text, convId, (newId) => {
        convId = newId;
        setActiveConvId(newId);
        // Refresh sidebar
        listConversations().then(setConversations);
      })) {
        handleSSEEvent(event, assistantMsgId);
      }
    } catch (err) {
      setMessages((prev) =>
        prev.map((m) =>
          m.id === assistantMsgId
            ? { ...m, content: "Sorry, something went wrong. Please try again.", isStreaming: false }
            : m
        )
      );
    } finally {
      setIsLoading(false);
    }
  }, [input, isLoading, activeConvId]);

  const handleSSEEvent = (event: SSEEvent, msgId: string) => {
    setMessages((prev) =>
      prev.map((m) => {
        if (m.id !== msgId) return m;

        switch (event.type) {
          case "token":
            return { ...m, content: m.content + event.data };

          case "tool_start":
            return {
              ...m,
              toolCalls: [
                ...(m.toolCalls ?? []),
                { name: event.data.name, status: "running" },
              ],
            };

          case "tool_end":
            return {
              ...m,
              toolCalls: (m.toolCalls ?? []).map((tc) =>
                tc.name === event.data.name
                  ? { ...tc, status: "done", output: event.data.output }
                  : tc
              ),
            };

          case "done":
            return {
              ...m,
              content: event.data.answer || m.content,
              sources: event.data.sources,
              grounded: event.data.grounded,
              citations: event.data.citations,
              interactionId: event.data.interaction_id ?? m.interactionId,
              isStreaming: false,
            };

          case "error":
            return { ...m, content: `⚠️ ${event.data.message}`, isStreaming: false };

          default:
            return m;
        }
      })
    );
  };

  const handleFeedback = async (msgId: string, feedback: 1 | -1) => {
    const msg = messages.find((m) => m.id === msgId);
    if (!msg?.interactionId) return;
    await submitFeedback(msg.interactionId, feedback);
    setMessages((prev) =>
      prev.map((m) => (m.id === msgId ? { ...m, feedback } : m))
    );
  };

  const handleLogout = () => {
    tokenStore.clear();
    router.push("/login");
  };

  const handleKeyDown = (e: React.KeyboardEvent) => {
    if ((e.metaKey || e.ctrlKey) && e.key === "Enter") handleSend();
  };

  return (
    <div className="flex h-screen bg-gray-950 text-gray-100">
      {/* ------------------------------------------------------------------ */}
      {/* Sidebar                                                              */}
      {/* ------------------------------------------------------------------ */}
      <aside className="w-64 flex flex-col bg-gray-900 border-r border-gray-800 shrink-0">
        {/* Brand */}
        <div className="flex items-center gap-2 p-4 border-b border-gray-800">
          <Zap className="text-cyan-400" size={22} />
          <span className="font-bold text-white">SentioBot</span>
        </div>

        {/* New chat */}
        <div className="p-3">
          <button
            onClick={async () => {
              const conv = await createConversation("New Conversation");
              setConversations((prev) => [conv, ...prev]);
              setActiveConvId(conv.id);
              setMessages([]);
            }}
            className="w-full flex items-center gap-2 px-3 py-2 rounded-lg bg-cyan-600 hover:bg-cyan-500 text-white text-sm font-medium transition-colors"
          >
            <Plus size={16} />
            New Chat
          </button>
        </div>

        {/* Conversation list */}
        <nav className="flex-1 overflow-y-auto px-2 space-y-1">
          {conversations.map((conv) => (
            <button
              key={conv.id}
              onClick={() => loadConversation(conv.id)}
              className={`w-full flex items-center gap-2 px-3 py-2 rounded-lg text-sm text-left transition-colors ${
                activeConvId === conv.id
                  ? "bg-gray-700 text-white"
                  : "text-gray-400 hover:bg-gray-800 hover:text-white"
              }`}
            >
              <MessageSquare size={14} className="shrink-0" />
              <span className="truncate">{conv.title}</span>
            </button>
          ))}
        </nav>

        {/* User / logout */}
        <div className="p-3 border-t border-gray-800 flex items-center justify-between">
          <span className="text-sm text-gray-400">👤 {userName}</span>
          <button onClick={handleLogout} className="text-gray-500 hover:text-red-400 transition-colors">
            <LogOut size={16} />
          </button>
        </div>
      </aside>

      {/* ------------------------------------------------------------------ */}
      {/* Main chat area                                                       */}
      {/* ------------------------------------------------------------------ */}
      <div className="flex flex-col flex-1 min-w-0">
        {/* Header */}
        <header className="px-6 py-3 border-b border-gray-800 flex items-center gap-2">
          <Zap className="text-cyan-400" size={18} />
          <span className="font-semibold text-sm text-gray-200">Nexora Electronics Support</span>
          <span className="ml-auto text-xs text-gray-600">⌘+Enter to send</span>
        </header>

        {/* Messages */}
        <div className="flex-1 overflow-y-auto px-4 py-6 space-y-6">
          {messages.length === 0 && (
            <div className="flex flex-col items-center justify-center h-full text-center text-gray-600 space-y-3">
              <Zap size={48} className="text-cyan-800" />
              <p className="text-lg font-medium text-gray-400">How can I help you today?</p>
              <p className="text-sm max-w-sm">
                Ask me about Nexora products, check your warranty, track an order, or troubleshoot an issue.
              </p>
            </div>
          )}

          {messages.map((msg) => (
            <div key={msg.id} className={`flex gap-3 ${msg.role === "user" ? "justify-end" : "justify-start"}`}>
              {msg.role === "assistant" && (
                <div className="w-8 h-8 rounded-full bg-cyan-900 flex items-center justify-center shrink-0 mt-1">
                  <Zap size={14} className="text-cyan-300" />
                </div>
              )}

              <div className={`max-w-2xl space-y-2 ${msg.role === "user" ? "items-end" : "items-start"} flex flex-col`}>
                {/* Tool call badges */}
                {msg.toolCalls && msg.toolCalls.length > 0 && (
                  <div className="flex flex-wrap gap-2">
                    {msg.toolCalls.map((tc, i) => (
                      <span
                        key={i}
                        className={`inline-flex items-center gap-1 text-xs px-2 py-1 rounded-full border ${
                          tc.status === "running"
                            ? "bg-yellow-950 border-yellow-700 text-yellow-400 animate-pulse"
                            : "bg-green-950 border-green-700 text-green-400"
                        }`}
                      >
                        <Wrench size={10} />
                        {tc.status === "running" ? `Calling ${tc.name}…` : `${tc.name} ✓`}
                      </span>
                    ))}
                  </div>
                )}

                {/* Message bubble */}
                <div
                  className={`rounded-2xl px-4 py-3 text-sm leading-relaxed ${
                    msg.role === "user"
                      ? "bg-cyan-700 text-white rounded-tr-sm"
                      : "bg-gray-800 text-gray-100 rounded-tl-sm"
                  }`}
                >
                  {msg.role === "assistant" ? (
                    <div className="prose prose-invert prose-sm max-w-none">
                      <ReactMarkdown remarkPlugins={[remarkGfm]}>{msg.content}</ReactMarkdown>
                      {msg.isStreaming && (
                        <span className="inline-block w-2 h-4 bg-cyan-400 animate-pulse ml-1 rounded-sm" />
                      )}
                    </div>
                  ) : (
                    <p>{msg.content}</p>
                  )}
                </div>

                {/* Groundedness badge (F2). Only for doc-grounded answers. */}
                {msg.role === "assistant" && !msg.isStreaming && msg.grounded && (
                  <GroundednessBadge grounded={msg.grounded} />
                )}

                {/* Sources + inline citation spans (F2). Spans render as TEXT
                    ({c.span}), which React escapes, so source markup cannot
                    inject HTML (XSS-safe); never dangerouslySetInnerHTML here. */}
                {msg.sources && msg.sources.length > 0 && (
                  <details className="text-xs text-gray-500 cursor-pointer">
                    <summary className="flex items-center gap-1 hover:text-gray-400">
                      <ChevronRight size={12} />
                      {msg.sources.length} source{msg.sources.length > 1 ? "s" : ""}
                    </summary>
                    <ul className="mt-1 ml-4 space-y-2">
                      {msg.sources.map((s, i) => {
                        const cites = (msg.citations ?? []).filter((c) => c.n === i + 1);
                        return (
                          <li key={i}>
                            <span className="text-cyan-700">{s.source}</span> - {s.section}
                            {cites.length > 0 && (
                              <ul className="mt-1 ml-1 space-y-1">
                                {cites.map((c, j) => (
                                  <li
                                    key={j}
                                    className="border-l-2 border-cyan-800 bg-cyan-950/30 pl-2 py-0.5 text-gray-400 italic"
                                  >
                                    &ldquo;{c.span}&rdquo;
                                  </li>
                                ))}
                              </ul>
                            )}
                          </li>
                        );
                      })}
                    </ul>
                  </details>
                )}

                {/* Feedback */}
                {msg.role === "assistant" && !msg.isStreaming && (
                  <div className="flex gap-2">
                    <button
                      onClick={() => handleFeedback(msg.id, 1)}
                      className={`p-1 rounded transition-colors ${
                        msg.feedback === 1 ? "text-green-400" : "text-gray-600 hover:text-green-500"
                      }`}
                    >
                      <ThumbsUp size={14} />
                    </button>
                    <button
                      onClick={() => handleFeedback(msg.id, -1)}
                      className={`p-1 rounded transition-colors ${
                        msg.feedback === -1 ? "text-red-400" : "text-gray-600 hover:text-red-500"
                      }`}
                    >
                      <ThumbsDown size={14} />
                    </button>
                  </div>
                )}
              </div>

              {msg.role === "user" && (
                <div className="w-8 h-8 rounded-full bg-gray-700 flex items-center justify-center shrink-0 mt-1 text-xs font-bold text-gray-300">
                  {userName[0]?.toUpperCase()}
                </div>
              )}
            </div>
          ))}

          <div ref={bottomRef} />
        </div>

        {/* Input area */}
        <div className="px-4 py-4 border-t border-gray-800">
          <div className="flex items-end gap-3 bg-gray-800 rounded-2xl px-4 py-3 border border-gray-700 focus-within:border-cyan-700 transition-colors">
            <textarea
              ref={textareaRef}
              value={input}
              onChange={(e) => setInput(e.target.value)}
              onKeyDown={handleKeyDown}
              placeholder="Ask about Nexora products, warranty, orders…"
              rows={1}
              className="flex-1 bg-transparent text-sm text-gray-100 placeholder-gray-600 resize-none outline-none max-h-32 overflow-y-auto"
              style={{ minHeight: "24px" }}
            />
            <button
              onClick={handleSend}
              disabled={!input.trim() || isLoading}
              className="p-2 rounded-xl bg-cyan-600 hover:bg-cyan-500 disabled:opacity-40 disabled:cursor-not-allowed transition-colors shrink-0"
            >
              <Send size={16} className="text-white" />
            </button>
          </div>
          <p className="text-center text-xs text-gray-700 mt-2">
            SentioBot can make mistakes. Verify important information.
          </p>
        </div>
      </div>
    </div>
  );
}
