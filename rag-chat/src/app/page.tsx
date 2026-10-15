"use client";
import { useState } from "react";

export default function Home() {
  const [input, setInput] = useState("");
  const [messages, setMessages] = useState<{role:"user"|"assistant", content:string}[]>([]);
  const [sources, setSources] = useState<any[]>([]);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState("");

  async function send() {
    if (!input.trim()) return;
    setError("");
    setMessages(m => [...m, { role:"user", content: input }]);
    setBusy(true);

    try {
      const res = await fetch("/api/chat", {
        method:"POST",
        headers: { "Content-Type":"application/json" },
        body: JSON.stringify({ message: input })
      });
      
      if (!res.ok) {
        throw new Error(`API Error: ${res.status}`);
      }
      
      const data = await res.json();

      if (data.error) {
        setError(data.error);
        setMessages(m => m.slice(0, -1)); // Remove user message
      } else {
        setMessages(m => [...m, { role:"assistant", content: data.answer }]);
        setSources(data.sources || []);
      }
    } catch (err: any) {
      setError(err.message || "Failed to get response");
      setMessages(m => m.slice(0, -1)); // Remove user message
    }
    
    setInput("");
    setBusy(false);
  }

  return (
    <main className="min-h-screen bg-gradient-to-br from-slate-900 to-slate-800 p-6">
      <div className="max-w-4xl mx-auto">
        {/* Header */}
        <div className="bg-slate-800 rounded-lg shadow-md p-6 mb-6 border border-slate-700">
          <h1 className="text-4xl font-bold text-indigo-400 mb-2">📊 SEC 10-K Analyzer</h1>
          <p className="text-slate-300">Ask questions about Apple's SEC 10-K filings powered by AI</p>
        </div>

        {/* Chat Container */}
        <div className="bg-slate-800 rounded-lg shadow-lg overflow-hidden flex flex-col h-[600px] border border-slate-700">
          {/* Messages */}
          <div className="flex-1 overflow-y-auto p-6 space-y-4 bg-slate-900">
            {messages.length === 0 ? (
              <div className="text-center text-slate-400 mt-12">
                <p className="text-lg">💡 Start by asking questions about:</p>
                <p className="text-sm mt-2">Revenue, profitability, risks, business segments, financial metrics...</p>
              </div>
            ) : (
              messages.map((m, i) => (
                <div key={i} className={`flex ${m.role === "user" ? "justify-end" : "justify-start"}`}>
                  <div
                    className={`max-w-xs lg:max-w-md xl:max-w-lg px-4 py-3 rounded-lg ${
                      m.role === "user"
                        ? "bg-indigo-600 text-white rounded-br-none"
                        : "bg-slate-700 text-slate-100 rounded-bl-none"
                    }`}
                  >
                    <p className="text-sm whitespace-pre-wrap">{m.content}</p>
                  </div>
                </div>
              ))
            )}
            {busy && (
              <div className="flex justify-start">
                <div className="bg-slate-700 px-4 py-3 rounded-lg rounded-bl-none">
                  <div className="flex space-x-2">
                    <div className="w-2 h-2 bg-indigo-400 rounded-full animate-bounce"></div>
                    <div className="w-2 h-2 bg-indigo-400 rounded-full animate-bounce delay-100"></div>
                    <div className="w-2 h-2 bg-indigo-400 rounded-full animate-bounce delay-200"></div>
                  </div>
                </div>
              </div>
            )}
          </div>

          {/* Input Area */}
          <div className="border-t border-slate-700 bg-slate-800 p-4">
            {error && (
              <div className="bg-red-900 border border-red-700 text-red-200 px-4 py-2 rounded mb-3 text-sm">
                ⚠️ {error}
              </div>
            )}
            <div className="flex gap-3">
              <input
                className="flex-1 border border-slate-600 bg-slate-700 text-slate-100 rounded-lg px-4 py-2 focus:outline-none focus:border-indigo-500 focus:ring-2 focus:ring-indigo-400 placeholder-slate-400"
                value={input}
                onChange={e => setInput(e.target.value)}
                onKeyDown={e => e.key === "Enter" && !busy && send()}
                placeholder="e.g., What was Apple's revenue in 2023?"
                disabled={busy}
              />
              <button
                onClick={send}
                disabled={busy}
                className="bg-indigo-600 hover:bg-indigo-700 disabled:bg-gray-400 text-white px-6 py-2 rounded-lg font-medium transition"
              >
                {busy ? "..." : "Send"}
              </button>
            </div>
          </div>
        </div>

        {/* Sources */}
        {sources.length > 0 && (
          <div className="mt-6 bg-slate-800 rounded-lg shadow-md p-6 border border-slate-700">
            <h2 className="text-xl font-bold text-indigo-400 mb-4">📚 Source Documents</h2>
            <div className="grid gap-3">
              {sources.map((s, i) => (
                <div key={i} className="border border-slate-700 bg-slate-900 rounded-lg p-4 hover:bg-slate-800 transition">
                  <div className="flex justify-between items-start mb-2">
                    <div className="font-semibold text-indigo-400">
                      [{i + 1}] {s.company} {s.year} • {s.item}
                    </div>
                    <div className="text-xs bg-green-900 text-green-300 px-2 py-1 rounded">
                      {typeof s.similarity === "number" ? (s.similarity * 100).toFixed(0) : "?"}% match
                    </div>
                  </div>
                  <p className="text-xs text-slate-400 mb-2">Section: {s.chunk_index}</p>
                  <p className="text-sm text-slate-200 line-clamp-3">{s.content}</p>
                </div>
              ))}
            </div>
          </div>
        )}
      </div>
    </main>
  );
}

