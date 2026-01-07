import { NextRequest } from "next/server";
import { createClient } from "@supabase/supabase-js";

console.log("ENV CHECK:", {
  SUPABASE_URL: !!process.env.SUPABASE_URL,
  SERVICE_ROLE: !!process.env.SUPABASE_SERVICE_ROLE_KEY,
  EMBEDDING_URL: !!process.env.EMBEDDING_SERVICE_URL,
  GEMINI_KEY: !!process.env.GEMINI_API_KEY,
});


export const runtime = "nodejs";
export const dynamic = "force-dynamic"; // no caching

// Supabase client (server-only)
const supabase = createClient(
  process.env.SUPABASE_URL!,
  process.env.SUPABASE_SERVICE_ROLE_KEY!,
  { auth: { persistSession: false, autoRefreshToken: false } }
);

// 1️⃣ Query embedding from Python service (only for user query)
async function embedQuery(query: string) {
  const resp = await fetch(process.env.EMBEDDING_SERVICE_URL!, {
    method: "POST",
    headers: { "content-type": "application/json" },
    body: JSON.stringify({ text: query }),
  });
  if (!resp.ok) throw new Error("Embedding service failed");
  const data = await resp.json();
  return data.embedding;
}

async function queryGemini(context: string, question: string) {
  // Use gemini-2.5-flash (the current stable model)
  const url = `https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash:generateContent?key=${process.env.GEMINI_API_KEY}`;

  const resp = await fetch(url, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      contents: [{
        parts: [{
          text: `CONTEXT:\n${context}\n\nQUESTION: ${question}`
        }]
      }]
    }),
  });

  if (!resp.ok) {
    const errorJson = await resp.json();
    console.error("Gemini Error:", errorJson);
    throw new Error(`API Error ${resp.status}: ${errorJson.error?.message}`);
  }

  const data = await resp.json();
  return data.candidates?.[0]?.content?.parts?.[0]?.text || "No response";
}

export async function POST(req: NextRequest) {
  try {
    const body = await req.json();
    const message = (body?.message ?? "").toString().trim();

    if (!message) {
      return new Response(JSON.stringify({ error: "Empty query" }), {
        status: 400,
        headers: { "content-type": "application/json" },
      });
    }

    // 1️⃣ Embed the user query
    const queryEmb = await embedQuery(message);

    // 2️⃣ Retrieve top-k similar chunks from Supabase
    const { data: chunks, error } = await supabase.rpc("match_documents", {
      query_embedding: queryEmb,
      match_count: 8,
      // Optional: filter by metadata, e.g., company/year/item
      // filter: { company: "AAPL", year: 2019, item: "item2" }
    });

    if (error) throw error;

    // 3️⃣ Build context with friendly labels for each chunk
    const context = (chunks ?? [])
      .map((c: any, i: number) =>
        `[${i + 1}] (${c.metadata.company} ${c.metadata.year} ${c.metadata.item} ${c.chunk_index}) ${c.content}`
      )
      .join("\n\n");

    if (!context) {
      return new Response(JSON.stringify({
        answer: "I couldn’t find this in the provided document. Try rephrasing or asking a different section.",
        sources: []
      }), { status: 200, headers: { "content-type": "application/json" }});
    }

    // 4️⃣ Ask Gemini for completion
    const answer = await queryGemini(context, message);

    return new Response(JSON.stringify({
      answer,
      sources: chunks ?? []
    }), { status: 200, headers: { "content-type": "application/json" }});

  } catch (err: any) {
    console.error("api/chat error:", err?.message || err);
    return new Response(JSON.stringify({ error: err?.message || "Unknown error" }), {
      status: 500,
      headers: { "content-type": "application/json" },
    });
  }
}
