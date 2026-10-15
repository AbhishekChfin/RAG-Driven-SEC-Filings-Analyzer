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

// ============================================================================
// Supabase client (server-only)
// ============================================================================
const supabase = createClient(
  process.env.SUPABASE_URL!,
  process.env.SUPABASE_SERVICE_ROLE_KEY!,
  { auth: { persistSession: false, autoRefreshToken: false } }
);

// ============================================================================
// PIPELINE STEP 1: Validate & Normalize User Input
// ============================================================================
function validateAndNormalizeQuery(message: string): string {
  const normalized = (message ?? "").toString().trim();
  if (!normalized) {
    throw new Error("Empty query");
  }
  return normalized;
}

// ============================================================================
// PIPELINE STEP 2: Generate Query Embedding
// ============================================================================
async function embedQuery(query: string): Promise<number[]> {
  console.log("📊 Generating embedding for query...");
  const resp = await fetch(process.env.EMBEDDING_SERVICE_URL!, {
    method: "POST",
    headers: { "content-type": "application/json" },
    body: JSON.stringify({ text: query }),
  });
  
  if (!resp.ok) {
    throw new Error(`Embedding service failed: ${resp.statusText}`);
  }
  
  const data = await resp.json();
  console.log("✅ Embedding generated");
  return data.embedding;
}

// ============================================================================
// PIPELINE STEP 3: Retrieve Context from Vector Database
// ============================================================================
async function retrieveContext(queryEmbedding: number[]): Promise<any[]> {
  console.log("🔍 Retrieving similar chunks from Supabase...");
  
  const { data: chunks, error } = await supabase.rpc("match_documents", {
    query_embedding: queryEmbedding,
    match_count: 15,
  });

  if (error) {
    throw new Error(`Supabase query failed: ${error.message}`);
  }

  console.log(`✅ Retrieved ${chunks?.length ?? 0} chunks`);
  return chunks ?? [];
}

// ============================================================================
// PIPELINE STEP 4: Format Context for LLM
// ============================================================================
function formatContext(chunks: any[]): string {
  if (!chunks || chunks.length === 0) {
    return "";
  }

  const formatted = chunks
    .map((c: any, i: number) =>
      `[${i + 1}] (${c.metadata.company} ${c.metadata.year} ${c.metadata.item} ${c.chunk_index})\n${c.content}`
    )
    .join("\n\n");

  console.log("✅ Context formatted");
  return formatted;
}

// ============================================================================
// PIPELINE STEP 5: Build System Instructions
// ============================================================================
function buildSystemPrompt(): string {
  return `You are a financial analyst expert specializing in SEC 10-K filings and corporate earnings calls.

ROLE & EXPERTISE:
- Analyze financial data, trends, and disclosures from SEC filings
- Extract key metrics, risks, and strategic insights
- Provide evidence-based answers with precise citations

CORE INSTRUCTIONS:
1. ONLY use information from the provided context - do NOT use external knowledge
2. If information is NOT in the context, respond: "This information is not available in the provided filing"
3. Always cite your source: company name, fiscal year, and section/item referenced
4. For financial figures, include units ($ millions, %, basis points, etc.)
5. Be concise, accurate, and professional
6. For complex questions, break down your answer into logical steps
7. If information is unclear or conflicting, note the discrepancy

RESPONSE STYLE:
- Direct and factual
- No speculation or assumptions
- Include numerical precision
- Reference document location when citing data`;
}

// ============================================================================
// PIPELINE STEP 6: Query LLM for Answer (with Retry Logic)
// ============================================================================
async function queryGemini(
  systemPrompt: string,
  context: string,
  question: string,
  retries = 3
): Promise<string> {
  const url = `https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash:generateContent?key=${process.env.GEMINI_API_KEY}`;

  const requestBody = {
    contents: [{
      parts: [{
        text: `${systemPrompt}\n\n---FILING CONTEXT---\n${context}\n\n---QUESTION---\n${question}\n\n---ANSWER---`
      }]
    }],
    generationConfig: {
      temperature: 0.1,        // Lower = more factual (important for finance)
      maxOutputTokens: 1024,
      topP: 0.9,
    }
  };

  for (let attempt = 1; attempt <= retries; attempt++) {
    try {
      console.log(`🤖 Querying Gemini for answer (attempt ${attempt}/${retries})...`);
      
      const resp = await fetch(url, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(requestBody),
      });

      if (!resp.ok) {
        const errorJson = await resp.json();
        
        // Retry on 503 (Service Unavailable) or 429 (Rate Limit)
        if ((resp.status === 503 || resp.status === 429) && attempt < retries) {
          const delay = Math.pow(2, attempt - 1) * 1000; // Exponential backoff: 1s, 2s, 4s
          console.log(`⏳ Rate limited/overloaded. Retrying in ${delay}ms...`);
          await new Promise(r => setTimeout(r, delay));
          continue;
        }
        
        console.error("Gemini Error:", errorJson);
        throw new Error(`Gemini API Error ${resp.status}: ${errorJson.error?.message}`);
      }

      const data = await resp.json();
      const answer = data.candidates?.[0]?.content?.parts?.[0]?.text || "No response";
      console.log("✅ Answer generated");
      return answer;
      
    } catch (err: any) {
      if (attempt === retries) {
        throw err; // Throw on final attempt
      }
      console.log(`❌ Attempt ${attempt} failed, retrying...`);
    }
  }

  throw new Error("Failed to get response from Gemini after retries");
}

// ============================================================================
// MAIN HANDLER: Orchestrate the RAG Pipeline
// ============================================================================
export async function POST(req: NextRequest) {
  try {
    console.log("📨 Incoming chat request...");
    const body = await req.json();

    // STEP 1: Validate input
    const query = validateAndNormalizeQuery(body?.message);
    console.log(`📝 Query: "${query}"`);

    // STEP 2: Generate embedding for user query
    const queryEmbedding = await embedQuery(query);

    // STEP 3: Retrieve context from Supabase
    const chunks = await retrieveContext(queryEmbedding);

    // STEP 4: Format context
    const context = formatContext(chunks);

    // Check if we have any context
    if (!context) {
      return new Response(
        JSON.stringify({
          answer: "I couldn't find relevant information in the provided documents. Try rephrasing your question or asking about a different topic.",
          sources: []
        }),
        { status: 200, headers: { "content-type": "application/json" } }
      );
    }

    // STEP 5: Build system instructions
    const systemPrompt = buildSystemPrompt();

    // STEP 6: Query Gemini for answer
    const answer = await queryGemini(systemPrompt, context, query);

    console.log("✅ Response complete");

    // STEP 7: Return formatted response
    return new Response(
      JSON.stringify({
        answer,
        sources: chunks.map((c: any) => ({
          company: c.metadata.company,
          year: c.metadata.year,
          item: c.metadata.item,
          chunk_index: c.chunk_index,
          similarity: c.similarity,
          content: c.content
        }))
      }),
      { status: 200, headers: { "content-type": "application/json" } }
    );

  } catch (err: any) {
    console.error("❌ API Error:", err?.message || err);
    return new Response(
      JSON.stringify({ error: err?.message || "Unknown error" }),
      { status: 500, headers: { "content-type": "application/json" } }
    );
  }
}
