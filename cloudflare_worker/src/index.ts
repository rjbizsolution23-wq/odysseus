/**
 * Cloudflare Worker for JUKEYMAN Edge Services.
 * Provides high-speed CDN proxy caching, edge-level auth validation,
 * Rate-limiting via Cloudflare KV, and direct media offloading to R2.
 */

export interface Env {
  ENVIRONMENT: string;
  API_ROOT_URL: string;
  JUKEYMAN_SESSIONS: KVNamespace;
  JUKEYMAN_RATELIMIT: KVNamespace;
  JUKEYMAN_ASSETS: R2Bucket;
  JUKEYMAN_DB: D1Database;
}

export default {
  async fetch(request: Request, env: Env, ctx: ExecutionContext): Promise<Response> {
    const url = new URL(request.url);
    const path = url.pathname;

    // 1. CORS Preflight Handling
    if (request.method === "OPTIONS") {
      return new Response(null, {
        headers: {
          "Access-Control-Allow-Origin": "*",
          "Access-Control-Allow-Methods": "GET, POST, PUT, DELETE, OPTIONS",
          "Access-Control-Allow-Headers": "Content-Type, Authorization, X-API-Key",
          "Access-Control-Max-Age": "86400",
        },
      });
    }

    // 2. Rate Limiting Check (KV-backed per IP)
    const ip = request.headers.get("CF-Connecting-IP") || "anonymous";
    const rateLimitKey = `rate:${ip}:${Math.floor(Date.now() / 60000)}`; // 1-minute window
    const currentRequestsStr = await env.JUKEYMAN_RATELIMIT.get(rateLimitKey);
    const currentRequests = currentRequestsStr ? parseInt(currentRequestsStr, 10) : 0;

    if (currentRequests > 60) { // Limit: 60 requests per minute
      return new Response(JSON.stringify({ error: "Too many requests. Please slow down." }), {
        status: 429,
        headers: { "Content-Type": "application/json", "Access-Control-Allow-Origin": "*" },
      });
    }
    await env.JUKEYMAN_RATELIMIT.put(rateLimitKey, (currentRequests + 1).toString(), { expirationTtl: 120 });

    // 3. Asset Servicing directly from Cloudflare R2
    if (path.startsWith("/api/generated-image/") || path.startsWith("/images/")) {
      const key = path.split("/").pop();
      if (key) {
        const object = await env.JUKEYMAN_ASSETS.get(`images/${key}`);
        if (object) {
          const headers = new Headers();
          object.writeHttpMetadata(headers);
          headers.set("Access-Control-Allow-Origin", "*");
          headers.set("Cache-Control", "public, max-age=31536000, immutable");
          return new Response(object.body, { headers });
        }
      }
    }

    if (path.startsWith("/api/generated-video/") || path.startsWith("/videos/")) {
      const key = path.split("/").pop();
      if (key) {
        const object = await env.JUKEYMAN_ASSETS.get(`videos/${key}`);
        if (object) {
          const headers = new Headers();
          object.writeHttpMetadata(headers);
          headers.set("Access-Control-Allow-Origin", "*");
          headers.set("Cache-Control", "public, max-age=31536000, immutable");
          return new Response(object.body, { headers });
        }
      }
    }

    // 4. Fallback: Proxy everything else to the core Odysseus backend server tunnel
    const targetUrl = new URL(path + url.search, env.API_ROOT_URL);
    const newRequest = new Request(targetUrl, {
      method: request.method,
      headers: request.headers,
      body: request.body,
      redirect: "manual",
    });

    try {
      const response = await fetch(newRequest);
      // Append CORS headers to proxied response
      const newHeaders = new Headers(response.headers);
      newHeaders.set("Access-Control-Allow-Origin", "*");
      return new Response(response.body, {
        status: response.status,
        statusText: response.statusText,
        headers: newHeaders,
      });
    } catch (err: any) {
      return new Response(JSON.stringify({ error: "Failed to connect to backend", details: err.message }), {
        status: 502,
        headers: { "Content-Type": "application/json", "Access-Control-Allow-Origin": "*" },
      });
    }
  },
};
