const predictorUrl = process.env.PREDICTOR_API_URL ?? "http://127.0.0.1:8000";

export const dynamic = "force-dynamic";

export async function POST(request: Request) {
  try {
    const body = await request.text();
    if (body.length > 8_192) {
      return Response.json({ error: "The forecast request is too large." }, { status: 413 });
    }

    const response = await fetch(`${predictorUrl}/predict`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body,
      cache: "no-store",
      signal: AbortSignal.timeout(10_000),
    });
    const payload: unknown = await response.json();
    return Response.json(payload, { status: response.status });
  } catch {
    return Response.json(
      { error: "The prediction service is not responding. Start it and try again." },
      { status: 503 },
    );
  }
}
