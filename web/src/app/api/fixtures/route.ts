const predictorUrl = process.env.PREDICTOR_API_URL ?? "http://127.0.0.1:8000";

export const dynamic = "force-dynamic";

export async function GET() {
  try {
    const response = await fetch(`${predictorUrl}/fixtures`, {
      cache: "no-store",
      signal: AbortSignal.timeout(5_000),
    });
    const payload: unknown = await response.json();
    return Response.json(payload, { status: response.status });
  } catch {
    return Response.json(
      { error: "The fixture schedule is unavailable. Please try again." },
      { status: 503 },
    );
  }
}
