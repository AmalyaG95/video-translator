import { NextRequest, NextResponse } from "next/server";

const API_URL = process.env.NEXT_PUBLIC_API_URL ?? "http://localhost:3001";

export async function GET(
  request: NextRequest,
  { params }: { params: Promise<{ path: string[] }> }
) {
  try {
    const resolvedParams = await params;
    const path = resolvedParams.path.join("/");
    const url = new URL(request.url);
    const searchParams = url.searchParams.toString();

    const backendUrl = `${API_URL}/translate/${path}${searchParams ? `?${searchParams}` : ""}`;

    const response = await fetch(backendUrl);

    if (!response.ok) {
      return new NextResponse("Video not found", { status: 404 });
    }

    const videoBuffer = await response.arrayBuffer();

    return new NextResponse(videoBuffer, {
      status: 200,
      headers: {
        "Content-Type": "video/mp4",
        "Content-Disposition": "inline",
        "Cache-Control": "no-cache",
      },
    });
  } catch (error) {
    console.error("Video proxy error:", error);
    return new NextResponse("Internal Server Error", { status: 500 });
  }
}
