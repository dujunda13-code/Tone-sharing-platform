import { describe, expect, it } from "vitest";

import { createPreviewFetch } from "../preview/previewFetch";

describe("frontend preview API", () => {
  it("returns an authenticated preview user without a backend", async () => {
    const fetcher = createPreviewFetch();

    const response = await fetcher("http://127.0.0.1:8000/api/auth/me");
    expect(response.status).toBe(200);
    expect(await response.json()).toMatchObject({
      username: "preview",
      role: "admin",
      status: "active",
    });
  });

  it("provides ready dashboard data and a usable voice", async () => {
    const fetcher = createPreviewFetch();

    const dashboard = await (await fetcher("/api/dashboard")).json();
    const voices = await (await fetcher("/api/voices")).json();

    expect(dashboard.readiness.status).toBe("ready");
    expect(voices.items[0]).toMatchObject({ status: "ready", can_synthesize: true });
  });
});
