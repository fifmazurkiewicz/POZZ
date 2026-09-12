import { act, renderHook } from "@testing-library/react";
import { afterEach, describe, expect, it, vi } from "vitest";
import { ApiError } from "@/lib/api";
import { useAbortableAction } from "@/lib/useAbortableAction";

afterEach(() => {
  vi.restoreAllMocks();
});

describe("useAbortableAction", () => {
  it("exposes busy=false and error=null initially", () => {
    const { result } = renderHook(() => useAbortableAction());
    expect(result.current.busy).toBe(false);
    expect(result.current.error).toBe(null);
  });

  it("clears busy and error on success", async () => {
    const { result } = renderHook(() => useAbortableAction());
    await act(async () => {
      await result.current.run(async () => ({ ok: true }), () => undefined);
    });
    expect(result.current.busy).toBe(false);
    expect(result.current.error).toBe(null);
  });

  it("surfaces ApiError.message verbatim on provider failure", async () => {
    const { result } = renderHook(() => useAbortableAction());
    await act(async () => {
      await result.current.run(
        async () => {
          throw new ApiError("Nie udało się wykonać operacji. Spróbuj ponownie.", 502, "provider_error");
        },
        () => undefined
      );
    });
    expect(result.current.error).toBe("Nie udało się wykonać operacji. Spróbuj ponownie.");
    expect(result.current.busy).toBe(false);
  });

  it("uses fallback message for network TypeError", async () => {
    const { result } = renderHook(() => useAbortableAction());
    await act(async () => {
      await result.current.run(
        async () => {
          throw new TypeError("Failed to fetch");
        },
        () => undefined
      );
    });
    expect(result.current.error).toBe("Nie udało się wykonać operacji. Spróbuj ponownie.");
  });

  it("does not surface the fallback when the user explicitly cancels", async () => {
    const { result } = renderHook(() => useAbortableAction());
    let cancelSeen = false;
    const promise = result.current.run(
      async (signal) =>
        new Promise<{ aborted: boolean }>((_, reject) => {
          signal.addEventListener("abort", () => {
            cancelSeen = true;
            reject(new DOMException("Aborted", "AbortError"));
          });
        }),
      () => undefined
    );
    await act(async () => {
      result.current.cancel();
      await promise;
    });
    expect(cancelSeen).toBe(true);
    expect(result.current.error).toBe(null);
    expect(result.current.busy).toBe(false);
    expect(result.current.cancelled).toBe(true);
  });

  it("flags cancelled=false on natural completion", async () => {
    const { result } = renderHook(() => useAbortableAction());
    await act(async () => {
      await result.current.run(async () => ({ ok: true }), () => undefined);
    });
    expect(result.current.cancelled).toBe(false);
  });

  it("exposes the latest ApiError.code via getError()", async () => {
    const { result } = renderHook(() => useAbortableAction());
    await act(async () => {
      await result.current.run(
        async () => { throw new ApiError("provider error", 502, "provider_error"); },
        () => undefined
      );
    });
    expect(result.current.getError()?.code).toBe("provider_error");
    expect(result.current.error).toBe("provider error");
  });

  it("aborts in-flight request when a second run is started", async () => {
    const { result } = renderHook(() => useAbortableAction());
    const first = vi.fn();
    const second = vi.fn();
    const firstPromise = result.current.run(
      async (signal) =>
        new Promise<void>((resolve) => {
          signal.addEventListener("abort", () => {
            first();
            resolve();
          });
        }),
      () => undefined
    );
    await act(async () => {
      await result.current.run(async () => {
        second();
        return null;
      }, () => undefined);
    });
    await act(async () => {
      await firstPromise;
    });
    expect(first).toHaveBeenCalledTimes(1);
    expect(second).toHaveBeenCalledTimes(1);
  });
});