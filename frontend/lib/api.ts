/**
 * Typed fetch wrappers for the SOCA backend.
 * Same-origin relative URLs — Next rewrites /api/* to the FastAPI process.
 */

export class ApiError extends Error {
  readonly status: number;
  readonly detail: unknown;
  constructor(status: number, detail: unknown, message?: string) {
    super(message || `API ${status}`);
    this.status = status;
    this.detail = detail;
    this.name = "ApiError";
  }
}

async function jsonOrThrow<T>(res: Response): Promise<T> {
  const ct = res.headers.get("content-type") || "";
  const body = ct.includes("application/json") ? await res.json() : await res.text();
  if (!res.ok) {
    const detail =
      typeof body === "object" && body && "detail" in body
        ? (body as { detail: unknown }).detail
        : body;
    throw new ApiError(res.status, detail, `${res.status} ${res.statusText}`);
  }
  return body as T;
}

const DEFAULT_INIT: RequestInit = {
  credentials: "include",
  headers: { "Content-Type": "application/json" },
};

export async function apiGet<T>(path: string): Promise<T> {
  const res = await fetch(path, { ...DEFAULT_INIT, method: "GET" });
  return jsonOrThrow<T>(res);
}

export async function apiPost<T>(
  path: string,
  body?: unknown,
): Promise<T> {
  const res = await fetch(path, {
    ...DEFAULT_INIT,
    method: "POST",
    body: body !== undefined ? JSON.stringify(body) : undefined,
  });
  return jsonOrThrow<T>(res);
}

export async function apiDelete<T>(path: string): Promise<T> {
  const res = await fetch(path, { ...DEFAULT_INIT, method: "DELETE" });
  return jsonOrThrow<T>(res);
}

export async function apiUpload<T>(
  path: string,
  files: File[],
): Promise<T> {
  const fd = new FormData();
  for (const f of files) fd.append("files", f, f.name);
  const res = await fetch(path, {
    method: "POST",
    credentials: "include",
    body: fd,
  });
  return jsonOrThrow<T>(res);
}
