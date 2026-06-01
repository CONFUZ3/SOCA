import { useStore } from "@/lib/store";
import type { DatasetSummary } from "@/types";

// Monotonic per-dataset request counter, shared across every component that can
// toggle a dataset's subtype filter (chat picker + sidebar). Bumped
// synchronously when a request is issued so a slow/earlier response can never
// clobber the state set by a later click (which would re-add a subtype the user
// just removed). Only the most recent request's response is applied.
const filterSeq = new Map<string, number>();

/**
 * Compute the next active-subcategory set for toggling one subtype, reading the
 * latest dataset from the store (not a render closure) so rapid clicks don't
 * both derive `next` from the same pre-click state.
 */
export function nextToggled(name: string, sub: string): string[] {
  const ds = useStore.getState().datasets.find((d) => d.name === name);
  const available = ds?.available_subcategories ?? [];
  const current = ds?.active_subcategories ?? available;
  return current.includes(sub)
    ? current.filter((s) => s !== sub)
    : [...current, sub];
}

/**
 * Optimistically apply a subtype filter for `name`, persist it via PATCH, and
 * reconcile with the server's authoritative summary — but only if no newer
 * request for the same dataset has been issued in the meantime.
 */
export async function applySubcategoryFilter(
  name: string,
  next: string[],
): Promise<void> {
  const seq = (filterSeq.get(name) ?? 0) + 1;
  filterSeq.set(name, seq);
  const { updateDatasetSubcategories, updateDatasetSummary } =
    useStore.getState();
  updateDatasetSubcategories(name, next); // optimistic
  try {
    const resp = await fetch(`/api/data/${encodeURIComponent(name)}/filter`, {
      method: "PATCH",
      credentials: "include",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ active_subcategories: next }),
    });
    if (resp.ok && filterSeq.get(name) === seq) {
      const body = (await resp.json()) as DatasetSummary;
      updateDatasetSummary(body);
    }
  } catch {
    // optimistic update stays; map re-syncs on next poll / queryKey change
  }
}
