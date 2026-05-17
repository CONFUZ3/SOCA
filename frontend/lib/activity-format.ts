import type { ActivityEvent } from "@/types";

type ActivityStatus = ActivityEvent["status"];

const STAGE_LABELS: Array<[RegExp, string]> = [
  [/^fetch\.boundary$|^boundary\./, "AOI boundary"],
  [/^fetch\.population$|^population\./, "Population grid"],
  [/^fetch\.pois$|^pois\./, "Facility locations"],
  [/^optimization\.stage$/, "Optimization setup"],
  [/^solver\.run$/, "Solver"],
  [/^network\./, "Road network"],
  [/^geocode\./, "Location lookup"],
  [/^dataset\./, "Dataset"],
  [/^aoi\./, "Area of interest"],
];

function labelForStage(stage: string): string {
  const match = STAGE_LABELS.find(([pattern]) => pattern.test(stage));
  if (match) return match[1];
  return stage
    .split(".")
    .map((part) => part.replace(/_/g, " "))
    .join(" ");
}

function statusVerb(status: ActivityStatus): string {
  switch (status) {
    case "try":
      return "in progress";
    case "ok":
      return "ready";
    case "fail":
      return "failed";
    default:
      return "updated";
  }
}

export function formatActivityStage(evt: ActivityEvent): string {
  return labelForStage(evt.stage);
}

export function formatActivityHeadline(evt: ActivityEvent, status = evt.status): string {
  return `${formatActivityStage(evt)} ${statusVerb(status)}`;
}

export function formatActivityDetail(evt: ActivityEvent): string {
  const detail = (evt.detail || "").trim();
  if (detail) return detail;
  return formatActivityHeadline(evt);
}

export function activityGroupStatus(events: ActivityEvent[]): ActivityStatus {
  const last = events[events.length - 1];
  if (last?.status === "try") return "try";
  if (events.some((evt) => evt.status === "fail")) return "fail";
  if (last?.status === "ok" || events.some((evt) => evt.status === "ok")) return "ok";
  return "info";
}

export function primaryActivityEvent(events: ActivityEvent[]): ActivityEvent | null {
  for (let i = events.length - 1; i >= 0; i -= 1) {
    if (events[i].status !== "info") return events[i];
  }
  return events[events.length - 1] || null;
}
