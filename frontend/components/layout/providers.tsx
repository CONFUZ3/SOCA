"use client";

import {
  QueryClient,
  QueryClientProvider,
} from "@tanstack/react-query";
import { TooltipProvider } from "@radix-ui/react-tooltip";
import { useState } from "react";

export function Providers({ children }: { children: React.ReactNode }) {
  const [qc] = useState(
    () =>
      new QueryClient({
        defaultOptions: {
          queries: {
            staleTime: 15_000,
            refetchOnWindowFocus: false,
            retry: (count, err) => {
              // Don't retry on 4xx.
              if (
                err &&
                typeof err === "object" &&
                "status" in err &&
                typeof (err as { status: number }).status === "number" &&
                (err as { status: number }).status >= 400 &&
                (err as { status: number }).status < 500
              ) {
                return false;
              }
              return count < 2;
            },
          },
        },
      }),
  );

  return (
    <QueryClientProvider client={qc}>
      <TooltipProvider delayDuration={200}>{children}</TooltipProvider>
    </QueryClientProvider>
  );
}
