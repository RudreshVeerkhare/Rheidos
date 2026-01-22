import { useEffect, useRef, useState } from "react";
import type { NodeDetails } from "../types";

export function useNodeDetails(selectedId: string | null) {
  const cacheRef = useRef<Map<string, NodeDetails>>(new Map());
  const [details, setDetails] = useState<NodeDetails | null>(null);
  const [loading, setLoading] = useState(false);

  useEffect(() => {
    if (selectedId === null || selectedId === undefined) {
      setDetails(null);
      setLoading(false);
      return;
    }

    const cached = cacheRef.current.get(selectedId);
    if (cached) {
      setDetails(cached);
      setLoading(false);
      return;
    }

    let cancelled = false;
    setLoading(true);

    fetch(`/api/node/${encodeURIComponent(selectedId)}`, { cache: "no-store" })
      .then((response) => (response.ok ? response.json() : null))
      .then((data) => {
        if (cancelled || !data) {
          return;
        }
        cacheRef.current.set(selectedId, data as NodeDetails);
        setDetails(data as NodeDetails);
      })
      .catch(() => {
        if (!cancelled) {
          setDetails(null);
        }
      })
      .finally(() => {
        if (!cancelled) {
          setLoading(false);
        }
      });

    return () => {
      cancelled = true;
    };
  }, [selectedId]);

  return { details, loading };
}
