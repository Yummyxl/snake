import { useCallback, useEffect, useRef, useState } from "react";
import { fetchStages } from "../api/stages.js";

function toErrMsg(e) {
  return e instanceof Error ? e.message : String(e);
}

export function useStageListData() {
  const [items, setItems] = useState([]);
  const [error, setError] = useState(null);
  const [loading, setLoading] = useState(true);
  const [lastOkAtMs, setLastOkAtMs] = useState(null);
  const hasSuccess = useRef(false);
  const [paused, setPausedState] = useState(false);
  const pausedRef = useRef(false);
  const inFlight = useRef(false);
  const failures = useRef(0);
  const abortRef = useRef(null);

  const setPaused = useCallback((v) => { pausedRef.current = v; setPausedState(v); }, []);
  const poll = useCallback(async (opts) => {
    const force = Boolean(opts?.force);
    if ((pausedRef.current && !force) || inFlight.current) return;
    inFlight.current = true;
    const ac = new AbortController();
    abortRef.current = ac;
    try {
      const data = await fetchStages(ac.signal);
      setItems(data);
      hasSuccess.current = true;
      failures.current = 0;
      setLastOkAtMs(Date.now());
      setError(null);
    } catch (e) {
      if (ac.signal.aborted) return;
      if (++failures.current >= 3) { setPaused(true); setError(toErrMsg(e)); }
    } finally {
      abortRef.current = null;
      inFlight.current = false;
      if (hasSuccess.current || failures.current >= 3) setLoading(false);
    }
  }, [setPaused]);

  useEffect(() => {
    poll();
    const t = window.setInterval(poll, 1000);
    return () => {
      window.clearInterval(t);
      abortRef.current?.abort();
    };
  }, [poll]);
  const retry = useCallback(() => { failures.current = 0; setPaused(false); setError(null); setLoading(true); poll(); }, [poll, setPaused]);
  const refresh = useCallback(() => poll({ force: true }), [poll]);
  return { items, error, loading, paused, retry, refresh, setPaused, lastOkAtMs };
}
