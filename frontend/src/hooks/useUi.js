import { useContext } from "react";
import { UiContext } from "../components/UiProvider.jsx";

export function useUi() {
  const ctx = useContext(UiContext);
  if (!ctx) throw new Error("useUi must be used within UiProvider");
  return ctx;
}

