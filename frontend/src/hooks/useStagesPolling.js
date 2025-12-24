import { useStageListData } from "./useStageListData.js";
import { useTrainingHint } from "./useTrainingHint.js";

export function useStagesPolling() {
  const data = useStageListData();
  return { ...data, trainingHint: useTrainingHint(data.items) };
}

