import { Link } from "react-router-dom";
import Banner from "../components/Banner.jsx";
import StageCards from "../components/StageCards.jsx";
import { useStagesPolling } from "../hooks/useStagesPolling.js";
import { useMemo } from "react";

export default function StageListPage() {
  const { items, error, loading, retry, trainingHint } = useStagesPolling();
  const orderedItems = useMemo(
    () => (Array.isArray(items) ? items.slice().sort((a, b) => Number(a?.stage_id) - Number(b?.stage_id)) : []),
    [items],
  );
  return (
    <div className="page">
      {error ? (
        <Banner kind="error" action={<button className="btn" onClick={retry} type="button">重试</button>}>
          后端不可用：{error}
        </Banner>
      ) : null}
      {trainingHint ? (
        <Banner
          kind="info"
          action={<Link className="link-inline" to={`/stages/${trainingHint.stage_id}`}>查看</Link>}
        >
          正在训练：Stage {trainingHint.stage_id} / {trainingHint.phase}
        </Banner>
      ) : null}
      <header className="header"><div className="brand">chichi</div><div className="title">Stage 列表</div></header>
      <StageCards items={orderedItems} loading={loading} />
    </div>
  );
}
