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
      <div className="pageHeader">
        <div className="pageHeader__left">
          <div className="breadcrumbs">
            <span className="breadcrumbs__current">Stages</span>
          </div>
          <div className="pageTitleRow">
            <div className="pageTitle">Stage 列表</div>
          </div>
          <div className="pageMetaRow">
            <span className="mono">共 {orderedItems.length} 个</span>
            {trainingHint ? (
              <Link className="link" to={`/stages/${trainingHint.stage_id}`}>
                <span className="pill pill--blue pill--sm">Running: Stage {trainingHint.stage_id} / {String(trainingHint.phase).toUpperCase()}</span>
              </Link>
            ) : (
              <span className="pill pill--gray pill--sm">Stopped</span>
            )}
          </div>
        </div>
        <div className="actionBar" />
      </div>
      <StageCards items={orderedItems} loading={loading} />
    </div>
  );
}
