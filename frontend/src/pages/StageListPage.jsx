import { Link } from "react-router-dom";
import Banner from "../components/Banner.jsx";
import StageCards from "../components/StageCards.jsx";
import { useStagesPolling } from "../hooks/useStagesPolling.js";

export default function StageListPage() {
  const { items, error, loading, retry, trainingHint } = useStagesPolling();
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
      <StageCards items={items} loading={loading} />
    </div>
  );
}
