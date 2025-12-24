import { Navigate, Route, Routes } from "react-router-dom";
import StageListPage from "./pages/StageListPage.jsx";
import StageDetailPage from "./pages/StageDetailPage.jsx";

export default function App() {
  return (
    <Routes>
      <Route path="/" element={<StageListPage />} />
      <Route path="/stages/:id" element={<StageDetailPage />} />
      <Route path="*" element={<Navigate to="/" replace />} />
    </Routes>
  );
}
