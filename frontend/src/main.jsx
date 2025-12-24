import React from "react";
import ReactDOM from "react-dom/client";
import { BrowserRouter } from "react-router-dom";
import App from "./App.jsx";
import UiProvider from "./components/UiProvider.jsx";
import "./styles.css";

ReactDOM.createRoot(document.getElementById("root")).render(
  <UiProvider>
    <BrowserRouter>
      <App />
    </BrowserRouter>
  </UiProvider>,
);
