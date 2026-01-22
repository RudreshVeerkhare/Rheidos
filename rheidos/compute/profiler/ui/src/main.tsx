import React from "react";
import ReactDOM from "react-dom/client";
import { ReactFlowProvider } from "@xyflow/react";
import App from "./App";
import "@xyflow/react/dist/style.css";
import "./styles.css";

const root = document.getElementById("root");

if (root) {
  ReactDOM.createRoot(root).render(
    <React.StrictMode>
      <ReactFlowProvider>
        <App />
      </ReactFlowProvider>
    </React.StrictMode>
  );
}
