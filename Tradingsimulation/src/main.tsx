
  import { createRoot } from "react-dom/client";
import App from "./App.tsx";
import "./index.css";
import { ModelDataProvider } from "./context/modelData";

createRoot(document.getElementById("root")!).render(
  <ModelDataProvider>
    <App />
  </ModelDataProvider>
);
  