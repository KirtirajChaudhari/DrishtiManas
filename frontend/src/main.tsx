import React from "react";
import ReactDOM from "react-dom/client";
import { createBrowserRouter, RouterProvider } from "react-router-dom";

import App from "./App";
import RouteError from "./components/RouteError";
import ClassifyPage from "./pages/Classify";
import LearnPage from "./pages/Learn";
import ReportPage from "./pages/Report";
import "./index.css";

const router = createBrowserRouter([
  {
    path: "/",
    element: <App />,
    errorElement: <RouteError />,
    children: [
      { index: true, element: <ClassifyPage /> },
      { path: "model", element: <ReportPage /> },
      { path: "learn", element: <LearnPage /> }
    ]
  }
]);

ReactDOM.createRoot(document.getElementById("root")!).render(
  <React.StrictMode>
    <RouterProvider router={router} />
  </React.StrictMode>
);
