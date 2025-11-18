import React from "react";
import ReactDOM from "react-dom/client";
import { createBrowserRouter, Navigate, RouterProvider } from "react-router-dom";

import App from "./App";
import "./index.css";

import DoctorDashboard from "./pages/DoctorDashboard";
import TechnicianUpload from "./pages/TechnicianUpload";

const router = createBrowserRouter([
  {
    path: "/",
    element: <App />,
    children: [
      { index: true, element: <Navigate to="/technician" replace /> },
      { path: "technician", element: <TechnicianUpload /> },
      { path: "doctor", element: <DoctorDashboard /> }
    ]
  }
]);

ReactDOM.createRoot(document.getElementById("root")!).render(
  <React.StrictMode>
    <RouterProvider router={router} />
  </React.StrictMode>
);
