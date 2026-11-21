import { StrictMode, useCallback, useEffect, useState } from "react";
import { createRoot } from "react-dom/client";
import { BrowserRouter, NavLink, Navigate, Route, Routes, useNavigate } from "react-router-dom";
import { api, session, setUnauthorizedHandler, type User } from "./api";
import { AuthContext, isReviewer, useAuth } from "./auth";
import { Logo } from "./components/bits";
import Admin from "./pages/Admin";
import Login from "./pages/Login";
import Queue from "./pages/Queue";
import Review from "./pages/Review";
import Upload from "./pages/Upload";
import "./styles.css";

function Shell({ children, openReviews }: { children: React.ReactNode; openReviews: number | null }) {
  const { user, logout } = useAuth();
  return (
    <div className="shell">
      <nav className="rail" aria-label="Main">
        <div className="brand">
          <Logo /> FinDoc Guard
        </div>
        {isReviewer(user) && (
          <NavLink to="/" end>
            Review queue {openReviews ? <span className="count">{openReviews}</span> : null}
          </NavLink>
        )}
        <NavLink to="/upload">{isReviewer(user) ? "Upload" : "My documents"}</NavLink>
        {user?.role === "admin" && <NavLink to="/admin">Users and audit</NavLink>}
        <div className="who">
          <div>{user?.email}</div>
          <div className="muted" style={{ color: "#9aa6ba" }}>
            {user?.role}
          </div>
          <button type="button" onClick={logout}>
            Sign out
          </button>
        </div>
      </nav>
      <main>{children}</main>
    </div>
  );
}

function App() {
  const [user, setUser] = useState<User | null>(null);
  const [checked, setChecked] = useState(false);
  const [openReviews, setOpenReviews] = useState<number | null>(null);
  const navigate = useNavigate();

  const refresh = useCallback(async () => {
    if (!session.token) {
      setUser(null);
      setChecked(true);
      return;
    }
    try {
      setUser(await api.me());
    } catch {
      setUser(null);
    }
    setChecked(true);
  }, []);

  const logout = useCallback(() => {
    session.token = null;
    setUser(null);
    navigate("/login");
  }, [navigate]);

  useEffect(() => {
    setUnauthorizedHandler(() => {
      setUser(null);
      navigate("/login");
    });
    refresh();
  }, [refresh, navigate]);

  useEffect(() => {
    if (!isReviewer(user)) return;
    const load = () => api.stats().then((s) => setOpenReviews(s.open_reviews)).catch(() => {});
    load();
    const t = setInterval(load, 15000);
    return () => clearInterval(t);
  }, [user]);

  if (!checked) return null;
  const auth = { user, refresh, logout };
  return (
    <AuthContext.Provider value={auth}>
      <Routes>
        <Route path="/login" element={user ? <Navigate to="/" /> : <Login />} />
        <Route
          path="/*"
          element={
            !user ? (
              <Navigate to="/login" />
            ) : (
              <Shell openReviews={openReviews}>
                <Routes>
                  <Route path="/" element={isReviewer(user) ? <Queue /> : <Navigate to="/upload" />} />
                  <Route path="/upload" element={<Upload />} />
                  <Route path="/documents/:id" element={<Review />} />
                  <Route path="/admin" element={user.role === "admin" ? <Admin /> : <Navigate to="/" />} />
                  <Route path="*" element={<Navigate to="/" />} />
                </Routes>
              </Shell>
            )
          }
        />
      </Routes>
    </AuthContext.Provider>
  );
}

createRoot(document.getElementById("root")!).render(
  <StrictMode>
    <BrowserRouter>
      <App />
    </BrowserRouter>
  </StrictMode>,
);
