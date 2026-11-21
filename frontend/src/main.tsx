import { Component, StrictMode, Suspense, lazy, useCallback, useEffect, useState, type ReactNode } from "react";
import { createRoot } from "react-dom/client";
import { BrowserRouter, NavLink, Navigate, Route, Routes, useLocation, useNavigate } from "react-router-dom";
import { api, session, setUnauthorizedHandler, type User } from "./api";
import { AuthContext, isReviewer, useAuth } from "./auth";
import { Logo } from "./components/bits";
import Login from "./pages/Login";
import Queue from "./pages/Queue";
import Review from "./pages/Review";
import Upload from "./pages/Upload";
import { CommandPalette, ShortcutsDialog, ToastProvider, useGlobalKeys, useTheme } from "./ui";
import "./styles.css";

// Pages most people never open are loaded on demand
const Overview = lazy(() => import("./pages/Overview"));
const Templates = lazy(() => import("./pages/Templates"));
const Admin = lazy(() => import("./pages/Admin"));

const icons: Record<string, ReactNode> = {
  queue: <path d="M3 4h12M3 9h12M3 14h7" />,
  overview: <path d="M3 15V9M8 15V4M13 15v-7" />,
  upload: <path d="M9 13V3M5 7l4-4 4 4M3 15h12" />,
  banks: <path d="M2 6l7-4 7 4M3 7v7M7 7v7M11 7v7M15 7v7M2 15h14" />,
  admin: <path d="M9 9a3 3 0 100-6 3 3 0 000 6zM3 16c0-3 2.7-5 6-5s6 2 6 5" />,
};
const Icon = ({ name }: { name: string }) => (
  <svg width="18" height="18" viewBox="0 0 18 18" fill="none" stroke="currentColor" strokeWidth="1.6" strokeLinecap="round" strokeLinejoin="round" aria-hidden="true">
    {icons[name]}
  </svg>
);

function Shell({ children, openReviews }: { children: ReactNode; openReviews: number | null }) {
  const { user, logout } = useAuth();
  const [theme, toggleTheme] = useTheme();
  const [palette, setPalette] = useState(false);
  const [help, setHelp] = useState(false);
  const openPalette = useCallback(() => setPalette(true), []);
  const openHelp = useCallback(() => setHelp(true), []);
  useGlobalKeys(openPalette, openHelp);
  const mac = /Mac|iPhone|iPad/.test(navigator.platform);

  return (
    <div className="shell">
      <a className="skip" href="#main">
        Skip to content
      </a>
      <nav className="rail" aria-label="Main">
        <div className="brand">
          <Logo /> FinDoc Guard
        </div>
        <button className="search" onClick={openPalette}>
          <svg width="14" height="14" viewBox="0 0 16 16" fill="none" stroke="currentColor" strokeWidth="1.8" aria-hidden="true">
            <circle cx="7" cy="7" r="5" />
            <path d="M11 11l4 4" />
          </svg>
          Search
          <kbd>{mac ? "⌘" : "Ctrl"} K</kbd>
        </button>
        {isReviewer(user) && (
          <>
            <NavLink to="/" end>
              <Icon name="queue" /> Review queue {openReviews ? <span className="count">{openReviews}</span> : null}
            </NavLink>
            <NavLink to="/overview">
              <Icon name="overview" /> Overview
            </NavLink>
          </>
        )}
        <NavLink to="/upload">
          <Icon name="upload" /> {isReviewer(user) ? "Upload" : "My documents"}
        </NavLink>
        {isReviewer(user) && (
          <NavLink to="/templates">
            <Icon name="banks" /> Cheque banks
          </NavLink>
        )}
        {user?.role === "admin" && (
          <NavLink to="/admin">
            <Icon name="admin" /> Users and audit
          </NavLink>
        )}
        <div className="who">
          <div className="avatar" aria-hidden="true">
            {(user?.name || user?.email || "?").charAt(0).toUpperCase()}
          </div>
          <div style={{ minWidth: 0 }}>
            <div className="email">{user?.email}</div>
            <div className="role">{user?.role}</div>
          </div>
          <div className="who-actions">
            <button type="button" onClick={toggleTheme} aria-label={`Switch to ${theme === "dark" ? "light" : "dark"} theme`} title="Theme">
              {theme === "dark" ? "☀" : "☾"}
            </button>
            <button type="button" onClick={openHelp} aria-label="Keyboard shortcuts" title="Shortcuts (?)">
              ?
            </button>
            <button type="button" onClick={logout} title="Sign out">
              Sign out
            </button>
          </div>
        </div>
      </nav>
      <main id="main" tabIndex={-1}>
        <ErrorBoundary>
          <Suspense fallback={<p className="muted">Loading…</p>}>{children}</Suspense>
        </ErrorBoundary>
      </main>
      {palette && <CommandPalette onClose={() => setPalette(false)} toggleTheme={toggleTheme} />}
      {help && <ShortcutsDialog onClose={() => setHelp(false)} />}
    </div>
  );
}

class ErrorBoundary extends Component<{ children: ReactNode }, { error: Error | null }> {
  state = { error: null as Error | null };
  static getDerivedStateFromError(error: Error) {
    return { error };
  }
  render() {
    if (this.state.error)
      return (
        <div className="error-box">
          <p>This screen failed to load: {this.state.error.message}</p>
          <button className="btn" style={{ marginTop: 8 }} onClick={() => location.reload()}>
            Reload
          </button>
        </div>
      );
    return this.props.children;
  }
}

/** Reset the error boundary and move focus to the page when the route changes. */
function RouteKey({ children }: { children: (key: string) => ReactNode }) {
  const loc = useLocation();
  useEffect(() => {
    document.getElementById("main")?.focus({ preventScroll: true });
  }, [loc.pathname]);
  return <>{children(loc.pathname)}</>;
}

function App() {
  const [user, setUser] = useState<User | null>(null);
  const [checked, setChecked] = useState(false);
  const [openReviews, setOpenReviews] = useState<number | null>(null);
  const navigate = useNavigate();
  useTheme(); // apply the saved theme on the sign-in page too

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
  return (
    <AuthContext.Provider value={{ user, refresh, logout }}>
      <ToastProvider>
        <Routes>
          <Route path="/login" element={user ? <Navigate to="/" /> : <Login />} />
          <Route
            path="/*"
            element={
              !user ? (
                <Navigate to="/login" />
              ) : (
                <Shell openReviews={openReviews}>
                  <RouteKey>
                    {(key) => (
                      <ErrorBoundary key={key}>
                        <Routes>
                          <Route path="/" element={isReviewer(user) ? <Queue /> : <Navigate to="/upload" />} />
                          <Route path="/overview" element={isReviewer(user) ? <Overview /> : <Navigate to="/upload" />} />
                          <Route path="/upload" element={<Upload />} />
                          <Route path="/documents/:id" element={<Review />} />
                          <Route path="/templates" element={isReviewer(user) ? <Templates /> : <Navigate to="/upload" />} />
                          <Route path="/admin" element={user.role === "admin" ? <Admin /> : <Navigate to="/" />} />
                          <Route path="*" element={<Navigate to="/" />} />
                        </Routes>
                      </ErrorBoundary>
                    )}
                  </RouteKey>
                </Shell>
              )
            }
          />
        </Routes>
      </ToastProvider>
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
