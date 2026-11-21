import { createContext, useContext } from "react";
import type { User } from "./api";

export interface Auth {
  user: User | null;
  refresh: () => Promise<void>;
  logout: () => void;
}

export const AuthContext = createContext<Auth>({ user: null, refresh: async () => {}, logout: () => {} });
export const useAuth = () => useContext(AuthContext);
export const isReviewer = (u: User | null) => u?.role === "reviewer" || u?.role === "admin";
