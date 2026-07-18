"use client";

import { useState } from "react";
import { useRouter } from "next/navigation";
import { Zap } from "lucide-react";
import { login } from "../../lib/api";

export default function LoginPage() {
  const router = useRouter();
  const [username, setUsername] = useState("");
  const [password, setPassword] = useState("");
  const [error, setError] = useState("");
  const [loading, setLoading] = useState(false);

  const handleLogin = async (e: React.FormEvent) => {
    e.preventDefault();
    setError("");
    setLoading(true);
    try {
      const user = await login(username, password);
      localStorage.setItem("sentiobot_name", user.name);
      router.push("/chat");
    } catch (err: unknown) {
      setError(err instanceof Error ? err.message : "Login failed");
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="min-h-screen bg-gray-950 flex items-center justify-center px-4">
      <div className="w-full max-w-sm">
        {/* Logo */}
        <div className="flex flex-col items-center mb-8 space-y-3">
          <div className="w-16 h-16 rounded-2xl bg-cyan-900 flex items-center justify-center">
            <Zap size={32} className="text-cyan-300" />
          </div>
          <div className="text-center">
            <h1 className="text-2xl font-bold text-white">SentioBot</h1>
            <p className="text-sm text-gray-500 mt-1">Nexora Electronics Support</p>
          </div>
        </div>

        {/* Card */}
        <form
          onSubmit={handleLogin}
          className="bg-gray-900 rounded-2xl p-8 border border-gray-800 space-y-5"
        >
          <div className="space-y-1">
            <label className="text-xs font-medium text-gray-400 uppercase tracking-wider">Username</label>
            <input
              type="text"
              value={username}
              onChange={(e) => setUsername(e.target.value)}
              placeholder="alice"
              required
              className="w-full bg-gray-800 border border-gray-700 rounded-lg px-4 py-3 text-sm text-white placeholder-gray-600 outline-none focus:border-cyan-600 transition-colors"
            />
          </div>

          <div className="space-y-1">
            <label className="text-xs font-medium text-gray-400 uppercase tracking-wider">Password</label>
            <input
              type="password"
              value={password}
              onChange={(e) => setPassword(e.target.value)}
              placeholder="••••••••"
              required
              className="w-full bg-gray-800 border border-gray-700 rounded-lg px-4 py-3 text-sm text-white placeholder-gray-600 outline-none focus:border-cyan-600 transition-colors"
            />
          </div>

          {error && (
            <p className="text-red-400 text-sm bg-red-950 border border-red-800 rounded-lg px-3 py-2">
              {error}
            </p>
          )}

          <button
            type="submit"
            disabled={loading}
            className="w-full py-3 rounded-lg bg-cyan-600 hover:bg-cyan-500 disabled:opacity-50 disabled:cursor-not-allowed text-white font-semibold text-sm transition-colors"
          >
            {loading ? "Signing in…" : "Sign In"}
          </button>

          <div className="text-xs text-gray-600 text-center space-y-1 pt-2 border-t border-gray-800">
            <p>Demo accounts: alice / password123</p>
            <p>or bob / password456</p>
          </div>
        </form>
      </div>
    </div>
  );
}
