"use client";

import { useEffect, useState } from "react";

interface Trade {
  id:                  string;
  timestamp:           string;
  game_id:             string;
  target_team:         string;
  action:              string;
  market_implied_prob: number;
  model_implied_prob:  number;
  stake_amount:        number;
  status:              string;
  pnl:                 number | null;
}

function fmt(dt: string) {
  return new Date(dt).toLocaleString(undefined, {
    month: "short", day: "numeric",
    hour: "2-digit", minute: "2-digit",
  });
}

export default function Dashboard() {
  const [trades, setTrades]         = useState<Trade[]>([]);
  const [loading, setLoading]       = useState(true);
  const [lastUpdated, setLastUpdated] = useState<Date | null>(null);
  const [connected, setConnected]   = useState(false);

  useEffect(() => {
    const fetchTrades = async () => {
      try {
        const res  = await fetch("/api/trades");
        const data = await res.json();
        setTrades(data);
        setLastUpdated(new Date());
        setConnected(true);
      } catch {
        setConnected(false);
      } finally {
        setLoading(false);
      }
    };

    fetchTrades();
    const interval = setInterval(fetchTrades, 3000);
    return () => clearInterval(interval);
  }, []);

  // ── Derived stats ──
  const won        = trades.filter(t => t.status === "WON").length;
  const lost       = trades.filter(t => t.status === "LOST").length;
  const open       = trades.filter(t => t.status === "OPEN").length;
  const resolved   = won + lost;
  const winRate    = resolved > 0 ? (won / resolved) * 100 : 0;
  const totalPnl   = trades.reduce((s, t) => s + (t.pnl ?? 0), 0);
  const totalStaked = trades.reduce((s, t) => s + t.stake_amount, 0);
  const avgEdge    = trades.length > 0
    ? trades.reduce((s, t) => s + Math.abs(t.model_implied_prob - t.market_implied_prob), 0) / trades.length * 100
    : null;

  const openTrades = trades.filter(t => t.status === "OPEN");

  return (
    <main className="min-h-screen bg-gray-950 text-gray-100 font-mono">

      {/* ── Sticky header ── */}
      <div className="sticky top-0 z-10 border-b border-gray-800 bg-gray-950/90 backdrop-blur">
        <div className="max-w-7xl mx-auto px-6 py-4 flex items-center justify-between">
          <div>
            <h1 className="text-lg font-bold text-green-400 tracking-widest">NBA SHADOW TRADER</h1>
            <p className="text-xs text-gray-600 mt-0.5">Quantitative prediction market engine · paper trading only</p>
          </div>
          <div className="flex items-center gap-4 text-xs">
            {lastUpdated && (
              <span className="text-gray-600 hidden sm:block">
                Updated {lastUpdated.toLocaleTimeString()}
              </span>
            )}
            <span className={`flex items-center gap-1.5 font-bold ${connected ? "text-green-400" : "text-red-400"}`}>
              <span className={`w-2 h-2 rounded-full ${connected ? "bg-green-400 animate-pulse" : "bg-red-500"}`} />
              {connected ? "LIVE" : "OFFLINE"}
            </span>
          </div>
        </div>
      </div>

      <div className="max-w-7xl mx-auto px-6 py-8 space-y-10">

        {/* ── Stats bar ── */}
        <div className="grid grid-cols-2 md:grid-cols-5 gap-4">
          <StatCard
            label="Simulated PnL"
            value={`${totalPnl >= 0 ? "+" : ""}$${totalPnl.toFixed(2)}`}
            color={totalPnl >= 0 ? "text-green-400" : "text-red-400"}
            sub={`$${totalStaked.toFixed(0)} total staked`}
          />
          <StatCard
            label="Win Rate"
            value={`${winRate.toFixed(1)}%`}
            color="text-blue-400"
            sub={resolved > 0 ? `${won}W / ${lost}L` : "No resolved bets yet"}
          />
          <StatCard
            label="Open Positions"
            value={String(open)}
            color="text-yellow-400"
            sub={open > 0 ? "awaiting resolution" : "none active"}
          />
          <StatCard
            label="Total Trades"
            value={String(trades.length)}
            color="text-gray-300"
          />
          <StatCard
            label="Avg Edge"
            value={avgEdge != null ? `${avgEdge.toFixed(1)}%` : "—"}
            color="text-purple-400"
            sub="model vs market"
          />
        </div>

        {/* ── Active Positions ── */}
        {(loading || openTrades.length > 0) && (
          <section>
            <SectionHeader
              label="Active Positions"
              count={openTrades.length}
              pulse
              color="text-yellow-400"
            />
            {loading ? (
              <div className="text-gray-600 text-sm py-6">Loading…</div>
            ) : openTrades.length === 0 ? null : (
              <div className="grid grid-cols-1 md:grid-cols-2 xl:grid-cols-3 gap-4">
                {openTrades.map(t => (
                  <ActivePositionCard key={t.id} trade={t} />
                ))}
              </div>
            )}
          </section>
        )}

        {/* ── Trade Feed ── */}
        <section>
          <SectionHeader label="Trade History" count={trades.length} color="text-gray-400" />
          {loading ? (
            <div className="text-gray-600 text-sm py-12 text-center">Loading…</div>
          ) : trades.length === 0 ? (
            <div className="text-gray-600 text-sm py-12 text-center border border-gray-800 rounded-lg">
              No trades logged yet — waiting for edge signal…
            </div>
          ) : (
            <div className="overflow-x-auto rounded-lg border border-gray-800">
              <table className="w-full text-xs">
                <thead>
                  <tr className="bg-gray-900 text-gray-500 text-left uppercase tracking-wider">
                    <th className="px-4 py-3">Time</th>
                    <th className="px-4 py-3">Market</th>
                    <th className="px-4 py-3">Action</th>
                    <th className="px-4 py-3">Stake</th>
                    <th className="px-4 py-3">Market %</th>
                    <th className="px-4 py-3">Model %</th>
                    <th className="px-4 py-3">Edge</th>
                    <th className="px-4 py-3">Status</th>
                    <th className="px-4 py-3 text-right">PnL</th>
                  </tr>
                </thead>
                <tbody>
                  {trades.map(t => {
                    const edge = (t.model_implied_prob - t.market_implied_prob) * 100;
                    return (
                      <tr
                        key={t.id}
                        className="border-t border-gray-800/60 hover:bg-gray-900/50 transition-colors"
                      >
                        <td className="px-4 py-3 text-gray-500 whitespace-nowrap">{fmt(t.timestamp)}</td>
                        <td className="px-4 py-3 max-w-xs truncate text-gray-300" title={t.target_team}>
                          {t.target_team}
                        </td>
                        <td className="px-4 py-3">
                          <span className={`font-semibold ${t.action !== "BUY_AWAY" ? "text-green-400" : "text-orange-400"}`}>
                            {t.action}
                          </span>
                        </td>
                        <td className="px-4 py-3 text-gray-400">${t.stake_amount.toFixed(0)}</td>
                        <td className="px-4 py-3 text-gray-300">{(t.market_implied_prob * 100).toFixed(1)}%</td>
                        <td className="px-4 py-3 text-gray-300">{(t.model_implied_prob  * 100).toFixed(1)}%</td>
                        <td className={`px-4 py-3 font-bold ${edge >= 0 ? "text-green-400" : "text-red-400"}`}>
                          {edge >= 0 ? "+" : ""}{edge.toFixed(1)}%
                        </td>
                        <td className="px-4 py-3">
                          <StatusBadge status={t.status} />
                        </td>
                        <td className={`px-4 py-3 text-right font-semibold ${(t.pnl ?? 0) >= 0 ? "text-green-400" : "text-red-400"}`}>
                          {t.pnl != null ? `${t.pnl >= 0 ? "+" : ""}$${t.pnl.toFixed(2)}` : "—"}
                        </td>
                      </tr>
                    );
                  })}
                </tbody>
              </table>
            </div>
          )}
        </section>

      </div>
    </main>
  );
}

// ── Sub-components ──

function StatCard({
  label, value, color, sub,
}: {
  label: string; value: string; color: string; sub?: string;
}) {
  return (
    <div className="bg-gray-900 border border-gray-800 rounded-lg p-4">
      <p className="text-gray-500 text-xs uppercase tracking-widest mb-1">{label}</p>
      <p className={`text-2xl font-bold ${color}`}>{value}</p>
      {sub && <p className="text-gray-600 text-xs mt-1">{sub}</p>}
    </div>
  );
}

function SectionHeader({
  label, count, pulse, color,
}: {
  label: string; count: number; pulse?: boolean; color: string;
}) {
  return (
    <h2 className={`text-xs font-bold uppercase tracking-widest mb-4 flex items-center gap-2 ${color}`}>
      {pulse && <span className="w-2 h-2 rounded-full bg-yellow-400 animate-pulse" />}
      {label}
      <span className="text-gray-600 font-normal">({count})</span>
    </h2>
  );
}

function ActivePositionCard({ trade }: { trade: Trade }) {
  const edge      = (trade.model_implied_prob - trade.market_implied_prob) * 100;
  const isBuyHome = trade.action !== "BUY_AWAY";
  return (
    <div className="bg-gray-900 border border-yellow-900/30 rounded-lg p-4 space-y-3 hover:border-yellow-700/40 transition-colors">
      <div className="flex items-start justify-between gap-3">
        <p className="text-gray-200 text-sm font-semibold leading-snug flex-1 min-w-0 truncate" title={trade.target_team}>
          {trade.target_team}
        </p>
        <StatusBadge status={trade.status} />
      </div>

      <div className="grid grid-cols-2 gap-x-6 gap-y-2 text-xs">
        <DataPoint label="Action">
          <span className={`font-bold ${isBuyHome ? "text-green-400" : "text-orange-400"}`}>
            {trade.action}
          </span>
        </DataPoint>
        <DataPoint label="Stake">
          <span className="text-gray-200 font-semibold">${trade.stake_amount.toFixed(0)} USDC</span>
        </DataPoint>
        <DataPoint label="Market odds">
          <span className="text-gray-300">{(trade.market_implied_prob * 100).toFixed(1)}%</span>
        </DataPoint>
        <DataPoint label="Model confidence">
          <span className="text-gray-300">{(trade.model_implied_prob * 100).toFixed(1)}%</span>
        </DataPoint>
      </div>

      <div className="flex items-center justify-between pt-2 border-t border-gray-800">
        <span className="text-gray-600 text-xs">{fmt(trade.timestamp)}</span>
        <span className={`text-sm font-bold ${edge >= 0 ? "text-green-400" : "text-red-400"}`}>
          EDGE {edge >= 0 ? "+" : ""}{edge.toFixed(1)}%
        </span>
      </div>
    </div>
  );
}

function DataPoint({ label, children }: { label: string; children: React.ReactNode }) {
  return (
    <div>
      <span className="text-gray-600 block">{label}</span>
      <div className="mt-0.5">{children}</div>
    </div>
  );
}

function StatusBadge({ status }: { status: string }) {
  const styles: Record<string, string> = {
    OPEN: "bg-yellow-900/50 text-yellow-300 border border-yellow-800/50",
    WON:  "bg-green-900/50  text-green-300  border border-green-800/50",
    LOST: "bg-red-900/50    text-red-300    border border-red-800/50",
  };
  return (
    <span className={`px-2 py-0.5 rounded text-xs font-bold whitespace-nowrap ${styles[status] ?? "bg-gray-700 text-gray-300"}`}>
      {status}
    </span>
  );
}
